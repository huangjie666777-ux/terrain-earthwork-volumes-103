"""Persistent cut/fill job store with bounded background execution."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .config import settings
from .cutfill import CutfillInterrupted, compute_cutfill
from .grid import aligned_grid, parse_region
from .storage import VersionError

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"

TERMINAL_STATUSES = {COMPLETED, FAILED, CANCELLED, INTERRUPTED}


class JobError(ValueError):
    """Raised for invalid job requests or state transitions."""


class JobStore:
    """JSON-backed job metadata with per-job atomic updates."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.data_dir
        self.jobs_dir = self.root / "jobs"
        self.results_dir = self.root / "results"
        self.tmp_dir = self.root / "tmp"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def result_path(self, job_id: str) -> Path:
        return self.results_dir / f"{job_id}.tif"

    def create(self, request: dict) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = time.utcnow().isoformat(timespec="seconds") + "Z"
        record = {
            "id": job_id,
            "status": PENDING,
            "request": request,
            "progress": 0.0,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "result": None,
        }
        self._write(record)
        return record

    def _write(self, record: dict) -> None:
        target = self._path(record["id"])
        tmp = target.with_name(f".{record['id']}.{uuid.uuid4().hex}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)

    def get(self, job_id: str) -> dict | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def list(self) -> list[dict]:
        records = []
        for path in self.jobs_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    records.append(json.load(fh))
            except (OSError, json.JSONDecodeError):
                continue
        records.sort(key=lambda item: item.get("created_at", ""))
        return records

    def update(self, job_id: str, **changes) -> dict | None:
        with self._lock:
            record = self.get(job_id)
            if record is None:
                return None
            record.update(changes)
            record["updated_at"] = time.utcnow().isoformat(timespec="seconds") + "Z"
            self._write(record)
            return record

    def mark_unfinished_interrupted(self) -> int:
        """Call at startup: pending/running jobs never finish after restart."""
        count = 0
        for record in self.list():
            if record["status"] in (PENDING, RUNNING):
                self.update(
                    record["id"],
                    status=INTERRUPTED,
                    progress=record.get("progress", 0.0),
                    error="process restarted before the job completed",
                )
                count += 1
        return count


class JobManager:
    def __init__(self, store: JobStore, registry, slots: threading.BoundedSemaphore | None = None) -> None:
        self.store = store
        self.registry = registry
        self._slots = slots or threading.BoundedSemaphore(settings.max_concurrent_cutfill)
        self._cancel = {}
        self._workers = {}
        self._guard = threading.Lock()

    def submit(
        self,
        *,
        before_version: str,
        after_version: str,
        region_geojson: dict,
        epsg: int,
        resolution: float,
    ) -> dict:
        region = parse_region(region_geojson)
        grid = aligned_grid(region, epsg, resolution, settings.max_cutfill_cells)
        # Resolve both published versions up front; unknown versions are rejected
        # before a job record is created.
        before_datasets = self.registry.version(before_version)
        after_datasets = self.registry.version(after_version)
        request = {
            "before_version": before_version,
            "after_version": after_version,
            "region": region_geojson,
            "epsg": int(epsg),
            "resolution": float(resolution),
            "grid": {
                "width": grid.width,
                "height": grid.height,
                "crs": grid.crs,
                "resolution": grid.resolution,
                "bounds": list(grid.bounds),
            },
        }
        record = self.store.create(request)
        event = threading.Event()
        self._cancel[record["id"]] = event
        worker = threading.Thread(
            target=self._run,
            args=(record["id"], before_datasets, after_datasets, event),
            name=f"cutfill-{record['id']}",
            daemon=True,
        )
        self._workers[record["id"]] = worker
        worker.start()
        return record

    def _run(self, job_id: str, before_datasets, after_datasets, cancel_event) -> None:
        try:
            with self._slots:
                if cancel_event.is_set():
                    self.store.update(job_id, status=CANCELLED, error="cancelled before execution")
                    return
                record = self.store.get(job_id)
                if record is None or record["status"] in TERMINAL_STATUSES:
                    return
                self.store.update(job_id, status=RUNNING)
                request = record["request"]
                region = parse_region(request["region"])
                grid_info = request["grid"]
                from affine import Affine

                left, _bottom, _right, top = grid_info["bounds"][0], grid_info["bounds"][1], grid_info["bounds"][2], grid_info["bounds"][3]
                resolution = grid_info["resolution"]
                transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)
                from .grid import Grid

                grid = Grid(
                    transform=transform,
                    width=grid_info["width"],
                    height=grid_info["height"],
                    crs=grid_info["crs"],
                    resolution=resolution,
                )
                output = self.store.result_path(job_id)
                metrics = compute_cutfill(
                    region=region,
                    grid=grid,
                    before_datasets=before_datasets,
                    after_datasets=after_datasets,
                    dataset_dir=str(self.registry.dataset_dir),
                    output_path=str(output),
                    is_cancelled=cancel_event.is_set,
                    progress=lambda value: self.store.update(job_id, progress=round(float(value), 6)),
                    block_rows=settings.cutfill_block_rows,
                )
            current = self.store.get(job_id)
            if cancel_event.is_set() or (current and current["status"] == CANCELLED):
                output.unlink(missing_ok=True)
                self.store.update(job_id, status=CANCELLED, error="cancelled after completion was requested")
                return
            self.store.update(
                job_id,
                status=COMPLETED,
                progress=1.0,
                result={
                    **metrics,
                    "download_url": f"/cutfill/jobs/{job_id}/difference.tif",
                },
            )
        except CutfillInterrupted as exc:
            self.store.update(job_id, status=CANCELLED, error=str(exc))
        except VersionError as exc:
            self.store.update(job_id, status=FAILED, error=str(exc))
        except Exception as exc:
            self.store.update(job_id, status=FAILED, error=str(exc))

    def cancel(self, job_id: str) -> dict:
        record = self.store.get(job_id)
        if record is None:
            raise JobError("unknown job")
        if record["status"] in TERMINAL_STATUSES:
            return record
        event = self._cancel.get(job_id)
        if event is not None:
            event.set()
        status = CANCELLED if record["status"] == PENDING else record["status"]
        updated = self.store.update(job_id, status=status, error="cancel requested")
        if record["status"] == PENDING:
            return updated
        # RUNNING jobs are finalized cooperatively by the worker.
        return self.store.get(job_id)

    def get_download(self, job_id: str) -> Path:
        record = self.store.get(job_id)
        if record is None:
            raise JobError("unknown job")
        if record["status"] != COMPLETED:
            raise JobError(f"job is not complete: {record['status']}")
        path = self.store.result_path(job_id)
        if not path.exists():
            raise JobError("result raster is missing from storage")
        return path
