"""FastAPI HTTP interface for dataset, version, and hillshade tile access."""

from __future__ import annotations

import threading

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .config import ensure_dirs, settings
from .grid import GeometryError
from .jobs import JobError, JobManager, JobStore
from .storage import Registry, UploadError, VersionError
from .tiles import TileCache, TileError, TileRenderer

ensure_dirs()
registry = Registry()
job_store = JobStore()
job_store.mark_unfinished_interrupted()
job_manager = JobManager(job_store, registry)
renderer = TileRenderer(
    dataset_dir=str(registry.dataset_dir),
    cache=TileCache(),
    slots=threading.BoundedSemaphore(settings.max_concurrent_jobs),
)
app = FastAPI(title="Terrain Hillshade Tiles", version="1.0.0")


class PublishRequest(BaseModel):
    version: str = Field(min_length=1, max_length=128)
    datasets: list[str] = Field(min_length=1, max_length=512)


class CutfillRequest(BaseModel):
    before_version: str = Field(min_length=1, max_length=128)
    after_version: str = Field(min_length=1, max_length=128)
    # GeoJSON Polygon/MultiPolygon geometry object in the given UTM metres CRS.
    region: dict
    epsg: int = Field(description="UTM EPSG code, e.g. 32647")
    resolution: float = Field(gt=0, description="regular grid cell size in metres")


@app.exception_handler(UploadError)
def upload_handler(_request, exc: UploadError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(VersionError)
def version_handler(_request, exc: VersionError):
    status_code = 409 if "already exists" in str(exc) else 400
    if str(exc).startswith("unknown version") or "missing from storage" in str(exc):
        status_code = 404
    return JSONResponse(status_code=status_code, content={"error": str(exc)})


@app.exception_handler(TileError)
def tile_handler(_request, exc: TileError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(GeometryError)
def geometry_handler(_request, exc: GeometryError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(JobError)
def job_handler(_request, exc: JobError):
    status_code = 404 if "unknown job" in str(exc) else 409
    return JSONResponse(status_code=status_code, content={"error": str(exc)})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "max_upload_bytes": settings.max_upload_bytes,
        "max_zoom": settings.max_zoom,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "cache_tiles": len(renderer.cache),
    }


@app.post("/datasets")
def upload_dataset(file: UploadFile = File(...)):
    payload = file.file.read(settings.max_upload_bytes + 1)
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="file exceeds upload size limit")
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        meta = registry.ingest(file.filename or "upload.tif", payload)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": meta.id, "metadata": meta.to_dict() | {"id": meta.id}}


@app.get("/datasets")
def list_datasets():
    items = [item.to_dict() for item in registry.list_datasets()]
    return {"datasets": items}


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    meta = registry.get_dataset(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return {"id": meta.id, "metadata": meta.to_dict() | {"id": meta.id}}


@app.post("/versions")
def publish_version(request: PublishRequest):
    try:
        items = registry.publish(request.version, request.datasets)
    except VersionError as exc:
        if "already exists" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "version": request.version,
        "datasets": [item.to_dict() for item in items],
        "dataset_ids": [item.id for item in items],
    }


@app.get("/versions/{version}")
def get_version(version: str):
    try:
        items = registry.version(version)
    except VersionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "version": version,
        "dataset_ids": [item.id for item in items],
        "datasets": [item.to_dict() for item in items],
    }


@app.get("/tiles/{version}/{z}/{x}/{y}.png")
def tile(
    version: str,
    z: int,
    x: int,
    y: int,
    azimuth: float = Query(default=settings.default_azimuth),
    altitude: float = Query(default=settings.default_altitude),
):
    try:
        datasets = registry.version(version)
        png = renderer.render(version, datasets, z, x, y, azimuth, altitude)
    except VersionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.post("/cutfill/jobs", status_code=202)
def submit_cutfill(request: CutfillRequest):
    try:
        record = job_manager.submit(
            before_version=request.before_version,
            after_version=request.after_version,
            region_geojson=request.region,
            epsg=request.epsg,
            resolution=request.resolution,
        )
    except VersionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeometryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _job_view(record)


def _job_view(record: dict) -> dict:
    return {
        "id": record["id"],
        "status": record["status"],
        "progress": record.get("progress", 0.0),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "error": record.get("error"),
        "request": record.get("request"),
        "result": record.get("result"),
    }


@app.get("/cutfill/jobs")
def list_cutfill_jobs():
    return {"jobs": [_job_view(record) for record in job_store.list()]}


@app.get("/cutfill/jobs/{job_id}")
def get_cutfill_job(job_id: str):
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return _job_view(record)


@app.post("/cutfill/jobs/{job_id}/cancel", status_code=200)
def cancel_cutfill_job(job_id: str):
    try:
        record = job_manager.cancel(job_id)
    except JobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _job_view(record)


@app.get("/cutfill/jobs/{job_id}/difference.tif")
def download_cutfill(job_id: str):
    try:
        path = job_manager.get_download(job_id)
    except JobError as exc:
        status_code = 404 if "unknown job" in str(exc) or "missing" in str(exc) else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="image/tiff",
        filename=f"difference-{job_id}.tif",
        headers={"Cache-Control": "private, max-age=3600"},
    )
