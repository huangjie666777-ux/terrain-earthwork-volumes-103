"""Block-wise before/after elevation differencing and cut/fill integration."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio

from .grid import Grid, cell_areas
from .mosaic import build_mosaic
from .storage import DatasetMeta

DIFF_NODATA = np.float32(-9999.0)
BLOCK_ROWS = 256


class CutfillError(ValueError):
    """Raised when cut/fill computation cannot proceed."""


def _block_bounds(grid: Grid, row_start: int, row_end: int):
    left = grid.transform.c
    right = left + grid.resolution * grid.width
    top = grid.transform.f - row_start * grid.resolution
    bottom = grid.transform.f - row_end * grid.resolution
    return (left, bottom, right, top)


def compute_cutfill(
    *,
    region,
    grid: Grid,
    before_datasets: list[DatasetMeta],
    after_datasets: list[DatasetMeta],
    dataset_dir: str,
    output_path: str,
    is_cancelled: Callable[[], bool] | None = None,
    progress: Callable[[float], None] | None = None,
    block_rows: int = BLOCK_ROWS,
) -> dict:
    """Difference two published versions over the region and write a GeoTIFF.

    Only cells valid in both versions contribute volume; every other cell is
    reported through its missing intersection area and written as NoData.
    """
    is_cancelled = is_cancelled or (lambda: False)
    progress = progress or (lambda _value: None)
    block_rows = max(1, min(int(block_rows), grid.height))

    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": float(DIFF_NODATA),
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    output_path = str(output_path)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=".cutfill-", suffix=".tif", dir=os.path.dirname(output_path) or "."
    )
    os.close(tmp_fd)
    totals = {
        "total_area": 0.0,
        "valid_area": 0.0,
        "missing_area": 0.0,
        "cut_m3": 0.0,
        "fill_m3": 0.0,
    }
    try:
        with rasterio.open(tmp_name, "w", **profile) as dst:
            for row_start in range(0, grid.height, block_rows):
                if is_cancelled():
                    raise CutfillInterrupted("job cancelled")
                row_end = min(row_start + block_rows, grid.height)
                shape = (row_end - row_start, grid.width)
                block_transform = rasterio.transform.from_origin(
                    grid.transform.c,
                    grid.transform.f - row_start * grid.resolution,
                    grid.resolution,
                    grid.resolution,
                )
                bounds = _block_bounds(grid, row_start, row_end)
                before = build_mosaic(
                    before_datasets,
                    shape,
                    block_transform,
                    bounds,
                    dataset_dir,
                    dst_crs=grid.crs,
                )
                after = build_mosaic(
                    after_datasets,
                    shape,
                    block_transform,
                    bounds,
                    dataset_dir,
                    dst_crs=grid.crs,
                )
                areas = cell_areas(region, grid, row_start, row_end)
                touched = areas > 0.0
                if not touched.any():
                    dst.write(
                        np.full(shape, DIFF_NODATA, dtype=np.float32),
                        1,
                        window=rasterio.windows.Window(0, row_start, grid.width, shape[0]),
                    )
                    progress(row_end / grid.height)
                    continue
                valid = touched & ~np.asarray(before.mask) & ~np.asarray(after.mask)
                totals["total_area"] += float(areas[touched].sum())
                totals["valid_area"] += float(areas[valid].sum())

                diff = np.full(shape, DIFF_NODATA, dtype=np.float32)
                before_v = np.asarray(before.data, dtype=np.float64)
                after_v = np.asarray(after.data, dtype=np.float64)
                both = np.isfinite(before_v) & np.isfinite(after_v) & valid
                delta = np.zeros(shape, dtype=np.float64)
                delta[both] = after_v[both] - before_v[both]
                diff[both] = delta[both].astype(np.float32)
                fill_cells = both & (delta > 0.0)
                cut_cells = both & (delta < 0.0)
                totals["fill_m3"] += float((delta[fill_cells] * areas[fill_cells]).sum())
                totals["cut_m3"] += float((-delta[cut_cells] * areas[cut_cells]).sum())
                dst.write(
                    diff,
                    1,
                    window=rasterio.windows.Window(0, row_start, grid.width, shape[0]),
                )
                progress(row_end / grid.height)
        os.replace(tmp_name, output_path)
    except CutfillInterrupted:
        Path(tmp_name).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
        raise
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
        raise

    totals["missing_area"] = totals["total_area"] - totals["valid_area"]
    totals["net_m3"] = totals["fill_m3"] - totals["cut_m3"]
    coverage = (
        totals["valid_area"] / totals["total_area"] if totals["total_area"] > 0.0 else 0.0
    )
    totals["coverage"] = coverage
    for key, value in totals.items():
        if key != "coverage":
            totals[key] = float(value)
    if not math.isfinite(totals["cut_m3"]) or not math.isfinite(totals["fill_m3"]):
        raise CutfillError("non-finite volumes accumulated")
    return totals


class CutfillInterrupted(RuntimeError):
    """Raised when a cooperative cancellation flag is observed."""
