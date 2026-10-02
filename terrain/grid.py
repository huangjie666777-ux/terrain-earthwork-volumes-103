"""UTM validation, resolution-snapped grids, and polygon intersection areas."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.prepared import prep


class GeometryError(ValueError):
    """Raised for an invalid work-area geometry or grid request."""


def validate_utm_epsg(epsg: int) -> int:
    if not isinstance(epsg, int) or isinstance(epsg, bool):
        raise GeometryError("epsg must be an integer UTM EPSG code")
    if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
        raise GeometryError(
            f"EPSG:{epsg} is not a UTM zone "
            "(expected EPSG:32601-32660 or EPSG:32701-32760)"
        )
    return epsg


def _all_coords(geom) -> list[float]:
    values: list[float] = []
    if geom.geom_type == "Polygon":
        rings = [geom.exterior, *geom.interiors]
    else:
        rings = []
        for part in geom.geoms:
            rings.extend([part.exterior, *part.interiors])
    for ring in rings:
        for x_coord, y_coord in ring.coords:
            values.extend((x_coord, y_coord))
    return values


def parse_region(geojson: dict):
    if not isinstance(geojson, dict):
        raise GeometryError("region must be a GeoJSON Polygon or MultiPolygon object")
    try:
        geom = shape(geojson)
    except Exception as exc:
        raise GeometryError(f"invalid GeoJSON geometry: {exc}") from exc
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise GeometryError("region must be a Polygon or MultiPolygon")
    if geom.is_empty or not geom.is_valid:
        raise GeometryError("region is empty or self-intersecting")
    if not math.isfinite(geom.area) or geom.area <= 0.0:
        raise GeometryError("region must have positive finite area")
    coords = np.asarray(_all_coords(geom), dtype=np.float64)
    if not np.isfinite(coords).all():
        raise GeometryError("region contains non-finite coordinates")
    return geom


@dataclass(frozen=True)
class Grid:
    transform: Affine
    width: int
    height: int
    crs: str
    resolution: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.transform.c
        top = self.transform.f
        return (
            left,
            top - self.resolution * self.height,
            left + self.resolution * self.width,
            top,
        )


def aligned_grid(region, epsg: int, resolution: float, max_cells: int) -> Grid:
    """Snap the region bbox outward to integer multiples of resolution."""
    validate_utm_epsg(epsg)
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
        raise GeometryError("resolution must be a positive number of metres")
    resolution = float(resolution)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise GeometryError("resolution must be a positive finite number of metres")
    minx, miny, maxx, maxy = region.bounds
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        raise GeometryError("region bounds contain non-finite values")
    if maxx <= minx or maxy <= miny:
        raise GeometryError("region bounds are invalid")
    col0 = math.floor(minx / resolution - 1e-9)
    col1 = math.ceil(maxx / resolution + 1e-9)
    row_top = math.ceil(maxy / resolution + 1e-9)
    row_bottom = math.floor(miny / resolution - 1e-9)
    left = col0 * resolution
    top = row_top * resolution
    width = col1 - col0
    height = row_top - row_bottom
    if width <= 0 or height <= 0:
        raise GeometryError("aligned grid is empty")
    if width * height > max_cells:
        raise GeometryError(
            f"grid would contain {width * height} cells; limit is {max_cells}"
        )
    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)
    return Grid(transform, width, height, f"EPSG:{epsg}", resolution)


def cell_areas(region, grid: Grid, row_start: int, row_end: int) -> np.ndarray:
    """Intersection area per cell; holes are subtracted by polygon clipping.

    Cells whose center lies inside the region and that do not touch a polygon
    ring receive the full cell area. Ring-touching cells are clipped exactly.
    """
    rows = row_end - row_start
    areas = np.zeros((rows, grid.width), dtype=np.float64)
    strip_transform = Affine(
        grid.resolution,
        0.0,
        grid.transform.c,
        0.0,
        -grid.resolution,
        grid.transform.f - row_start * grid.resolution,
    )
    touched = rasterize(
        ((region, 1),),
        out_shape=(rows, grid.width),
        transform=strip_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    ).astype(bool)
    if not touched.any():
        return areas
    center_in = rasterize(
        ((region, 1),),
        out_shape=(rows, grid.width),
        transform=strip_transform,
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    ).astype(bool)
    boundary = rasterize(
        ((region.boundary, 1),),
        out_shape=(rows, grid.width),
        transform=strip_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    ).astype(bool)
    full = center_in & ~boundary
    areas[full] = grid.resolution * grid.resolution

    exact_rows, exact_cols = np.nonzero(touched & ~full)
    prepared = prep(region)
    cells = [
        box(
            grid.transform.c + int(col) * grid.resolution,
            grid.transform.f - (row_start + int(local_row) + 1) * grid.resolution,
            grid.transform.c + (int(col) + 1) * grid.resolution,
            grid.transform.f - (row_start + int(local_row)) * grid.resolution,
        )
        for local_row, col in zip(exact_rows.tolist(), exact_cols.tolist())
    ]
    for index, cell in enumerate(cells):
        if prepared.intersects(cell):
            areas[exact_rows[index], exact_cols[index]] = region.intersection(cell).area
    return areas


def region_in_crs(region, src_crs: str, dst_epsg: int):
    if str(src_crs).upper() == f"EPSG:{dst_epsg}":
        return region
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, f"EPSG:{dst_epsg}", always_xy=True)
    return shapely_transform(transformer.transform, region)


def crs_utm(epsg: int) -> CRS:
    return CRS.from_epsg(validate_utm_epsg(epsg))
