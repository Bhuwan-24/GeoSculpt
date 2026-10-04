# src/dw/core/dem.py

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.transform import Affine
from pyproj import CRS


LOGGER = logging.getLogger("depthwizard.dem")


# ---------------------------------------------------------------------
# FABDEM configuration
# ---------------------------------------------------------------------

FABDEM_BASE_URL = (
    "https://huggingface.co/buckets/links-ads/fabdem/resolve/tiles"
)

FABDEM_VERSION = "V1-2"

FABDEM_NODATA = -9999.0

FABDEM_SOURCE_CRS = "EPSG:4326"

# FABDEM v1.2 uses approximately 1 degree x 1 degree tiles.
TILE_SIZE_DEGREES = 1

# Groups are 10° x 10°.
GROUP_SIZE_DEGREES = 10

DOWNLOAD_TIMEOUT_SECONDS = 120

USER_AGENT = (
    "DepthWizard/0.1 "
    "(prototype FABDEM client)"
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FabdemTile:
    """
    One 1° x 1° FABDEM tile.

    lat/lon identify the south-west corner of the tile.
    """

    lat: float
    lon: float

    @property
    def tile_name(self) -> str:
        return (
            f"{_lat_code(self.lat)}"
            f"{_lon_code(self.lon)}"
            f"_FABDEM_{FABDEM_VERSION}.tif"
        )

    @property
    def group_name(self) -> str:
        return _group_name(
            self.lat,
            self.lon,
        )

    @property
    def url(self) -> str:
        return (
            f"{FABDEM_BASE_URL}/"
            f"{self.group_name}/"
            f"{self.tile_name}"
        )


# ---------------------------------------------------------------------
# Coordinate / naming helpers
# ---------------------------------------------------------------------

def _lat_code(lat: float) -> str:
    """
    Convert a latitude into the FABDEM tile code.

    Examples:

        52.0  -> N52
        52.9  -> N52
        -0.2  -> S01
        -52.7 -> S53
    """
    lat_deg = math.floor(float(lat))

    if lat_deg >= 0:
        return f"N{lat_deg:02d}"

    return f"S{abs(lat_deg):02d}"


def _lon_code(lon: float) -> str:
    """
    Convert a longitude into the FABDEM tile code.

    Examples:

        13.0   -> E013
        13.9   -> E013
        -0.2   -> W001
        -13.7  -> W014
    """
    lon_deg = math.floor(float(lon))

    if lon_deg >= 0:
        return f"E{lon_deg:03d}"

    return f"W{abs(lon_deg):03d}"


def _group_name(
    lat: float,
    lon: float,
) -> str:
    """
    Build the 10° x 10° FABDEM group name.

    Example:

        tile N52E013
        -> group N50E010-N60E020
    """
    tile_lat = math.floor(float(lat))
    tile_lon = math.floor(float(lon))

    south_lat = math.floor(
        tile_lat / GROUP_SIZE_DEGREES
    ) * GROUP_SIZE_DEGREES

    west_lon = math.floor(
        tile_lon / GROUP_SIZE_DEGREES
    ) * GROUP_SIZE_DEGREES

    north_lat = south_lat + GROUP_SIZE_DEGREES
    east_lon = west_lon + GROUP_SIZE_DEGREES

    return (
        f"{_lat_code(south_lat)}"
        f"{_lon_code(west_lon)}-"
        f"{_lat_code(north_lat)}"
        f"{_lon_code(east_lon)}"
        f"_FABDEM_{FABDEM_VERSION}"
    )


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def _value(
    obj: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Return the first existing key.
    """
    for key in keys:
        if key in obj:
            return obj[key]

    return default


def _grid_dict(
    tiling_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Find the canonical grid description in tiling_meta.json.

    Supports the common layouts used by the prototype.
    """
    candidates = [
        tiling_meta.get("canonical_grid"),
        tiling_meta.get("canonical"),
        tiling_meta.get("grid"),
    ]

    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate

    return tiling_meta


def _affine_from_value(
    value: Any,
) -> Affine:
    """
    Convert a six-value transform representation into Affine.
    """
    if isinstance(value, Affine):
        return value

    if isinstance(value, (list, tuple)) and len(value) == 6:
        return Affine(
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
            float(value[4]),
            float(value[5]),
        )

    if isinstance(value, dict):
        return Affine(
            float(value["a"]),
            float(value["b"]),
            float(value["c"]),
            float(value["d"]),
            float(value["e"]),
            float(value["f"]),
        )

    raise ValueError(
        "Could not parse canonical transform from tiling metadata."
    )


def _extract_canonical_grid(
    tiling_meta: dict[str, Any],
) -> tuple[int, int, Affine, CRS]:
    """
    Extract canonical width, height, transform, and CRS.

    The function accepts several reasonable metadata layouts so
    dem.py stays compatible with the existing prototype tiling code.
    """
    grid = _grid_dict(
        tiling_meta
    )

    width = _value(
        grid,
        "width",
        "canonical_width",
    )

    height = _value(
        grid,
        "height",
        "canonical_height",
    )

    transform_value = _value(
        grid,
        "transform",
        "canonical_transform",
    )

    crs_value = _value(
        grid,
        "crs",
        "canonical_crs",
    )

    # If nested canonical_grid was not used, try top-level keys.
    if width is None:
        width = _value(
            tiling_meta,
            "canonical_width",
        )

    if height is None:
        height = _value(
            tiling_meta,
            "canonical_height",
        )

    if transform_value is None:
        transform_value = _value(
            tiling_meta,
            "canonical_transform",
        )

    if crs_value is None:
        crs_value = _value(
            tiling_meta,
            "canonical_crs",
        )

    if width is None or height is None:
        raise KeyError(
            "Canonical grid width/height not found in tiling_meta.json."
        )

    if transform_value is None:
        raise KeyError(
            "Canonical transform not found in tiling_meta.json."
        )

    if crs_value is None:
        raise KeyError(
            "Canonical CRS not found in tiling_meta.json."
        )

    transform = _affine_from_value(
        transform_value
    )

    crs = CRS.from_user_input(
        crs_value
    )

    return (
        int(width),
        int(height),
        transform,
        crs,
    )


# ---------------------------------------------------------------------
# Scene bounds
# ---------------------------------------------------------------------

def _scene_wgs84_bounds(
    scene_meta: dict[str, Any],
) -> tuple[float, float, float, float]:
    """
    Extract WGS84 scene bounds.

    Returns:
        left, bottom, right, top
    """
    bounds = scene_meta.get(
        "bounds_wgs84"
    )

    if not isinstance(bounds, dict):
        raise KeyError(
            "scene_meta.json does not contain bounds_wgs84."
        )

    left = float(
        bounds["left"]
    )

    bottom = float(
        bounds["bottom"]
    )

    right = float(
        bounds["right"]
    )

    top = float(
        bounds["top"]
    )

    return (
        left,
        bottom,
        right,
        top,
    )


# ---------------------------------------------------------------------
# FABDEM tile discovery
# ---------------------------------------------------------------------

def _tile_indices_for_bounds(
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> list[FabdemTile]:
    """
    Determine all 1° FABDEM tiles intersecting the scene.

    np.nextafter() prevents an exact boundary such as 14.0°
    from incorrectly pulling in an unnecessary extra tile.
    """
    if right <= left or top <= bottom:
        raise ValueError(
            "Invalid WGS84 bounds."
        )

    # Treat the right/top bound as an exclusive raster edge.
    right_inside = np.nextafter(
        right,
        -np.inf,
    )

    top_inside = np.nextafter(
        top,
        -np.inf,
    )

    min_lon = math.floor(
        left
    )

    max_lon = math.floor(
        right_inside
    )

    min_lat = math.floor(
        bottom
    )

    max_lat = math.floor(
        top_inside
    )

    tiles: list[FabdemTile] = []

    for lat in range(
        min_lat,
        max_lat + 1,
    ):
        for lon in range(
            min_lon,
            max_lon + 1,
        ):
            tiles.append(
                FabdemTile(
                    lat=float(lat),
                    lon=float(lon),
                )
            )

    return tiles


# ---------------------------------------------------------------------
# HTTP download
# ---------------------------------------------------------------------

def _download_file(
    url: str,
    destination: Path,
) -> None:
    """
    Download one FABDEM tile to a local file.

    Downloads are written to a temporary .part file first so an
    interrupted download does not look like a valid cache entry.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        suffix=".part",
        dir=str(destination.parent),
    )

    os.close(fd)

    tmp_path = Path(
        tmp_name
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
        },
    )

    try:
        LOGGER.info(
            "Downloading FABDEM: %s",
            url,
        )

        with urllib.request.urlopen(
            request,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:

            with tmp_path.open(
                "wb"
            ) as out:

                while True:
                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    out.write(
                        chunk
                    )

        if not tmp_path.exists():
            raise RuntimeError(
                "FABDEM download produced no file."
            )

        if tmp_path.stat().st_size == 0:
            raise RuntimeError(
                "FABDEM download produced an empty file."
            )

        tmp_path.replace(
            destination
        )

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"FABDEM HTTP error "
            f"{exc.code} for {url}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"FABDEM download failed for {url}: "
            f"{exc.reason}"
        ) from exc

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _get_or_download_tile(
    tile: FabdemTile,
    cache_dir: Path,
) -> tuple[Path, bool]:
    """
    Return a cached FABDEM tile or download it.

    Returns:
        local_path, cache_hit
    """
    group_dir = (
        cache_dir
        / tile.group_name
    )

    local_path = (
        group_dir
        / tile.tile_name
    )

    if local_path.exists():
        if local_path.stat().st_size > 0:
            LOGGER.info(
                "FABDEM cache hit: %s",
                local_path,
            )

            return (
                local_path,
                True,
            )

        # Remove an empty/corrupt cache entry.
        local_path.unlink(
            missing_ok=True
        )

    _download_file(
        tile.url,
        local_path,
    )

    return (
        local_path,
        False,
    )


# ---------------------------------------------------------------------
# FABDEM raster preparation
# ---------------------------------------------------------------------

def _open_and_mosaic(
    paths: list[Path],
) -> tuple[np.ndarray, Affine, CRS]:
    """
    Open local FABDEM tiles and mosaic them.
    """
    if not paths:
        raise ValueError(
            "No FABDEM tiles were supplied for mosaicking."
        )

    datasets = []

    try:
        for path in paths:
            LOGGER.info(
                "Opening FABDEM tile: %s",
                path,
            )

            src = rasterio.open(
                path
            )

            datasets.append(
                src
            )

        mosaic, transform = merge(
            datasets,
            method="first",
            nodata=FABDEM_NODATA,
        )

        if mosaic.ndim != 3:
            raise RuntimeError(
                "Unexpected FABDEM mosaic dimensions."
            )

        array = mosaic[0].astype(
            np.float32,
            copy=False,
        )

        source_crs = datasets[0].crs

        if source_crs is None:
            source_crs = CRS.from_epsg(
                4326
            )

        else:
            source_crs = CRS.from_user_input(
                source_crs
            )

        return (
            array,
            transform,
            source_crs,
        )

    finally:
        for dataset in datasets:
            dataset.close()


def _warp_to_canonical_grid(
    source_array: np.ndarray,
    source_transform: Affine,
    source_crs: CRS,
    width: int,
    height: int,
    transform: Affine,
    crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproject/resample the FABDEM mosaic onto the canonical grid.
    """
    destination = np.full(
        (height, width),
        FABDEM_NODATA,
        dtype=np.float32,
    )

    reproject(
        source=source_array,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=FABDEM_NODATA,
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=FABDEM_NODATA,
        resampling=Resampling.cubic_spline,
        num_threads=2,
    )

    valid = (
        np.isfinite(destination)
        & (
            destination
            != FABDEM_NODATA
        )
    )

    return (
        destination,
        valid,
    )


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

def _write_raw_array(
    path: Path,
    array: np.ndarray,
) -> None:
    """
    Write a raw NumPy-compatible .dat file.
    """
    array.tofile(
        path
    )


def _write_canonical_geotiff(
    path: Path,
    array: np.ndarray,
    transform: Affine,
    crs: CRS,
) -> None:
    """
    Write canonical DTM GeoTIFF.
    """
    height, width = array.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": FABDEM_NODATA,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            array.astype(
                np.float32,
                copy=False,
            ),
            1,
        )

        dst.set_band_description(
            1,
            "FABDEM bare-earth DTM",
        )


def _save_dem_metadata(
    path: Path,
    scene_meta: dict[str, Any],
    tiling_meta: dict[str, Any],
    tiles: list[FabdemTile],
    cache_hits: int,
    downloaded: int,
    width: int,
    height: int,
    transform: Affine,
    crs: CRS,
    valid_fraction: float,
) -> None:
    meta = {
        "provider": "FABDEM",
        "version": "V1-2",

        "source": {
            "base_url": FABDEM_BASE_URL,
            "crs": FABDEM_SOURCE_CRS,
            "nodata": FABDEM_NODATA,
            "tile_size_degrees": TILE_SIZE_DEGREES,
            "group_size_degrees": GROUP_SIZE_DEGREES,
            "vertical_datum": "EGM2008",
        },

        "scene": {
            "classification": scene_meta.get(
                "classification",
                scene_meta.get(
                    "kind"
                ),
            ),
            "crs": scene_meta.get(
                "crs"
            ),
            "gsd_true_m": scene_meta.get(
                "gsd_true_m"
            ),
            "bounds_wgs84": scene_meta.get(
                "bounds_wgs84"
            ),
        },

        "canonical_grid": {
            "width": width,
            "height": height,
            "crs": crs.to_string(),
            "transform": [
                float(transform.a),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e),
                float(transform.f),
            ],
        },

        "tiles": [
            {
                "lat": tile.lat,
                "lon": tile.lon,
                "name": tile.tile_name,
                "group": tile.group_name,
                "url": tile.url,
            }
            for tile in tiles
        ],

        "download": {
            "required_tiles": len(tiles),
            "cache_hits": cache_hits,
            "downloaded": downloaded,
        },

        "valid_fraction": valid_fraction,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            meta,
            f,
            indent=2,
        )


# ---------------------------------------------------------------------
# Main FABDEM function
# ---------------------------------------------------------------------

def get_fabdem(
    scene_meta_path: str | Path,
    tiling_meta_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path = "data/dem_cache/fabdem",
) -> Path:
    """
    Acquire FABDEM tiles for the scene, mosaic them, and align
    the result to the canonical inference grid.

    Parameters
    ----------
    scene_meta_path:
        Path to scene_meta.json.

    tiling_meta_path:
        Path to tiling_meta.json.

    output_dir:
        Directory where DTM outputs are written.

    cache_dir:
        Local FABDEM tile cache.

    Returns
    -------
    Path
        Path to dtm_canonical.tif.
    """
    scene_meta_path = Path(
        scene_meta_path
    ).expanduser().resolve()

    tiling_meta_path = Path(
        tiling_meta_path
    ).expanduser().resolve()

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    cache_dir = Path(
        cache_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load metadata
    # -------------------------------------------------------------

    scene_meta = _load_json(
        scene_meta_path
    )

    tiling_meta = _load_json(
        tiling_meta_path
    )

    # -------------------------------------------------------------
    # Scene bounds
    # -------------------------------------------------------------

    (
        left,
        bottom,
        right,
        top,
    ) = _scene_wgs84_bounds(
        scene_meta
    )

    LOGGER.info(
        "FABDEM scene bounds: "
        "%.6f, %.6f, %.6f, %.6f",
        left,
        bottom,
        right,
        top,
    )

    # -------------------------------------------------------------
    # Canonical grid
    # -------------------------------------------------------------

    (
        width,
        height,
        canonical_transform,
        canonical_crs,
    ) = _extract_canonical_grid(
        tiling_meta
    )

    # Compute approximate GSD from transform.
    canonical_gsd_x = abs(
        canonical_transform.a
    )

    canonical_gsd_y = abs(
        canonical_transform.e
    )

    LOGGER.info(
        "Target grid: %dx%d, "
        "GSD=%.3f m, CRS=%s",
        width,
        height,
        (
            canonical_gsd_x
            + canonical_gsd_y
        ) / 2.0,
        canonical_crs.to_string(),
    )

    # -------------------------------------------------------------
    # Determine FABDEM tiles
    # -------------------------------------------------------------

    tiles = _tile_indices_for_bounds(
        left,
        bottom,
        right,
        top,
    )

    LOGGER.info(
        "FABDEM tiles required: %d",
        len(tiles),
    )

    if not tiles:
        raise RuntimeError(
            "No FABDEM tiles intersect the scene."
        )

    # -------------------------------------------------------------
    # Acquire tiles
    # -------------------------------------------------------------

    local_paths: list[Path] = []
    cache_hits = 0
    downloaded = 0

    for tile in tiles:
        local_path, was_cache_hit = (
            _get_or_download_tile(
                tile=tile,
                cache_dir=cache_dir,
            )
        )

        local_paths.append(
            local_path
        )

        if was_cache_hit:
            cache_hits += 1

        else:
            downloaded += 1

    LOGGER.info(
        "FABDEM acquisition complete: "
        "%d cache hit(s), %d download(s)",
        cache_hits,
        downloaded,
    )

    # -------------------------------------------------------------
    # Mosaic
    # -------------------------------------------------------------

    LOGGER.info(
        "Mosaicking FABDEM tiles..."
    )

    (
        source_array,
        source_transform,
        source_crs,
    ) = _open_and_mosaic(
        local_paths
    )

    LOGGER.info(
        "FABDEM mosaic size: %dx%d",
        source_array.shape[1],
        source_array.shape[0],
    )

    # -------------------------------------------------------------
    # Reproject/resample
    # -------------------------------------------------------------

    LOGGER.info(
        "Aligning FABDEM to canonical grid..."
    )

    (
        dtm,
        valid,
    ) = _warp_to_canonical_grid(
        source_array=source_array,
        source_transform=source_transform,
        source_crs=source_crs,
        width=width,
        height=height,
        transform=canonical_transform,
        crs=canonical_crs,
    )

    valid_fraction = float(
        valid.mean()
    )

    LOGGER.info(
        "Canonical FABDEM valid fraction: %.4f",
        valid_fraction,
    )

    if valid_fraction <= 0.0:
        raise RuntimeError(
            "FABDEM contains no valid pixels "
            "on the canonical scene grid."
        )

    # -------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------

    dtm_dat_path = (
        output_dir
        / "dtm.dat"
    )

    valid_dat_path = (
        output_dir
        / "dtm_valid.dat"
    )

    canonical_tif_path = (
        output_dir
        / "dtm_canonical.tif"
    )

    dem_meta_path = (
        output_dir
        / "dem_meta.json"
    )

    # -------------------------------------------------------------
    # Raw outputs
    # -------------------------------------------------------------

    _write_raw_array(
        dtm_dat_path,
        dtm.astype(
            np.float32,
            copy=False,
        ),
    )

    _write_raw_array(
        valid_dat_path,
        valid.astype(
            np.uint8,
            copy=False,
        ),
    )

    # -------------------------------------------------------------
    # GeoTIFF output
    # -------------------------------------------------------------

    _write_canonical_geotiff(
        path=canonical_tif_path,
        array=dtm,
        transform=canonical_transform,
        crs=canonical_crs,
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    _save_dem_metadata(
        path=dem_meta_path,
        scene_meta=scene_meta,
        tiling_meta=tiling_meta,
        tiles=tiles,
        cache_hits=cache_hits,
        downloaded=downloaded,
        width=width,
        height=height,
        transform=canonical_transform,
        crs=canonical_crs,
        valid_fraction=valid_fraction,
    )

    LOGGER.info(
        "FABDEM DTM complete: %s",
        canonical_tif_path,
    )

    return canonical_tif_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire FABDEM tiles and align them "
            "to the DepthWizard canonical grid."
        )
    )

    parser.add_argument(
        "--scene-meta",
        required=True,
        help="Path to scene_meta.json.",
    )

    parser.add_argument(
        "--tiling-meta",
        required=True,
        help="Path to tiling_meta.json.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for FABDEM outputs.",
    )

    parser.add_argument(
        "--cache-dir",
        default="data/dem_cache/fabdem",
        help="Local FABDEM cache directory.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    get_fabdem(
        scene_meta_path=args.scene_meta,
        tiling_meta_path=args.tiling_meta,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )