# src/dw/core/compose.py

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


LOGGER = logging.getLogger("depthwizard.compose")

NODATA = -9999.0

# Placeholder FABDEM vertical uncertainty used by the prototype.
# Replace this with a per-source uncertainty table in a later release.
DEFAULT_SIGMA_DEM_M = 2.5

# FABDEM-derived terrain is documented here as the vertical-datum source
# carried into the exported metric products.
VERTICAL_DATUM = "EGM2008 (FABDEM source)"
DEFAULT_MAX_AGL_M = 300.0


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object in {path}."
        )

    return data


def _affine_from_value(value: Any) -> Affine:
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
        "Could not parse transform."
    )


def _as_crs(value: Any) -> CRS:
    if isinstance(value, CRS):
        return value

    return CRS.from_user_input(value)


def _read_raw_array(
    path: Path,
    shape: tuple[int, int],
    dtype: np.dtype = np.dtype("float32"),
) -> np.ndarray:
    """
    Read a raw binary array written with ndarray.tofile().
    """
    expected = int(
        shape[0] * shape[1]
    )

    array = np.fromfile(
        path,
        dtype=dtype,
    )

    if array.size != expected:
        raise ValueError(
            f"Unexpected array size for {path}. "
            f"Expected {expected}, got {array.size}."
        )

    return array.reshape(
        shape
    )


def _write_raw_array(
    path: Path,
    array: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.asarray(
        array
    ).tofile(
        path
    )


# ---------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------


def _extract_canonical_grid(
    tiling_meta: dict[str, Any],
) -> tuple[int, int, Affine, CRS]:
    """
    Read canonical grid information from tiling_meta.json.

    Supports the prototype's possible metadata layouts.
    """
    candidates: list[dict[str, Any]] = []

    canonical_grid = tiling_meta.get(
        "canonical_grid"
    )

    if isinstance(canonical_grid, dict):
        candidates.append(
            canonical_grid
        )

    canonical = tiling_meta.get(
        "canonical"
    )

    if isinstance(canonical, dict):
        candidates.append(
            canonical
        )

    grid = tiling_meta.get(
        "grid"
    )

    if isinstance(grid, dict):
        candidates.append(
            grid
        )

    candidates.append(
        tiling_meta
    )

    for grid_data in candidates:
        width = grid_data.get(
            "width"
        )

        height = grid_data.get(
            "height"
        )

        transform_value = grid_data.get(
            "transform"
        )

        crs_value = grid_data.get(
            "crs"
        )

        if (
            width is not None
            and height is not None
            and transform_value is not None
            and crs_value is not None
        ):
            return (
                int(width),
                int(height),
                _affine_from_value(
                    transform_value
                ),
                _as_crs(
                    crs_value
                ),
            )

    # Alternate flat-key layout.
    width = tiling_meta.get(
        "canonical_width"
    )

    height = tiling_meta.get(
        "canonical_height"
    )

    transform_value = tiling_meta.get(
        "canonical_transform"
    )

    crs_value = tiling_meta.get(
        "canonical_crs"
    )

    if (
        width is not None
        and height is not None
        and transform_value is not None
        and crs_value is not None
    ):
        return (
            int(width),
            int(height),
            _affine_from_value(
                transform_value
            ),
            _as_crs(
                crs_value
            ),
        )

    raise KeyError(
        "Canonical width, height, transform, and CRS "
        "could not be found in tiling_meta.json."
    )


def _extract_native_grid(
    scene_meta: dict[str, Any],
) -> tuple[int, int, Affine, CRS] | None:
    """
    Extract native raster grid from scene metadata.
    """
    width = scene_meta.get(
        "width"
    )

    height = scene_meta.get(
        "height"
    )

    transform_value = scene_meta.get(
        "transform"
    )

    crs_value = scene_meta.get(
        "crs"
    )

    if (
        width is None
        or height is None
        or transform_value is None
        or crs_value is None
    ):
        return None

    return (
        int(width),
        int(height),
        _affine_from_value(
            transform_value
        ),
        _as_crs(
            crs_value
        ),
    )


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------


def _load_calibration(
    path: Path,
) -> tuple[float, float, float, float]:
    """
    Load metric calibration parameters.

    Returns:
        scale, offset, scale_sigma, agl_min

    Supported calibration JSON keys:

        scale / a
        offset / b
        scale_sigma / sigma_a
        agl_min
    """

    calibration = _load_json(
        path
    )

    # ---------------------------------------------------------------
    # Scale
    # ---------------------------------------------------------------

    scale = calibration.get(
        "scale"
    )

    if scale is None:
        scale = calibration.get(
            "a"
        )

    if scale is None:
        raise KeyError(
            f"No scale parameter found in {path}."
        )

    scale = float(
        scale
    )

    # ---------------------------------------------------------------
    # Offset
    # ---------------------------------------------------------------

    offset = calibration.get(
        "offset"
    )

    if offset is None:
        offset = calibration.get(
            "b",
            0.0,
        )

    offset = float(
        offset
    )

    # ---------------------------------------------------------------
    # Scale uncertainty
    # ---------------------------------------------------------------

    scale_sigma = calibration.get(
        "scale_sigma"
    )

    if scale_sigma is None:
        scale_sigma = calibration.get(
            "sigma_a",
            0.0,
        )

    scale_sigma = float(
        scale_sigma
    )

    # ---------------------------------------------------------------
    # Physical AGL minimum
    # ---------------------------------------------------------------

    agl_min = float(
        calibration.get(
            "agl_min",
            0.0,
        )
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            f"Invalid calibration scale: {scale}"
        )

    if not np.isfinite(offset):
        raise ValueError(
            f"Invalid calibration offset: {offset}"
        )

    if (
        not np.isfinite(scale_sigma)
        or scale_sigma < 0.0
    ):
        raise ValueError(
            f"Invalid calibration scale sigma: {scale_sigma}"
        )

    if not np.isfinite(agl_min):
        raise ValueError(
            f"Invalid AGL minimum: {agl_min}"
        )

    return (
        scale,
        offset,
        scale_sigma,
        agl_min,
    )


# ---------------------------------------------------------------------
# Array processing
# ---------------------------------------------------------------------


def _make_agl(
    relative_height: np.ndarray,
    scale: float,
    offset: float,
    agl_min: float,
    max_agl_m: float,
) -> np.ndarray:
    """
    Convert relative model output into metric AGL.

    Current prototype equation:

        AGL_raw = scale * R + offset

        AGL = max(agl_min, AGL_raw)

    Then apply the safety ceiling:

        AGL = min(AGL, max_agl_m)

    For the current Potsdam calibration:

        scale    = 2.94690071
        offset   = -1.63295141
        agl_min  = 0

    therefore:

        AGL = max(0, 2.94690071 * R - 1.63295141)
    """

    r = np.asarray(
        relative_height,
        dtype=np.float32,
    )

    finite = np.isfinite(
        r
    )

    agl = np.full_like(
        r,
        NODATA,
        dtype=np.float32,
    )

    if not np.any(finite):
        return agl

    # ---------------------------------------------------------------
    # Complete linear metric conversion first.
    # ---------------------------------------------------------------

    raw_agl = (
        float(scale)
        * r[finite]
        + float(offset)
    )

    # ---------------------------------------------------------------
    # Physical lower bound.
    #
    # This is where the calibration's negative intercept is handled.
    # ---------------------------------------------------------------

    raw_agl = np.maximum(
        raw_agl,
        float(agl_min),
    )

    # ---------------------------------------------------------------
    # Safety ceiling.
    # ---------------------------------------------------------------

    raw_agl = np.minimum(
        raw_agl,
        float(max_agl_m),
    )

    agl[finite] = raw_agl.astype(
        np.float32
    )

    return agl


def _make_sigma(
    relative_height: np.ndarray,
    scale_sigma: float,
    sigma_tta: np.ndarray | None = None,
    scale: float = 1.0,
    sigma_dem_m: float = DEFAULT_SIGMA_DEM_M,
) -> np.ndarray:
    """
    Combine model/TTA, scale-prior, and DEM uncertainties.

    TTA uncertainty from infer.py is produced in relative-model units, so
    it is first converted to metres using the fitted/prior scale. The
    combined metric uncertainty is:

        sigma = sqrt(
            (scale * sigma_tta)^2
            + (scale_sigma * R)^2
            + sigma_dem^2
        )

    The DEM term is currently a placeholder for FABDEM (~2.5 m).
    """
    r = np.asarray(
        relative_height,
        dtype=np.float32,
    )

    valid = np.isfinite(r)

    if sigma_tta is None:
        tta_m = np.zeros_like(r, dtype=np.float32)
    else:
        tta = np.asarray(
            sigma_tta,
            dtype=np.float32,
        )
        if tta.shape != r.shape:
            raise ValueError(
                "sigma_tta shape does not match the relative-height grid."
            )
        tta = np.where(
            np.isfinite(tta),
            tta,
            0.0,
        )
        tta_m = np.abs(
            float(scale)
        ) * np.abs(tta)

    sigma = np.sqrt(
        np.square(tta_m)
        + np.square(
            float(scale_sigma) * np.abs(r)
        )
        + np.square(
            float(sigma_dem_m)
        )
    ).astype(
        np.float32,
        copy=False,
    )

    sigma[~valid] = NODATA

    return sigma


def _compose_dsm_array(
    dtm: np.ndarray,
    agl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compose:

        DSM = DTM + AGL
    """
    valid = (
        np.isfinite(dtm)
        & np.isfinite(agl)
        & (dtm != NODATA)
        & (agl != NODATA)
    )

    dsm = np.full_like(
        dtm,
        NODATA,
        dtype=np.float32,
    )

    dsm[valid] = (
        dtm[valid]
        + agl[valid]
    )

    return (
        dsm,
        valid,
    )


def _clean_array_for_tif(
    array: np.ndarray,
) -> np.ndarray:
    """
    Ensure non-finite output values become GeoTIFF nodata.
    """
    result = np.asarray(
        array,
        dtype=np.float32,
    ).copy()

    invalid = ~np.isfinite(
        result
    )

    result[invalid] = NODATA

    return result


# ---------------------------------------------------------------------
# GeoTIFF helpers
# ---------------------------------------------------------------------


def _write_float_tif(
    path: Path,
    array: np.ndarray,
    transform: Affine,
    crs: CRS,
    product: str,
    tags: dict[str, Any] | None = None,
    description: str | None = None,
) -> None:
    """
    Write a single-band Float32 GeoTIFF.

    PRODUCT is written exactly once.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned = _clean_array_for_tif(
        array
    )

    height, width = cleaned.shape

    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
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
            cleaned,
            1,
        )

        if description:
            dst.set_band_description(
                1,
                description,
            )

        safe_tags = dict(
            tags or {}
        )

        safe_tags.pop(
            "PRODUCT",
            None,
        )

        dst.update_tags(
            PRODUCT=product,
            **safe_tags,
        )


def _write_mask_tif(
    path: Path,
    mask: np.ndarray,
    transform: Affine,
    crs: CRS,
    product: str,
) -> None:
    """
    Write a valid-mask GeoTIFF.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    array = np.asarray(
        mask,
        dtype=np.uint8,
    )

    height, width = array.shape

    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            array,
            1,
        )

        dst.update_tags(
            PRODUCT=product
        )


# ---------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------


def _warp_array(
    source: np.ndarray,
    source_transform: Affine,
    source_crs: CRS,
    width: int,
    height: int,
    destination_transform: Affine,
    destination_crs: CRS,
    resampling: Resampling,
) -> np.ndarray:
    """
    Warp a Float32 array between raster grids.
    """
    destination = np.full(
        (height, width),
        NODATA,
        dtype=np.float32,
    )

    reproject(
        source=source,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=NODATA,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        dst_nodata=NODATA,
        resampling=resampling,
        num_threads=2,
    )

    return destination


def _warp_mask(
    source: np.ndarray,
    source_transform: Affine,
    source_crs: CRS,
    width: int,
    height: int,
    destination_transform: Affine,
    destination_crs: CRS,
) -> np.ndarray:
    """
    Warp a binary validity mask using nearest-neighbour.
    """
    source_uint8 = np.asarray(
        source,
        dtype=np.uint8,
    )

    destination = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    reproject(
        source=source_uint8,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=0,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
        num_threads=2,
    )

    return (
        destination > 0
    ).astype(
        np.uint8
    )


# ---------------------------------------------------------------------
# Metadata writing
# ---------------------------------------------------------------------


def _save_compose_metadata(
    path: Path,
    scene_meta: dict[str, Any],
    tiling_meta: dict[str, Any],
    scale: float,
    offset: float,
    scale_sigma: float,
    agl_min: float,
    max_agl_m: float,
    canonical_width: int,
    canonical_height: int,
    canonical_transform: Affine,
    canonical_crs: CRS,
    native_grid_used: bool,
    native_export_requested: bool,
) -> None:

    metadata = {
        "pipeline": "DepthWizard prototype",
        "stage": "DSM composition",

        "formula": {
            "agl_raw": "scale * R + offset",
            "agl": "max(agl_min, scale * R + offset)",
            "agl_ceiling": "min(AGL, max_agl_m)",
            "dsm": "DTM + AGL",
        },

        "calibration": {
            "scale": scale,
            "offset": offset,
            "scale_sigma": scale_sigma,
            "agl_min": agl_min,
        },

        "limits": {
            "max_agl_m": max_agl_m,
        },

        "canonical_grid": {
            "width": canonical_width,
            "height": canonical_height,
            "crs": canonical_crs.to_string(),
            "transform": [
                float(canonical_transform.a),
                float(canonical_transform.b),
                float(canonical_transform.c),
                float(canonical_transform.d),
                float(canonical_transform.e),
                float(canonical_transform.f),
            ],
        },

        "native_export": {
            "requested": bool(native_export_requested),
            "written": native_grid_used,
            "crs": scene_meta.get(
                "crs"
            ),
            "width": scene_meta.get(
                "width"
            ),
            "height": scene_meta.get(
                "height"
            ),
        },

        "scene": {
            "path": scene_meta.get(
                "path"
            ),
            "classification": scene_meta.get(
                "classification",
                scene_meta.get(
                    "kind"
                ),
            ),
            "gsd_true_m": scene_meta.get(
                "gsd_true_m"
            ),
        },

        "outputs": [
            "agl.dat",
            "ndsm.dat",
            "dsm.dat",
            "dtm.dat",
            "sigma.dat",
            "valid_mask.dat",
            "agl_canonical.tif",
            "ndsm_canonical.tif",
            "dsm_canonical.tif",
            "dtm_canonical.tif",
            "sigma_canonical.tif",
            "valid_mask_canonical.tif",
            "agl.tif",
            "ndsm.tif",
            "dsm.tif",
            "dtm.tif",
            "sigma.tif",
            "valid_mask.tif",
            "agl_native.tif",
            "ndsm_native.tif",
            "dsm_native.tif",
            "dtm_native.tif",
            "sigma_native.tif",
            "valid_mask_native.tif",
        ],
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


# ---------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------


def compose_dsm(
    ground_dir: str | Path,
    dem_dir: str | Path,
    calibration_path: str | Path,
    tiling_meta_path: str | Path,
    scene_meta_path: str | Path,
    output_dir: str | Path,
    max_agl_m: float = DEFAULT_MAX_AGL_M,
    inference_dir: str | Path | None = None,
    write_native_products: bool = False,
) -> Path:
    """
    Compose a metric DSM from:

        R     = relative prediction
        DTM   = aligned FABDEM terrain
        scale = metric calibration
        offset = metric calibration intercept

    Prototype equation:

        AGL_raw = scale * R + offset
        AGL     = max(agl_min, AGL_raw)
        AGL     = min(AGL, max_agl_m)

        DSM = DTM + AGL

    Outputs canonical-grid products as the primary/final products.
    Native-grid products are optional because native export can multiply
    storage substantially for high-resolution inputs.
    """

    ground_dir = (
        Path(ground_dir)
        .expanduser()
        .resolve()
    )

    dem_dir = (
        Path(dem_dir)
        .expanduser()
        .resolve()
    )

    calibration_path = (
        Path(calibration_path)
        .expanduser()
        .resolve()
    )

    tiling_meta_path = (
        Path(tiling_meta_path)
        .expanduser()
        .resolve()
    )

    scene_meta_path = (
        Path(scene_meta_path)
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )

    if inference_dir is not None:
        inference_dir = (
            Path(inference_dir)
            .expanduser()
            .resolve()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load metadata
    # -------------------------------------------------------------

    tiling_meta = _load_json(
        tiling_meta_path
    )

    scene_meta = _load_json(
        scene_meta_path
    )

    (
        canonical_width,
        canonical_height,
        canonical_transform,
        canonical_crs,
    ) = _extract_canonical_grid(
        tiling_meta
    )

    LOGGER.info(
        "Canonical grid: %dx%d, CRS=%s",
        canonical_width,
        canonical_height,
        canonical_crs.to_string(),
    )

    # -------------------------------------------------------------
    # Load calibration
    # -------------------------------------------------------------

    (
        scale,
        offset,
        scale_sigma,
        agl_min,
    ) = _load_calibration(
        calibration_path
    )

    LOGGER.info(
        "Metric conversion: "
        "AGL = max(%.6f, %.6f * R + %.6f)",
        agl_min,
        scale,
        offset,
    )

    LOGGER.info(
        "Maximum AGL ceiling: %.3f m",
        max_agl_m,
    )

    # -------------------------------------------------------------
    # Load R and DTM
    # -------------------------------------------------------------

    r_path = (
        ground_dir
        / "R.dat"
    )

    dtm_path = (
        dem_dir
        / "dtm.dat"
    )

    if not r_path.exists():
        raise FileNotFoundError(
            f"Relative AGL array not found: {r_path}"
        )

    if not dtm_path.exists():
        raise FileNotFoundError(
            f"DTM array not found: {dtm_path}"
        )

    relative_height = _read_raw_array(
        r_path,
        (
            canonical_height,
            canonical_width,
        ),
    )

    dtm = _read_raw_array(
        dtm_path,
        (
            canonical_height,
            canonical_width,
        ),
    )

    # -------------------------------------------------------------
    # Load optional DTM validity mask
    # -------------------------------------------------------------

    dtm_valid_path = (
        dem_dir
        / "dtm_valid.dat"
    )

    if dtm_valid_path.exists():

        dtm_valid = np.fromfile(
            dtm_valid_path,
            dtype=np.uint8,
        )

        expected = (
            canonical_width
            * canonical_height
        )

        if dtm_valid.size == expected:

            dtm_valid = dtm_valid.reshape(
                (
                    canonical_height,
                    canonical_width,
                )
            )

            dtm = np.where(
                dtm_valid > 0,
                dtm,
                NODATA,
            ).astype(
                np.float32
            )

    # -------------------------------------------------------------
    # Load TTA uncertainty from inference
    # -------------------------------------------------------------

    sigma_tta = None

    if inference_dir is not None:
        sigma_tta_path = (
            inference_dir
            / "sigma_tta.dat"
        )

        if sigma_tta_path.exists():
            sigma_tta = _read_raw_array(
                sigma_tta_path,
                (
                    canonical_height,
                    canonical_width,
                ),
            )
        else:
            LOGGER.warning(
                "TTA uncertainty raster not found: %s. "
                "Using zero TTA contribution.",
                sigma_tta_path,
            )
    else:
        LOGGER.warning(
            "inference_dir was not supplied to compose_dsm; "
            "using zero TTA contribution."
        )

    # -------------------------------------------------------------
    # Metric conversion
    # -------------------------------------------------------------

    agl = _make_agl(
        relative_height=relative_height,
        scale=scale,
        offset=offset,
        agl_min=agl_min,
        max_agl_m=max_agl_m,
    )

    # -------------------------------------------------------------
    # DSM
    # -------------------------------------------------------------

    dsm, valid_mask = _compose_dsm_array(
        dtm=dtm,
        agl=agl,
    )

    # NDSM is the metric AGL.
    ndsm = agl.copy()

    # -------------------------------------------------------------
    # Ensure invalid cells are nodata.
    # -------------------------------------------------------------

    ndsm[~valid_mask] = NODATA
    agl[~valid_mask] = NODATA
    dtm[~valid_mask] = NODATA

    # -------------------------------------------------------------
    # Uncertainty
    # -------------------------------------------------------------

    sigma = _make_sigma(
        relative_height=relative_height,
        scale_sigma=scale_sigma,
        sigma_tta=sigma_tta,
        scale=scale,
        sigma_dem_m=DEFAULT_SIGMA_DEM_M,
    )

    sigma[~valid_mask] = NODATA

    # -------------------------------------------------------------
    # Raw canonical arrays
    # -------------------------------------------------------------

    _write_raw_array(
        output_dir / "agl.dat",
        agl,
    )

    _write_raw_array(
        output_dir / "ndsm.dat",
        ndsm,
    )

    _write_raw_array(
        output_dir / "dsm.dat",
        dsm,
    )

    _write_raw_array(
        output_dir / "dtm.dat",
        dtm,
    )

    _write_raw_array(
        output_dir / "sigma.dat",
        sigma,
    )

    _write_raw_array(
        output_dir / "valid_mask.dat",
        valid_mask.astype(
            np.uint8
        ),
    )

    # -------------------------------------------------------------
    # Canonical GeoTIFFs
    # -------------------------------------------------------------

    common_tags = {
        "CRS": canonical_crs.to_string(),
        "SOURCE": "DepthWizard prototype",
        "GRID": "canonical",
        "VERTICAL_DATUM": VERTICAL_DATUM,
        "VERTICAL_DATUM_SOURCE": "FABDEM",
    }

    _write_float_tif(
        path=output_dir / "agl_canonical.tif",
        array=agl,
        transform=canonical_transform,
        crs=canonical_crs,
        product="AGL",
        tags=common_tags,
        description="Metric above-ground height",
    )

    _write_float_tif(
        path=output_dir / "ndsm_canonical.tif",
        array=ndsm,
        transform=canonical_transform,
        crs=canonical_crs,
        product="NDSM",
        tags=common_tags,
        description="Metric normalized DSM / above-ground height",
    )

    _write_float_tif(
        path=output_dir / "dsm_canonical.tif",
        array=dsm,
        transform=canonical_transform,
        crs=canonical_crs,
        product="DSM",
        tags=common_tags,
        description="Metric digital surface model",
    )

    _write_float_tif(
        path=output_dir / "dtm_canonical.tif",
        array=dtm,
        transform=canonical_transform,
        crs=canonical_crs,
        product="DTM",
        tags=common_tags,
        description="FABDEM bare-earth digital terrain model",
    )

    _write_float_tif(
        path=output_dir / "sigma_canonical.tif",
        array=sigma,
        transform=canonical_transform,
        crs=canonical_crs,
        product="SIGMA",
        tags=common_tags,
        description="Prototype DSM uncertainty estimate",
    )

    _write_mask_tif(
        path=output_dir / "valid_mask_canonical.tif",
        mask=valid_mask,
        transform=canonical_transform,
        crs=canonical_crs,
        product="VALID_MASK",
    )

    # -------------------------------------------------------------
    # Primary final outputs: canonical 1 m grid
    # -------------------------------------------------------------

    LOGGER.info(
        "Writing primary final products on canonical grid: %dx%d.",
        canonical_width,
        canonical_height,
    )

    _write_float_tif(
        path=output_dir / "agl.tif",
        array=agl,
        transform=canonical_transform,
        crs=canonical_crs,
        product="AGL",
        tags=common_tags,
        description="Metric above-ground height",
    )

    _write_float_tif(
        path=output_dir / "ndsm.tif",
        array=ndsm,
        transform=canonical_transform,
        crs=canonical_crs,
        product="NDSM",
        tags=common_tags,
        description="Metric normalized DSM / above-ground height",
    )

    _write_float_tif(
        path=output_dir / "dsm.tif",
        array=dsm,
        transform=canonical_transform,
        crs=canonical_crs,
        product="DSM",
        tags=common_tags,
        description="Metric digital surface model",
    )

    _write_float_tif(
        path=output_dir / "dtm.tif",
        array=dtm,
        transform=canonical_transform,
        crs=canonical_crs,
        product="DTM",
        tags=common_tags,
        description="FABDEM bare-earth digital terrain model",
    )

    _write_float_tif(
        path=output_dir / "sigma.tif",
        array=sigma,
        transform=canonical_transform,
        crs=canonical_crs,
        product="SIGMA",
        tags=common_tags,
        description="Prototype DSM uncertainty estimate",
    )

    _write_mask_tif(
        path=output_dir / "valid_mask.tif",
        mask=valid_mask,
        transform=canonical_transform,
        crs=canonical_crs,
        product="VALID_MASK",
    )

    # -------------------------------------------------------------
    # Optional native scene export
    # -------------------------------------------------------------

    native_grid = _extract_native_grid(
        scene_meta
    )

    native_grid_used = False

    if native_grid is not None and write_native_products:

        (
            native_width,
            native_height,
            native_transform,
            native_crs,
        ) = native_grid

        LOGGER.info(
            "Writing optional native final products: "
            "%dx%d, CRS=%s",
            native_width,
            native_height,
            native_crs.to_string(),
        )

        agl_native = _warp_array(
            source=agl,
            source_transform=canonical_transform,
            source_crs=canonical_crs,
            width=native_width,
            height=native_height,
            destination_transform=native_transform,
            destination_crs=native_crs,
            resampling=Resampling.cubic_spline,
        )

        ndsm_native = _warp_array(
            source=ndsm,
            source_transform=canonical_transform,
            source_crs=canonical_crs,
            width=native_width,
            height=native_height,
            destination_transform=native_transform,
            destination_crs=native_crs,
            resampling=Resampling.cubic_spline,
        )

        dsm_native = _warp_array(
            source=dsm,
            source_transform=canonical_transform,
            source_crs=canonical_crs,
            width=native_width,
            height=native_height,
            destination_transform=native_transform,
            destination_crs=native_crs,
            resampling=Resampling.cubic_spline,
        )

        dtm_native = _warp_array(
            source=dtm,
            source_transform=canonical_transform,
            source_crs=canonical_crs,
            width=native_width,
            height=native_height,
            destination_transform=native_transform,
            destination_crs=native_crs,
            resampling=Resampling.cubic_spline,
        )

        sigma_native = _warp_array(
            source=sigma,
            source_transform=canonical_transform,
            source_crs=canonical_crs,
            width=native_width,
            height=native_height,
            destination_transform=native_transform,
            destination_crs=native_crs,
            resampling=Resampling.cubic_spline,
        )

        valid_native = _warp_mask(
            source=valid_mask,
            source_transform=canonical_transform,
            source_crs=canonical_crs,
            width=native_width,
            height=native_height,
            destination_transform=native_transform,
            destination_crs=native_crs,
        )

        native_valid = valid_native > 0

        for array in (
            agl_native,
            ndsm_native,
            dsm_native,
            dtm_native,
            sigma_native,
        ):
            array[~native_valid] = NODATA

        native_tags = {
            "CRS": native_crs.to_string(),
            "SOURCE": "DepthWizard prototype",
            "GRID": "native",
            "VERTICAL_DATUM": VERTICAL_DATUM,
            "VERTICAL_DATUM_SOURCE": "FABDEM",
        }

        _write_float_tif(
            path=output_dir / "agl_native.tif",
            array=agl_native,
            transform=native_transform,
            crs=native_crs,
            product="AGL",
            tags=native_tags,
            description="Metric above-ground height",
        )

        _write_float_tif(
            path=output_dir / "ndsm_native.tif",
            array=ndsm_native,
            transform=native_transform,
            crs=native_crs,
            product="NDSM",
            tags=native_tags,
            description="Metric normalized DSM / above-ground height",
        )

        _write_float_tif(
            path=output_dir / "dsm_native.tif",
            array=dsm_native,
            transform=native_transform,
            crs=native_crs,
            product="DSM",
            tags=native_tags,
            description="Metric digital surface model",
        )

        _write_float_tif(
            path=output_dir / "dtm_native.tif",
            array=dtm_native,
            transform=native_transform,
            crs=native_crs,
            product="DTM",
            tags=native_tags,
            description="FABDEM bare-earth digital terrain model",
        )

        _write_float_tif(
            path=output_dir / "sigma_native.tif",
            array=sigma_native,
            transform=native_transform,
            crs=native_crs,
            product="SIGMA",
            tags=native_tags,
            description="Prototype DSM uncertainty estimate",
        )

        _write_mask_tif(
            path=output_dir / "valid_mask_native.tif",
            mask=valid_native.astype(np.uint8),
            transform=native_transform,
            crs=native_crs,
            product="VALID_MASK",
        )

        native_grid_used = True

    elif native_grid is not None and not write_native_products:

        LOGGER.info(
            "Native final export disabled; canonical 1 m products "
            "are the primary outputs. Use --write-native-compose "
            "only when native-resolution final rasters are required."
        )

    else:

        LOGGER.info(
            "Native georeferenced grid is not available; "
            "canonical products remain the final outputs."
        )

    # -------------------------------------------------------------
    # Compose metadata
    # -------------------------------------------------------------

    compose_meta_path = (
        output_dir
        / "compose_meta.json"
    )

    _save_compose_metadata(
        path=compose_meta_path,
        scene_meta=scene_meta,
        tiling_meta=tiling_meta,
        scale=scale,
        offset=offset,
        scale_sigma=scale_sigma,
        agl_min=agl_min,
        max_agl_m=max_agl_m,
        canonical_width=canonical_width,
        canonical_height=canonical_height,
        canonical_transform=canonical_transform,
        canonical_crs=canonical_crs,
        native_grid_used=native_grid_used,
        native_export_requested=bool(write_native_products),
    )

    # -------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------

    valid_count = int(
        valid_mask.sum()
    )

    total_count = int(
        valid_mask.size
    )

    valid_fraction = (
        valid_count / total_count
        if total_count > 0
        else 0.0
    )

    LOGGER.info(
        "Valid DSM fraction: %.4f",
        valid_fraction,
    )

    if valid_count > 0:

        valid_agl = agl[
            valid_mask
        ]

        valid_dtm = dtm[
            valid_mask
        ]

        valid_dsm = dsm[
            valid_mask
        ]

        LOGGER.info(
            "AGL range: %.3f to %.3f m",
            float(
                np.min(valid_agl)
            ),
            float(
                np.max(valid_agl)
            ),
        )

        LOGGER.info(
            "DTM range: %.3f to %.3f m",
            float(
                np.min(valid_dtm)
            ),
            float(
                np.max(valid_dtm)
            ),
        )

        LOGGER.info(
            "DSM range: %.3f to %.3f m",
            float(
                np.min(valid_dsm)
            ),
            float(
                np.max(valid_dsm)
            ),
        )

    LOGGER.info(
        "DSM composition complete."
    )

    LOGGER.info(
        "DSM: %s",
        output_dir / "dsm.tif",
    )

    return output_dir / "dsm.tif"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Compose metric AGL/DSM products from "
            "DepthWizard prototype intermediate outputs."
        )
    )

    parser.add_argument(
        "--ground-dir",
        required=True,
        help="Directory containing R.dat.",
    )

    parser.add_argument(
        "--dem-dir",
        required=True,
        help="Directory containing dtm.dat.",
    )

    parser.add_argument(
        "--calibration",
        required=True,
        help="Path to calibration.json.",
    )

    parser.add_argument(
        "--tiling-meta",
        required=True,
        help="Path to tiling_meta.json.",
    )

    parser.add_argument(
        "--scene-meta",
        required=True,
        help="Path to scene_meta.json.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for DSM outputs.",
    )

    parser.add_argument(
        "--inference-dir",
        default=None,
        help=(
            "Inference directory containing sigma_tta.dat. "
            "When omitted, TTA uncertainty contributes zero."
        ),
    )

    parser.add_argument(
        "--max-agl-m",
        type=float,
        default=DEFAULT_MAX_AGL_M,
        help="Maximum predicted AGL in metres.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.max_agl_m <= 0.0:
        raise ValueError(
            "--max-agl-m must be > 0."
        )

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

    compose_dsm(
        ground_dir=args.ground_dir,
        dem_dir=args.dem_dir,
        calibration_path=args.calibration,
        tiling_meta_path=args.tiling_meta,
        scene_meta_path=args.scene_meta,
        output_dir=args.output_dir,
        max_agl_m=args.max_agl_m,
        inference_dir=args.inference_dir,
        write_native_products=args.write_native_compose,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )