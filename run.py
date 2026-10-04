# run.py

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import rasterio
import torch
from pyproj import CRS, Transformer
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds

from src.dw.core.meta import resolve_scene, save_scene_meta
from src.dw.core.infer import run_inference
from src.dw.core.dem import get_fabdem
from src.dw.core.ground import (
    DEFAULT_GROUND_WINDOW_M,
    DEFAULT_SMOOTH_SIGMA_M,
    estimate_ground,
)
from src.dw.core.calibrate import (
    DEFAULT_GCP_LOO_ACTION,
    DEFAULT_GCP_LOO_RATIO_LIMIT,
    DEFAULT_GCP_MIN_R_SPREAD,
    DEFAULT_GCP_TAU_M,
    DEFAULT_GCP_Z_REFERENCE,
    DEFAULT_GCP_GEOID_GRID,
    DEFAULT_OFFSET_PRIOR,
    DEFAULT_OFFSET_SIGMA,
    DEFAULT_SCALE_PRIOR,
    DEFAULT_SCALE_SIGMA,
    calibrate_relative_height,
    calibrate_with_gcps,
)
from src.dw.core.compose import (
    DEFAULT_MAX_AGL_M,
    compose_dsm,
)
from src.dw.core.tiling import (
    DEFAULT_OVERLAP,
    DEFAULT_TILE_SIZE,
    DEFAULT_TRAIN_GSD_M,
    prepare_tiles,
)
from src.dw.core.gcp import (
    build_gcp_context,
    load_gcps,
    save_gcp_context,
)


LOGGER = logging.getLogger("depthwizard")


# ---------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------

RELATIVE_ONLY = "RELATIVE_ONLY"
METRIC_PRIOR = "METRIC_PRIOR"
METRIC_GCP = "METRIC_GCP"

SUPPORTED_GEO_KINDS = {
    "GEO_PROJECTED",
    "GEO_GEOGRAPHIC",
    "GEO_WEBMERC",
}


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------


def meta_get(
    meta: Any,
    key: str,
    default: Any = None,
) -> Any:
    if isinstance(meta, dict):
        return meta.get(key, default)

    return getattr(meta, key, default)


def meta_to_dict(
    meta: Any,
) -> dict[str, Any]:
    if isinstance(meta, dict):
        return dict(meta)

    if is_dataclass(meta):
        return asdict(meta)

    if hasattr(meta, "__dict__"):
        return dict(vars(meta))

    return {}


# ---------------------------------------------------------------------
# Phase 2 georeferencing recovery
# ---------------------------------------------------------------------


def _is_identity_transform(
    transform: Affine,
) -> bool:
    return (
        abs(transform.a - 1.0) < 1e-12
        and abs(transform.b) < 1e-12
        and abs(transform.c) < 1e-12
        and abs(transform.d) < 1e-12
        and abs(transform.e - 1.0) < 1e-12
        and abs(transform.f) < 1e-12
    )


def _is_valid_transform(
    transform: Affine,
) -> bool:
    values = (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )

    if not all(
        math.isfinite(float(value))
        for value in values
    ):
        return False

    if _is_identity_transform(transform):
        return False

    determinant = (
        transform.a * transform.e
        - transform.b * transform.d
    )

    if abs(determinant) < 1e-15:
        return False

    return True


def _utm_crs_from_lon_lat(
    lon: float,
    lat: float,
) -> CRS:
    zone = int(
        math.floor(
            (lon + 180.0) / 6.0
        ) + 1
    )

    zone = max(
        1,
        min(
            60,
            zone,
        ),
    )

    if lat >= 0.0:
        return CRS.from_epsg(
            32600 + zone
        )

    return CRS.from_epsg(
        32700 + zone
    )


def _compute_phase2_georef(
    input_path: Path,
) -> dict[str, Any] | None:
    """
    Recover trustworthy georeferencing directly from the raster.

    This exists because Phase 1 metadata intentionally treats some
    unusual/unsupported GSDs conservatively. For Phase 2, a raster
    with a valid CRS and valid non-identity transform is still a
    georeferenced raster.

    This helper does NOT make a non-georeferenced raster metric.
    """

    with rasterio.open(
        input_path
    ) as src:

        src_crs = src.crs
        transform = src.transform

        if src_crs is None:
            return None

        if not _is_valid_transform(
            transform
        ):
            return None

        crs = CRS.from_user_input(
            src_crs
        )

        bounds_native = array_bounds(
            src.height,
            src.width,
            transform,
        )

        left, bottom, right, top = (
            float(bounds_native[0]),
            float(bounds_native[1]),
            float(bounds_native[2]),
            float(bounds_native[3]),
        )

        bounds_wgs84 = transform_bounds(
            crs,
            "EPSG:4326",
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )

        west, south, east, north = (
            float(bounds_wgs84[0]),
            float(bounds_wgs84[1]),
            float(bounds_wgs84[2]),
            float(bounds_wgs84[3]),
        )

        centre_lon = (
            west + east
        ) / 2.0

        centre_lat = (
            south + north
        ) / 2.0

        utm_crs = _utm_crs_from_lon_lat(
            centre_lon,
            centre_lat,
        )

        # ---------------------------------------------------------
        # Local GSD in metres
        # ---------------------------------------------------------

        if crs.is_projected:

            px_size = math.hypot(
                transform.a,
                transform.d,
            )

            py_size = math.hypot(
                transform.b,
                transform.e,
            )

            gsd_true_m = (
                px_size + py_size
            ) / 2.0

        else:

            centre_x = (
                transform.c
                + transform.a
                * (src.width / 2.0)
                + transform.b
                * (src.height / 2.0)
            )

            centre_y = (
                transform.f
                + transform.d
                * (src.width / 2.0)
                + transform.e
                * (src.height / 2.0)
            )

            x1 = centre_x + transform.a
            y1 = centre_y + transform.d

            x2 = centre_x + transform.b
            y2 = centre_y + transform.e

            transformer = Transformer.from_crs(
                crs,
                utm_crs,
                always_xy=True,
            )

            cx, cy = transformer.transform(
                centre_x,
                centre_y,
            )

            px, py = transformer.transform(
                x1,
                y1,
            )

            qx, qy = transformer.transform(
                x2,
                y2,
            )

            x_gsd = math.hypot(
                px - cx,
                py - cy,
            )

            y_gsd = math.hypot(
                qx - cx,
                qy - cy,
            )

            gsd_true_m = (
                x_gsd + y_gsd
            ) / 2.0

        epsg = crs.to_epsg()

        if epsg == 3857:
            kind = "GEO_WEBMERC"

        elif crs.is_projected:
            kind = "GEO_PROJECTED"

        elif crs.is_geographic:
            kind = "GEO_GEOGRAPHIC"

        else:
            return None

        return {
            "kind": kind,
            "crs": crs.to_string(),
            "utm_crs": utm_crs.to_string(),
            "transform": [
                float(transform.a),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e),
                float(transform.f),
            ],
            "bounds_native": {
                "left": left,
                "bottom": bottom,
                "right": right,
                "top": top,
            },
            "bounds_wgs84": {
                "left": west,
                "bottom": south,
                "right": east,
                "top": north,
            },
            "centre": {
                "lon": float(centre_lon),
                "lat": float(centre_lat),
            },
            "gsd_true_m": float(
                gsd_true_m
            ),
        }


def repair_phase2_scene_metadata(
    input_path: Path,
    scene_meta: Any,
) -> Any:
    """
    Restore valid raster georeferencing for Phase 2 without changing
    Phase 1's conservative meta.py behavior.

    Only CRS + valid non-identity transform are accepted.

    A true NONE/unreferenced raster remains NONE.
    """

    recovered = _compute_phase2_georef(
        input_path
    )

    if recovered is None:
        return scene_meta

    current_kind = str(
        meta_get(
            scene_meta,
            "kind",
            "",
        )
    ).upper()

    current_crs = meta_get(
        scene_meta,
        "crs",
    )

    current_gsd = meta_get(
        scene_meta,
        "gsd_true_m",
    )

    # -------------------------------------------------------------
    # We only need recovery when metadata is incomplete/inconsistent.
    # -------------------------------------------------------------

    needs_repair = (
        current_kind not in SUPPORTED_GEO_KINDS
        or current_crs is None
        or current_gsd is None
    )

    if not needs_repair:
        return scene_meta

    LOGGER.warning(
        "Phase 2 georeferencing recovery: raster contains valid "
        "CRS and transform but Phase 1 metadata classified it "
        "conservatively. Restoring raster georeferencing for "
        "metric routing."
    )

    if isinstance(
        scene_meta,
        dict,
    ):
        scene_meta.update(
            recovered
        )

        scene_meta["use_dem"] = True
        scene_meta["relative_only"] = False

        warnings = list(
            scene_meta.get(
                "warnings",
                [],
            )
        )

        warnings.append(
            "Phase 2 retained metric routing from valid raster "
            "CRS/transform despite conservative Phase 1 metadata."
        )

        scene_meta["warnings"] = warnings

        scene_meta["classification"] = (
            recovered["kind"]
        )

        return scene_meta

    # SceneMeta is mutable, so update fields directly.
    for key, value in recovered.items():
        setattr(
            scene_meta,
            key,
            value,
        )

    scene_meta.use_dem = True
    scene_meta.relative_only = False

    if hasattr(
        scene_meta,
        "warnings",
    ):
        scene_meta.warnings.append(
            "Phase 2 retained metric routing from valid raster "
            "CRS/transform despite conservative Phase 1 metadata."
        )

    return scene_meta


def scene_is_reliably_georeferenced(
    scene_meta: Any,
) -> bool:
    """
    Determine whether the raster has usable map georeferencing.

    GSD range is deliberately not part of this decision.
    """

    crs = meta_get(
        scene_meta,
        "crs",
    )

    if crs is None:
        return False

    if not str(crs).strip():
        return False

    transform_value = meta_get(
        scene_meta,
        "transform",
    )

    if isinstance(
        transform_value,
        (list, tuple),
    ):

        if len(transform_value) != 6:
            return False

        try:
            a, b, c, d, e, f = (
                float(value)
                for value in transform_value
            )

        except (
            TypeError,
            ValueError,
        ):
            return False

    elif all(
        hasattr(
            transform_value,
            name,
        )
        for name in (
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        )
    ):

        try:
            a = float(transform_value.a)
            b = float(transform_value.b)
            c = float(transform_value.c)
            d = float(transform_value.d)
            e = float(transform_value.e)
            f = float(transform_value.f)

        except (
            TypeError,
            ValueError,
        ):
            return False

    else:
        return False

    values = (
        a,
        b,
        c,
        d,
        e,
        f,
    )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        return False

    if (
        abs(a - 1.0) < 1e-12
        and abs(b) < 1e-12
        and abs(c) < 1e-12
        and abs(d) < 1e-12
        and abs(e - 1.0) < 1e-12
        and abs(f) < 1e-12
    ):
        return False

    determinant = (
        a * e
        - b * d
    )

    if abs(determinant) < 1e-15:
        return False

    return True


def resolve_processing_mode(
    scene_meta: Any,
    gcps_requested: bool,
) -> str:

    georeferenced = (
        scene_is_reliably_georeferenced(
            scene_meta
        )
    )

    if (
        gcps_requested
        and not georeferenced
    ):
        raise ValueError(
            "GCPs were supplied, but the input image is not reliably "
            "georeferenced. GCP calibration requires a georeferenced "
            "input raster with a valid CRS and transform."
        )

    if not georeferenced:
        return RELATIVE_ONLY

    if gcps_requested:
        return METRIC_GCP

    return METRIC_PRIOR


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def setup_logging(
    verbose: bool = False,
) -> None:

    level = (
        logging.DEBUG
        if verbose
        else logging.INFO
    )

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def resolve_device(
    device: str,
) -> str:

    if device == "auto":

        if torch.cuda.is_available():
            return "cuda"

        LOGGER.warning(
            "CUDA is not available to PyTorch. "
            "Falling back to CPU."
        )

        return "cpu"

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):

        LOGGER.warning(
            "CUDA was requested but is not available to PyTorch. "
            "Falling back to CPU."
        )

        return "cpu"

    return device


def ensure_exists(
    path: Path,
    description: str,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not created as expected: "
            f"{path}"
        )


def stage_start(
    name: str,
) -> float:

    LOGGER.info("")
    LOGGER.info("=" * 72)
    LOGGER.info(
        "%s",
        name,
    )
    LOGGER.info("=" * 72)

    return time.perf_counter()


def stage_end(
    name: str,
    start_time: float,
) -> None:

    elapsed = (
        time.perf_counter()
        - start_time
    )

    LOGGER.info(
        "%s completed in %.2f seconds",
        name,
        elapsed,
    )


def update_raster_tags(
    path: Path,
    **tags: str,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot update raster tags; file not found: {path}"
        )

    with rasterio.open(
        path,
        "r+",
    ) as dst:

        dst.update_tags(
            **{
                key: str(value)
                for key, value in tags.items()
            }
        )


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------


def save_run_manifest(
    *,
    output_dir: Path,
    input_path: Path,
    checkpoint_path: Path,
    device: str,
    args: argparse.Namespace,
    scene_meta: Any,
    processing_mode: str,
    gcp_context_path: Path | None = None,
    gcp_fitting_completed: bool = False,
    calibration_mode: str | None = None,
) -> None:

    scene_dict = meta_to_dict(
        scene_meta
    )

    manifest = {
        "pipeline": "DepthWizard prototype",
        "pipeline_phase": "PHASE_2",
        "processing_mode": processing_mode,
        "input": str(
            input_path.resolve()
        ),
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "device": device,
        "arguments": vars(args),
        "scene": {
            "classification": meta_get(
                scene_meta,
                "classification",
            ),
            "kind": meta_get(
                scene_meta,
                "kind",
            ),
            "crs": meta_get(
                scene_meta,
                "crs",
            ),
            "width": meta_get(
                scene_meta,
                "width",
            ),
            "height": meta_get(
                scene_meta,
                "height",
            ),
            "gsd_true_m": meta_get(
                scene_meta,
                "gsd_true_m",
            ),
            "use_dem": meta_get(
                scene_meta,
                "use_dem",
            ),
            "relative_only": meta_get(
                scene_meta,
                "relative_only",
            ),
        },
        "calibration_semantics": {
            "mode": (
                calibration_mode
                if calibration_mode is not None
                else (
                    "NONE"
                    if processing_mode == RELATIVE_ONLY
                    else (
                        "GCP_READY_FOR_PHASE3"
                        if processing_mode == METRIC_GCP
                        else "PRIOR_OR_DEM_EVIDENCE"
                    )
                )
            ),
            "gcp_fitting_completed": bool(
                gcp_fitting_completed
            ),
            "phase2_note": (
                "GCP fitting and metric DSM composition were completed "
                "in this call."
                if (
                    processing_mode == METRIC_GCP
                    and gcp_fitting_completed
                )
                else (
                    "GCP fitting is intentionally deferred to Phase 3."
                    if processing_mode == METRIC_GCP
                    else None
                )
            ),
        },
    }

    if gcp_context_path is not None:
        manifest["gcp"] = {
            "provided": True,
            "context_path": str(
                gcp_context_path.resolve()
            ),
        }

    else:
        manifest["gcp"] = {
            "provided": False,
        }

    manifest["scene_metadata_snapshot"] = (
        scene_dict
    )

    manifest_path = (
        output_dir
        / "run_meta.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            default=str,
        )

    LOGGER.info(
        "Run manifest: %s",
        manifest_path,
    )


# ---------------------------------------------------------------------
# Phase 2 -> Phase 3 calibration request
# ---------------------------------------------------------------------


def save_gcp_calibration_request(
    *,
    output_path: Path,
    gcp_context_path: Path,
    scale_prior: float,
    scale_sigma: float,
    dem_dir: Path,
    ground_dir: Path,
) -> None:

    request = {
        "stage": "PHASE2_TO_PHASE3",
        "processing_mode": METRIC_GCP,
        "calibration_status": "READY_FOR_PHASE3_GCP_FIT",
        "gcp_context": str(
            gcp_context_path.resolve()
        ),
        "reference_surface": {
            "dem_dtm": str(
                dem_dir / "dtm_canonical.tif"
            ),
            "ground_estimate": str(
                ground_dir / "G.dat"
            ),
            "relative_agl": str(
                ground_dir / "R.dat"
            ),
        },
        "prior": {
            "scale_a0": float(
                scale_prior
            ),
            "scale_sigma": float(
                scale_sigma
            ),
            "role": (
                "Bayesian regulariser for Phase 3 GCP "
                "calibration, not the final fitted result."
            ),
        },
        "phase3_fit_contract": {
            "base_model": (
                "z_ref - DTM = a * R + b"
            ),
            "minimum_gcp_count": 1,
            "one_gcp": (
                "solve b with a fixed to the prior"
            ),
            "two_gcps": (
                "solve a,b with prior regularisation"
            ),
            "three_to_seven_gcps": (
                "add planar residual term"
            ),
            "eight_or_more_gcps": (
                "planar model plus residual correction"
            ),
            "leave_one_out": (
                "required when enough GCPs are available"
            ),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            request,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Shared GCP fit + compose implementation
# ---------------------------------------------------------------------


def fit_and_compose_gcp_scene(
    *,
    gcp_context_path: Path,
    ground_dir: Path,
    dem_dir: Path,
    tiling_meta_path: Path,
    scene_meta_path: Path,
    calibration_dir: Path,
    compose_dir: Path,
    inference_dir: Path,
    scale_prior: float = DEFAULT_SCALE_PRIOR,
    scale_sigma: float = DEFAULT_SCALE_SIGMA,
    offset_prior: float = DEFAULT_OFFSET_PRIOR,
    offset_sigma: float = DEFAULT_OFFSET_SIGMA,
    gcp_tau_m: float = DEFAULT_GCP_TAU_M,
    max_agl_m: float = DEFAULT_MAX_AGL_M,
    gcp_loo_ratio_limit: float = DEFAULT_GCP_LOO_RATIO_LIMIT,
    gcp_loo_action: str = DEFAULT_GCP_LOO_ACTION,
    gcp_min_r_spread: float = DEFAULT_GCP_MIN_R_SPREAD,
    gcp_z_reference: str = DEFAULT_GCP_Z_REFERENCE,
    gcp_geoid_grid: Path | None = DEFAULT_GCP_GEOID_GRID,
    gcp_crs: str | None = None,
    reuse_existing_calibration: bool = False,
    write_native_compose: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Fit scene-specific GCP calibration and compose metric DSM."""

    calibration_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    compose_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_path = calibration_dir / "calibration.json"

    if reuse_existing_calibration:
        ensure_exists(
            calibration_path,
            "Existing calibration.json",
        )
        LOGGER.info(
            "Reusing existing calibration: %s",
            calibration_path,
        )
    else:
        calibration_path = calibrate_with_gcps(
            gcp_context_path=gcp_context_path,
            ground_dir=ground_dir,
            dem_dir=dem_dir,
            tiling_meta_path=tiling_meta_path,
            output_dir=calibration_dir,
            scale_prior=scale_prior,
            scale_sigma=scale_sigma,
            offset_prior=offset_prior,
            offset_sigma=offset_sigma,
            tau=gcp_tau_m,
            max_agl_m=max_agl_m,
            gcp_loo_ratio_limit=gcp_loo_ratio_limit,
            gcp_loo_action=gcp_loo_action,
            gcp_min_r_spread=gcp_min_r_spread,
            gcp_z_reference=gcp_z_reference,
            gcp_geoid_grid=gcp_geoid_grid,
            gcp_crs=gcp_crs,
        )

    with calibration_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        calibration = json.load(f)

    if not isinstance(calibration, dict):
        raise ValueError(
            "Calibration output must be a JSON object."
        )

    mode = calibration.get("mode")
    allowed_modes = {
        "GCP_FITTED",
        "GCP_FITTED_UNVALIDATED",
        "GCP_OFFSET_ONLY",
        "PRIOR_FALLBACK_FROM_GCP",
    }

    if mode not in allowed_modes:
        raise ValueError(
            "Unexpected GCP calibration mode: "
            f"{mode!r}. Expected one of {sorted(allowed_modes)}."
        )

    contract = calibration.get("contract")
    if contract != "z_ref - DTM = a * R + b":
        raise ValueError(
            "Calibration contract mismatch. Expected: "
            "z_ref - DTM = a * R + b"
        )

    if mode != "PRIOR_FALLBACK_FROM_GCP":
        scale = float(
            calibration.get(
                "a",
                calibration.get("scale"),
            )
        )
        offset = float(
            calibration.get(
                "b",
                calibration.get("offset"),
            )
        )
        if scale <= 0.0:
            raise ValueError(
                "Scene-specific calibration scale a must be positive."
            )

        LOGGER.info(
            "GCP calibration mode : %s",
            mode,
        )
        LOGGER.info(
            "a = %.9f",
            scale,
        )
        LOGGER.info(
            "b = %.9f m",
            offset,
        )
    else:
        LOGGER.warning(
            "Using prior calibration fallback after failed GCP quality gate."
        )

    compose_dsm(
        ground_dir=ground_dir,
        dem_dir=dem_dir,
        calibration_path=calibration_path,
        tiling_meta_path=tiling_meta_path,
        scene_meta_path=scene_meta_path,
        output_dir=compose_dir,
        max_agl_m=max_agl_m,
        inference_dir=inference_dir,
        write_native_products=write_native_compose,
    )

    dsm_path = compose_dir / "dsm.tif"
    ndsm_path = compose_dir / "ndsm.tif"
    dtm_path = compose_dir / "dtm.tif"

    for path, description in (
        (dsm_path, "Final DSM"),
        (ndsm_path, "Final nDSM"),
        (dtm_path, "Final DTM"),
    ):
        ensure_exists(path, description)

    if mode == "GCP_FITTED":
        final_calibration_mode = "GCP_FITTED"
        final_status = "GCP_FIT_VALIDATED"
        gcp_fitted = "true"
    elif mode == "GCP_FITTED_UNVALIDATED":
        final_calibration_mode = "GCP_FITTED_UNVALIDATED"
        final_status = "GCP_FIT_UNVALIDATED"
        gcp_fitted = "true"
    elif mode == "GCP_OFFSET_ONLY":
        final_calibration_mode = "GCP_OFFSET_ONLY"
        final_status = "GCP_OFFSET_VALIDATED"
        gcp_fitted = "false"
    else:
        final_calibration_mode = "PRIOR_OR_DEM_EVIDENCE"
        final_status = "GCP_FIT_REJECTED_FALLBACK_TO_PRIOR"
        gcp_fitted = "false"

    vertical_info = calibration.get(
        "gcp_vertical_reference",
        {},
    )
    vertical_source = str(
        vertical_info.get(
            "output_reference",
            "unknown",
        )
    )
    scale_source = str(
        calibration.get(
            "scale_source",
            "unknown",
        )
    )
    offset_source = str(
        calibration.get(
            "offset_source",
            "unknown",
        )
    )

    common_tags = {
        "PROCESSING_MODE": METRIC_GCP,
        "CALIBRATION_MODE": final_calibration_mode,
        "CALIBRATION_STATUS": final_status,
        "METRIC": "true",
        "GCP_FITTED": gcp_fitted,
        "GCP_VERTICAL_REFERENCE": vertical_source,
        "SCALE_SOURCE": scale_source,
        "OFFSET_SOURCE": offset_source,
    }

    update_raster_tags(
        dsm_path,
        **common_tags,
    )

    update_raster_tags(
        ndsm_path,
        **common_tags,
    )

    update_raster_tags(
        dtm_path,
        **{
            "PROCESSING_MODE": METRIC_GCP,
            "CALIBRATION_MODE": final_calibration_mode,
            "CALIBRATION_STATUS": final_status,
            "METRIC": "true",
            "GCP_VERTICAL_REFERENCE": vertical_source,
        },
    )

    return calibration_path, calibration


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def run_pipeline(
    args: argparse.Namespace,
) -> Path:

    input_path = (
        Path(args.input)
        .expanduser()
        .resolve()
    )

    checkpoint_path = (
        Path(args.checkpoint)
        .expanduser()
        .resolve()
    )

    external_model_dir = (
        Path(args.external_model_dir)
        .expanduser()
        .resolve()
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {input_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    if not external_model_dir.exists():
        raise FileNotFoundError(
            "Depth Anything V2 directory not found: "
            f"{external_model_dir}"
        )

    output_root = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    scene_name = (
        args.scene_name
        if args.scene_name
        else input_path.stem
    )

    scene_dir = (
        output_root
        / scene_name
    )

    meta_dir = scene_dir / "meta"
    tiling_dir = scene_dir / "tiling"
    inference_dir = scene_dir / "inference"
    dem_dir = scene_dir / "dem"
    ground_dir = scene_dir / "ground"
    calibration_dir = (
        scene_dir / "calibration"
    )
    compose_dir = (
        scene_dir / "compose"
    )

    for directory in (
        meta_dir,
        tiling_dir,
        inference_dir,
        dem_dir,
        ground_dir,
        calibration_dir,
        compose_dir,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    device = resolve_device(
        args.device
    )

    LOGGER.info(
        "DepthWizard prototype pipeline"
    )

    LOGGER.info(
        "Input       : %s",
        input_path,
    )

    LOGGER.info(
        "Output      : %s",
        scene_dir,
    )

    LOGGER.info(
        "Checkpoint  : %s",
        checkpoint_path,
    )

    LOGGER.info(
        "Model dir   : %s",
        external_model_dir,
    )

    LOGGER.info(
        "Device      : %s",
        device,
    )

    # =============================================================
    # 1. Scene metadata and routing
    # =============================================================

    start = stage_start(
        "1/7 — Scene metadata and routing"
    )

    scene_meta_path = (
        meta_dir
        / "scene_meta.json"
    )

    scene_meta = resolve_scene(
        input_path
    )

    # -------------------------------------------------------------
    # PHASE 2 FIX:
    #
    # Recover valid CRS/transform/GSD directly from raster when
    # Phase 1 metadata conservatively returned NONE.
    # -------------------------------------------------------------

    scene_meta = (
        repair_phase2_scene_metadata(
            input_path,
            scene_meta,
        )
    )

    save_scene_meta(
        scene_meta,
        scene_meta_path,
    )

    ensure_exists(
        scene_meta_path,
        "Scene metadata",
    )

    processing_mode = (
        resolve_processing_mode(
            scene_meta,
            args.gcps is not None,
        )
    )

    LOGGER.info(
        "Classification : %s",
        meta_get(
            scene_meta,
            "classification",
        ),
    )

    LOGGER.info(
        "Kind           : %s",
        meta_get(
            scene_meta,
            "kind",
        ),
    )

    LOGGER.info(
        "CRS            : %s",
        meta_get(
            scene_meta,
            "crs",
        ),
    )

    LOGGER.info(
        "True GSD       : %.6f m",
        float(
            meta_get(
                scene_meta,
                "gsd_true_m",
            )
            if meta_get(
                scene_meta,
                "gsd_true_m",
            ) is not None
            else float("nan")
        ),
    )

    LOGGER.info(
        "Metadata use_dem: %s",
        meta_get(
            scene_meta,
            "use_dem",
        ),
    )

    LOGGER.info(
        "Relative only  : %s",
        meta_get(
            scene_meta,
            "relative_only",
        ),
    )

    LOGGER.info(
        "PROCESSING MODE: %s",
        processing_mode,
    )

    gcp_set = None

    if args.gcps is not None:

        if args.scale is not None:
            raise ValueError(
                "--scale cannot be used together with --gcps. "
                "The GCP fit will replace the prior in Phase 3."
            )

        gcp_set = load_gcps(
            args.gcps,
            default_crs=args.gcp_crs,
        )

        LOGGER.info(
            "GCP file      : %s",
            gcp_set.source_path,
        )

        LOGGER.info(
            "GCP CRS       : %s",
            gcp_set.source_crs,
        )

        LOGGER.info(
            "GCP count     : %d",
            len(
                gcp_set.gcps
            ),
        )

    stage_end(
        "Scene metadata and routing",
        start,
    )

    # =============================================================
    # 2. Native tiling + GCP mapping
    # =============================================================

    start = stage_start(
        "2/7 — Native tiling and GCP mapping"
    )

    tiling_meta_path = (
        tiling_dir
        / "tiling_meta.json"
    )

    prepare_tiles(
        input_path=input_path,
        scene_meta_path=scene_meta_path,
        output_dir=tiling_dir,
        train_gsd_m=args.train_gsd_m,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    ensure_exists(
        tiling_meta_path,
        "Tiling metadata",
    )

    LOGGER.info(
        "Tiles prepared in: %s",
        tiling_dir / "tiles",
    )

    gcp_context_path = None

    if gcp_set is not None:

        gcp_context = build_gcp_context(
            gcp_set,
            tiling_meta_path,
        )

        gcp_context_path = (
            calibration_dir
            / "gcp_context.json"
        )

        save_gcp_context(
            gcp_context,
            gcp_context_path,
        )

        LOGGER.info(
            "GCP context    : %s",
            gcp_context_path,
        )

        LOGGER.info(
            "GCP coordinates mapped successfully to native and "
            "canonical grids."
        )

        for item in gcp_context["gcps"]:

            LOGGER.info(
                "GCP %s: native(row=%.3f,col=%.3f), "
                "canonical(row=%.3f,col=%.3f), z=%.3f m",
                item["id"],
                item["native_pixel"]["row"],
                item["native_pixel"]["col"],
                item["canonical_pixel"]["row"],
                item["canonical_pixel"]["col"],
                item["z"],
            )

    stage_end(
        "Native tiling and GCP mapping",
        start,
    )

    # =============================================================
    # 3. Depth inference
    # =============================================================

    start = stage_start(
        "3/7 — Depth Anything V2 inference"
    )

    run_inference(
        tiles_dir=(
            tiling_dir / "tiles"
        ),
        tiling_meta_path=tiling_meta_path,
        output_dir=inference_dir,
        checkpoint_path=checkpoint_path,
        external_model_dir=external_model_dir,
        device=device,
        use_flip_tta=not args.no_flip_tta,
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
    )

    ensure_exists(
        inference_dir / "P.dat",
        "Relative depth prediction",
    )

    ensure_exists(
        inference_dir / "relative_dsm.tif",
        "Relative DSM raster",
    )

    LOGGER.info(
        "Relative depth map: %s",
        inference_dir / "P.dat",
    )

    if processing_mode == RELATIVE_ONLY:

        update_raster_tags(
            inference_dir / "relative_dsm.tif",
            PROCESSING_MODE=RELATIVE_ONLY,
            CALIBRATION_MODE="NONE",
            METRIC="false",
            GCP_FITTED="false",
        )

    elif processing_mode == METRIC_GCP:

        update_raster_tags(
            inference_dir / "relative_dsm.tif",
            PROCESSING_MODE=METRIC_GCP,
            CALIBRATION_MODE="GCP_READY_FOR_PHASE3",
            CALIBRATION_STATUS="NOT_YET_FITTED",
            METRIC="false",
            GCP_FITTED="false",
        )

    else:

        update_raster_tags(
            inference_dir / "relative_dsm.tif",
            PROCESSING_MODE=METRIC_PRIOR,
            CALIBRATION_MODE="PRIOR_OR_DEM_EVIDENCE",
            METRIC="false",
            GCP_FITTED="false",
        )

    stage_end(
        "Depth Anything V2 inference",
        start,
    )

    # =============================================================
    # RELATIVE_ONLY route
    # =============================================================

    if processing_mode == RELATIVE_ONLY:

        LOGGER.warning("")
        LOGGER.warning(
            "Metric processing is disabled because the input "
            "is not reliably georeferenced."
        )

        save_run_manifest(
            output_dir=scene_dir,
            input_path=input_path,
            checkpoint_path=checkpoint_path,
            device=device,
            args=args,
            scene_meta=scene_meta,
            processing_mode=processing_mode,
            gcp_context_path=None,
        )

        LOGGER.info("")
        LOGGER.info("=" * 72)
        LOGGER.info(
            "RELATIVE PIPELINE COMPLETE"
        )
        LOGGER.info("=" * 72)

        LOGGER.info(
            "Scene directory : %s",
            scene_dir,
        )

        LOGGER.info(
            "Relative depth  : %s",
            inference_dir / "P.dat",
        )

        LOGGER.info(
            "Relative DSM    : %s",
            inference_dir / "relative_dsm.tif",
        )

        return scene_dir

    # =============================================================
    # 4. FABDEM
    # =============================================================

    start = stage_start(
        "4/7 — FABDEM acquisition and alignment"
    )

    get_fabdem(
        scene_meta_path=scene_meta_path,
        tiling_meta_path=tiling_meta_path,
        output_dir=dem_dir,
        cache_dir=(
            Path(
                args.fabdem_cache
            )
            .expanduser()
            .resolve()
        ),
    )

    ensure_exists(
        dem_dir / "dtm.dat",
        "FABDEM DTM",
    )

    ensure_exists(
        dem_dir / "dtm_canonical.tif",
        "Canonical FABDEM DTM",
    )

    stage_end(
        "FABDEM acquisition and alignment",
        start,
    )

    # =============================================================
    # 5. Ground estimation
    # =============================================================

    start = stage_start(
        "5/7 — Ground estimation"
    )

    estimate_ground(
        inference_dir=inference_dir,
        tiling_meta_path=tiling_meta_path,
        output_dir=ground_dir,
        ground_window_m=args.ground_window_m,
        smooth_sigma_m=args.smooth_sigma_m,
        semantic_mask_path=args.semantic_ground_mask,
    )

    ensure_exists(
        ground_dir / "G.dat",
        "Ground surface estimate",
    )

    ensure_exists(
        ground_dir / "R.dat",
        "Relative above-ground surface",
    )

    LOGGER.info(
        "Ground estimate: %s",
        ground_dir / "G.dat",
    )

    LOGGER.info(
        "Relative AGL: %s",
        ground_dir / "R.dat",
    )

    stage_end(
        "Ground estimation",
        start,
    )

    # =============================================================
    # 6. GCP calibration + metric DSM
    # =============================================================

    if processing_mode == METRIC_GCP:

        if gcp_context_path is None:
            raise RuntimeError(
                "Internal error: METRIC_GCP mode has no GCP context."
            )

        start = stage_start(
            "6/7 — Scene-specific GCP calibration and DSM composition"
        )

        calibration_path, calibration = fit_and_compose_gcp_scene(
            gcp_context_path=gcp_context_path,
            ground_dir=ground_dir,
            dem_dir=dem_dir,
            tiling_meta_path=tiling_meta_path,
            scene_meta_path=scene_meta_path,
            calibration_dir=calibration_dir,
            compose_dir=compose_dir,
            inference_dir=inference_dir,
            scale_prior=args.scale_prior,
            scale_sigma=args.scale_sigma,
            offset_prior=args.offset_prior,
            offset_sigma=args.offset_sigma,
            gcp_tau_m=args.gcp_tau_m,
            max_agl_m=args.max_agl_m,
            gcp_loo_ratio_limit=args.gcp_loo_ratio_limit,
            gcp_loo_action=args.gcp_loo_action,
            gcp_min_r_spread=args.gcp_min_r_spread,
            gcp_z_reference=args.gcp_z_reference,
            gcp_geoid_grid=Path(args.gcp_geoid_grid),
            gcp_crs=args.gcp_crs,
            write_native_compose=args.write_native_compose,
        )

        stage_end(
            "Scene-specific GCP calibration and DSM composition",
            start,
        )

        calibration_mode = str(
            calibration.get(
                "mode",
                "UNKNOWN",
            )
        )

        save_run_manifest(
            output_dir=scene_dir,
            input_path=input_path,
            checkpoint_path=checkpoint_path,
            device=device,
            args=args,
            scene_meta=scene_meta,
            processing_mode=processing_mode,
            gcp_context_path=gcp_context_path,
            gcp_fitting_completed=True,
            calibration_mode=calibration_mode,
        )

        LOGGER.info("")
        LOGGER.info("=" * 72)
        LOGGER.info("PIPELINE COMPLETE")
        LOGGER.info("=" * 72)
        LOGGER.info("Scene directory : %s", scene_dir)
        LOGGER.info("Processing mode : %s", processing_mode)
        LOGGER.info("Calibration mode: %s", calibration_mode)
        LOGGER.info("DSM             : %s", compose_dir / "dsm.tif")
        LOGGER.info("nDSM            : %s", compose_dir / "ndsm.tif")
        LOGGER.info("DTM             : %s", compose_dir / "dtm.tif")

        return scene_dir

    # =============================================================
    # Prior route
    # =============================================================

    start = stage_start(
        "6/7 — Prior height calibration"
    )

    calibrate_relative_height(
        ground_dir=ground_dir,
        output_dir=calibration_dir,
        scale_prior=args.scale_prior,
        scale_sigma=args.scale_sigma,
        scale_override=args.scale,
    )

    calibration_path = (
        calibration_dir
        / "calibration.json"
    )

    ensure_exists(
        calibration_path,
        "Calibration metadata",
    )

    with calibration_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        calibration = json.load(f)

    LOGGER.info(
        "Calibration mode: PRIOR_OR_DEM_EVIDENCE"
    )

    LOGGER.info(
        "Scale a = %s",
        calibration.get(
            "scale"
        ),
    )

    LOGGER.info(
        "Offset b = %s",
        calibration.get(
            "offset"
        ),
    )

    stage_end(
        "Prior height calibration",
        start,
    )

    # =============================================================
    # DSM composition
    # =============================================================

    start = stage_start(
        "7/7 — DSM composition and export"
    )

    compose_dsm(
        ground_dir=ground_dir,
        dem_dir=dem_dir,
        calibration_path=calibration_path,
        tiling_meta_path=tiling_meta_path,
        scene_meta_path=scene_meta_path,
        output_dir=compose_dir,
        max_agl_m=args.max_agl_m,
        inference_dir=inference_dir,
        write_native_products=args.write_native_compose,
    )

    ensure_exists(
        compose_dir / "dsm.tif",
        "Final DSM",
    )

    ensure_exists(
        compose_dir / "ndsm.tif",
        "Final nDSM",
    )

    ensure_exists(
        compose_dir / "dtm.tif",
        "Final DTM",
    )

    update_raster_tags(
        compose_dir / "dsm.tif",
        PROCESSING_MODE=METRIC_PRIOR,
        CALIBRATION_MODE="PRIOR_OR_DEM_EVIDENCE",
        CALIBRATION_STATUS="PRIOR_CALIBRATION",
        METRIC="true",
        GCP_FITTED="false",
    )

    update_raster_tags(
        compose_dir / "ndsm.tif",
        PROCESSING_MODE=METRIC_PRIOR,
        CALIBRATION_MODE="PRIOR_OR_DEM_EVIDENCE",
        CALIBRATION_STATUS="PRIOR_CALIBRATION",
        METRIC="true",
        GCP_FITTED="false",
    )

    update_raster_tags(
        compose_dir / "dtm.tif",
        PROCESSING_MODE=METRIC_PRIOR,
        CALIBRATION_MODE="PRIOR_OR_DEM_EVIDENCE",
        METRIC="true",
    )

    stage_end(
        "DSM composition and export",
        start,
    )

    save_run_manifest(
        output_dir=scene_dir,
        input_path=input_path,
        checkpoint_path=checkpoint_path,
        device=device,
        args=args,
        scene_meta=scene_meta,
        processing_mode=processing_mode,
        gcp_context_path=None,
    )

    LOGGER.info("")
    LOGGER.info("=" * 72)
    LOGGER.info(
        "PIPELINE COMPLETE"
    )
    LOGGER.info("=" * 72)

    LOGGER.info(
        "Scene directory : %s",
        scene_dir,
    )

    LOGGER.info(
        "Processing mode : %s",
        processing_mode,
    )

    LOGGER.info(
        "DSM             : %s",
        compose_dir / "dsm.tif",
    )

    LOGGER.info(
        "nDSM            : %s",
        compose_dir / "ndsm.tif",
    )

    LOGGER.info(
        "DTM             : %s",
        compose_dir / "dtm.tif",
    )

    return scene_dir


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "DepthWizard prototype: "
            "RGB -> relative depth -> DEM/ground -> "
            "prior or GCP metric calibration"
        )
    )

    # -------------------------------------------------------------
    # Input
    # -------------------------------------------------------------

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input RGB remote-sensing image.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/runs",
        help="Root directory for pipeline outputs.",
    )

    parser.add_argument(
        "--scene-name",
        default=None,
        help=(
            "Optional output scene name. "
            "Defaults to input filename stem."
        ),
    )

    # -------------------------------------------------------------
    # GCP
    # -------------------------------------------------------------

    parser.add_argument(
        "--gcps",
        default=None,
        help=(
            "Optional GCP CSV or JSON file. "
            "CSV requires id,x,y,z columns."
        ),
    )

    parser.add_argument(
        "--gcp-crs",
        default=None,
        help=(
            "CRS of GCP X/Y coordinates. "
            "Example: EPSG:25830."
        ),
    )

    parser.add_argument(
        "--gcp-z-reference",
        choices=["as_supplied", "ellipsoidal", "orthometric"],
        default=DEFAULT_GCP_Z_REFERENCE,
        help=(
            "Vertical reference of GCP Z values. Use 'ellipsoidal' "
            "for the Spanish EGM08-REDNAP H=Z-N conversion."
        ),
    )

    parser.add_argument(
        "--gcp-geoid-grid",
        default=str(DEFAULT_GCP_GEOID_GRID),
        help=(
            "EGM08-REDNAP geoid GeoTIFF used when GCP Z is ellipsoidal."
        ),
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the fine-tuned checkpoint.",
    )

    parser.add_argument(
        "--external-model-dir",
        default="external/Depth-Anything-V2",
        help="Path to Depth Anything V2 source directory.",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
        help="Inference device.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--no-flip-tta",
        action="store_true",
        help="Disable horizontal flip TTA.",
    )

    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 inference.",
    )

    # -------------------------------------------------------------
    # Tiling
    # -------------------------------------------------------------

    parser.add_argument(
        "--train-gsd-m",
        type=float,
        default=DEFAULT_TRAIN_GSD_M,
        help=(
            "Canonical downstream processing GSD in metres/pixel."
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Native RGB tile size.",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help="Native tile overlap fraction.",
    )

    # -------------------------------------------------------------
    # FABDEM
    # -------------------------------------------------------------

    parser.add_argument(
        "--fabdem-cache",
        default="data/dem_cache/fabdem",
        help="Local FABDEM cache directory.",
    )

    # -------------------------------------------------------------
    # Ground
    # -------------------------------------------------------------

    parser.add_argument(
        "--ground-window-m",
        type=float,
        default=DEFAULT_GROUND_WINDOW_M,
        help="Ground estimation window in metres.",
    )

    parser.add_argument(
        "--smooth-sigma-m",
        type=float,
        default=DEFAULT_SMOOTH_SIGMA_M,
        help="Ground smoothing sigma in metres.",
    )

    parser.add_argument(
        "--semantic-ground-mask",
        default=None,
        help=(
            "Optional ground/road/low-vegetation/water mask. "
            "Nonzero pixels are treated as ground support."
        ),
    )

    # -------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------

    parser.add_argument(
        "--scale-prior",
        type=float,
        # Keep this aligned with DEFAULT_SCALE_PRIOR in calibrate.py.
        default=DEFAULT_SCALE_PRIOR,
        help="Global model-to-metre scale prior.",
    )

    parser.add_argument(
        "--scale-sigma",
        type=float,
        # Keep this aligned with DEFAULT_SCALE_SIGMA in calibrate.py.
        default=DEFAULT_SCALE_SIGMA,
        help="Uncertainty on scale prior.",
    )

    parser.add_argument(
        "--offset-prior",
        type=float,
        # Keep this aligned with DEFAULT_OFFSET_PRIOR in calibrate.py.
        default=DEFAULT_OFFSET_PRIOR,
        help="Prior mean for scene-specific offset b.",
    )

    parser.add_argument(
        "--offset-sigma",
        type=float,
        # Keep this aligned with DEFAULT_OFFSET_SIGMA in calibrate.py.
        default=DEFAULT_OFFSET_SIGMA,
        help="Prior standard deviation for scene-specific offset b.",
    )

    parser.add_argument(
        "--gcp-tau-m",
        type=float,
        # Keep this aligned with DEFAULT_GCP_TAU_M in calibrate.py.
        default=DEFAULT_GCP_TAU_M,
        help="GCP observation noise scale in metres.",
    )

    parser.add_argument(
        "--gcp-loo-ratio-limit",
        type=float,
        # Keep this aligned with DEFAULT_GCP_LOO_RATIO_LIMIT in calibrate.py.
        default=DEFAULT_GCP_LOO_RATIO_LIMIT,
        help="Maximum allowed LOO/fit RMSE ratio.",
    )

    parser.add_argument(
        "--gcp-loo-action",
        choices=["error", "fallback_to_prior"],
        # Keep this aligned with DEFAULT_GCP_LOO_ACTION in calibrate.py.
        default=DEFAULT_GCP_LOO_ACTION,
        help="Action when GCP LOO quality gate fails.",
    )

    parser.add_argument(
        "--gcp-min-r-spread",
        type=float,
        # Keep this aligned with DEFAULT_GCP_MIN_R_SPREAD in calibrate.py.
        default=DEFAULT_GCP_MIN_R_SPREAD,
        help="Minimum relative-height spread warning threshold.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "Explicit scale override. "
            "Not allowed with --gcps."
        ),
    )

    # -------------------------------------------------------------
    # DSM
    # -------------------------------------------------------------

    parser.add_argument(
        "--max-agl-m",
        type=float,
        default=DEFAULT_MAX_AGL_M,
        help="Maximum allowed predicted AGL.",
    )

    parser.add_argument(
        "--write-native-compose",
        action="store_true",
        help=(
            "Write additional native-resolution final GeoTIFFs "
            "(*_native.tif). Disabled by default; canonical 1 m "
            "products are the primary final outputs."
        ),
    )

    # -------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:

    parser = build_parser()
    args = parser.parse_args()

    setup_logging(
        args.verbose
    )

    try:

        run_pipeline(
            args
        )

        return 0

    except KeyboardInterrupt:

        LOGGER.error(
            "Pipeline interrupted by user."
        )

        return 130

    except Exception as exc:

        LOGGER.exception(
            "Pipeline failed: %s",
            exc,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )