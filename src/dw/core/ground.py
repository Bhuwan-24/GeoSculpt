"""
DepthWizard prototype ground estimation.

Pipeline role
-------------
    infer.py
        ↓
      P.dat
        ↓
    estimate ground surface G
        ↓
    R = max(P - G, 0)
        ↓
    calibrate.py

Ground-estimation paths
-----------------------
1. semantic-mask:
   Normalised Gaussian convolution over an eroded ground-class mask.

2. opening-fallback:
   Gaussian smoothing followed by a true morphological opening
   (grey erosion + grey dilation), then Gaussian smoothing.

3. min-pool-last-resort:
   Legacy minimum-pool envelope used only when the opening kernel cannot
   be sized safely.

The selected method is written to ground_meta.json and a per-pixel
ground_method.dat map:
    1 = semantic-mask
    2 = opening-fallback
    3 = min-pool-last-resort
"""

from __future__ import annotations

from collections.abc import Mapping
import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    grey_opening,
    minimum_filter,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_GROUND_WINDOW_M = 30.0
DEFAULT_SMOOTH_SIGMA_M = 3.0

DEFAULT_MIN_R_M = 0.0
DEFAULT_MAX_R_M = 500.0

METHOD_SEMANTIC_MASK = 1
METHOD_OPENING_FALLBACK = 2
METHOD_MIN_POOL_LAST_RESORT = 3


def estimate_ground(
    *,
    inference_dir: str | Path,
    tiling_meta_path: str | Path,
    output_dir: str | Path,
    ground_window_m: float = DEFAULT_GROUND_WINDOW_M,
    smooth_sigma_m: float = DEFAULT_SMOOTH_SIGMA_M,
    semantic_mask_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Estimate the ground surface G and relative above-ground component R.

    semantic_mask_path, when supplied, must identify a raster/mask where
    nonzero pixels represent ground/road/low-vegetation/water classes.
    """
    inference_dir = Path(inference_dir).expanduser().resolve()
    tiling_meta_path = Path(tiling_meta_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if semantic_mask_path is not None:
        semantic_mask_path = (
            Path(semantic_mask_path).expanduser().resolve()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _validate_parameters(
        ground_window_m=ground_window_m,
        smooth_sigma_m=smooth_sigma_m,
    )

    inference_meta_path = inference_dir / "inference_meta.json"
    if not inference_meta_path.exists():
        raise FileNotFoundError(
            f"Missing inference metadata: {inference_meta_path}"
        )

    inference_meta = _read_json(inference_meta_path)
    grid = _load_grid(inference_meta)

    if tiling_meta_path.exists():
        _validate_grid_consistency(
            inference_grid=grid,
            tiling_meta=_read_json(tiling_meta_path),
        )

    height = grid["height"]
    width = grid["width"]
    gsd_m = grid["gsd_m"]

    P_path = inference_dir / "P.dat"
    if not P_path.exists():
        raise FileNotFoundError(
            f"Missing relative prediction: {P_path}"
        )

    expected_bytes = (
        height
        * width
        * np.dtype(np.float32).itemsize
    )
    if P_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"P.dat has wrong size: {P_path.stat().st_size} bytes; "
            f"expected {expected_bytes}."
        )

    P = np.fromfile(
        P_path,
        dtype=np.float32,
    ).reshape(height, width)

    valid = np.isfinite(P)
    if not np.any(valid):
        raise ValueError("P.dat contains no finite prediction values.")

    P_filled = _fill_invalid(P, valid)

    window_px = _metres_to_odd_pixels(
        ground_window_m,
        gsd_m,
    )

    smooth_sigma_px = max(
        0.5,
        float(smooth_sigma_m / gsd_m),
    )

    LOGGER.info(
        "Ground estimation: GSD=%.3f m, window=%d px, "
        "Gaussian sigma=%.2f px",
        gsd_m,
        window_px,
        smooth_sigma_px,
    )

    method_code = METHOD_OPENING_FALLBACK
    semantic_mask = None

    if semantic_mask_path is not None:
        semantic_mask = _load_semantic_mask(
            semantic_mask_path,
            shape=(height, width),
            grid=grid,
        )

    if semantic_mask is not None:
        LOGGER.info(
            "Ground path: semantic-mask (%s)",
            semantic_mask_path,
        )

        G, support = _estimate_ground_from_semantic_mask(
            P=P_filled,
            semantic_mask=semantic_mask,
            smooth_sigma_px=smooth_sigma_px,
        )

        method_code = METHOD_SEMANTIC_MASK

    else:
        try:
            G, support = _estimate_ground_opening(
                P=P_filled,
                window_px=window_px,
                smooth_sigma_px=smooth_sigma_px,
                shape=(height, width),
            )
            method_code = METHOD_OPENING_FALLBACK

            LOGGER.info(
                "Ground path: opening-fallback "
                "(morphological opening)"
            )

        except ValueError as exc:
            LOGGER.warning(
                "Morphological opening unavailable: %s. "
                "Using min-pool last resort.",
                exc,
            )

            G = _estimate_ground_min_pool(
                P=P_filled,
                window_px=window_px,
                smooth_sigma_px=smooth_sigma_px,
            )
            support = np.ones(
                (height, width),
                dtype=np.float32,
            )
            method_code = METHOD_MIN_POOL_LAST_RESORT

            LOGGER.warning(
                "Ground path: min-pool-last-resort"
            )

    G = np.asarray(
        G,
        dtype=np.float32,
    )

    R = (
        P_filled
        - G
    ).astype(
        np.float32,
        copy=False,
    )

    R = np.maximum(
        R,
        DEFAULT_MIN_R_M,
    )

    extreme = (
        R > DEFAULT_MAX_R_M
    )

    if np.any(
        extreme
        & valid
    ):
        count = int(
            np.count_nonzero(
                extreme
                & valid
            )
        )
        LOGGER.warning(
            "%d pixels exceed the prototype relative-height "
            "limit of %.1f model units; they will be masked.",
            count,
            DEFAULT_MAX_R_M,
        )

    valid_R = (
        valid
        & np.isfinite(R)
        & ~extreme
    )

    G_out = G.copy()
    R_out = R.copy()

    G_out[~valid] = np.nan
    R_out[~valid_R] = np.nan

    confidence = _estimate_ground_confidence(
        P=P_filled,
        G=G,
        valid=valid,
        gsd_m=gsd_m,
        support=support,
    )
    confidence[~valid] = 0.0

    method_map = np.full(
        (height, width),
        np.uint8(method_code),
        dtype=np.uint8,
    )
    method_map[~valid] = 0

    G_path = output_dir / "G.dat"
    R_path = output_dir / "R.dat"
    mask_path = output_dir / "ground_mask.dat"
    confidence_path = output_dir / "ground_confidence.dat"
    method_path = output_dir / "ground_method.dat"

    G_out.astype(
        np.float32,
        copy=False,
    ).tofile(G_path)

    R_out.astype(
        np.float32,
        copy=False,
    ).tofile(R_path)

    valid_R.astype(
        np.uint8,
        copy=False,
    ).tofile(mask_path)

    confidence.astype(
        np.float32,
        copy=False,
    ).tofile(confidence_path)

    method_map.tofile(method_path)

    finite_R = (
        valid_R
        & np.isfinite(R_out)
    )

    if np.any(finite_R):
        r_values = R_out[
            finite_R
        ].astype(
            np.float64,
            copy=False,
        )
        r_stats = {
            "min": float(np.min(r_values)),
            "max": float(np.max(r_values)),
            "mean": float(np.mean(r_values)),
            "median": float(np.median(r_values)),
            "p95": float(np.percentile(r_values, 95)),
        }
    else:
        r_stats = {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }

    metadata = {
        "stage": "S2_GROUND",
        "input": {
            "relative_prediction": str(P_path),
            "semantic_mask": (
                str(semantic_mask_path)
                if semantic_mask_path is not None
                else None
            ),
        },
        "method": {
            "code": int(method_code),
            "name": _method_name(method_code),
            "ground_window_m": float(ground_window_m),
            "smooth_sigma_m": float(smooth_sigma_m),
            "ground_window_px": int(window_px),
            "smooth_sigma_px": float(smooth_sigma_px),
            "semantic_mask_used": bool(
                semantic_mask is not None
            ),
        },
        "outputs": {
            "ground": str(G_path),
            "relative_above_ground": str(R_path),
            "valid_mask": str(mask_path),
            "confidence": str(confidence_path),
            "method_map": str(method_path),
        },
        "statistics": {
            "valid_fraction": (
                float(np.count_nonzero(valid_R))
                / float(valid_R.size)
            ),
            "relative_height": r_stats,
            "mean_ground_confidence": (
                float(confidence[valid].mean())
                if np.any(valid)
                else 0.0
            ),
        },
        "warnings": [
            (
                "Ground is estimated from model-relative geometry unless "
                "a semantic ground mask is explicitly supplied; it is not "
                "an independently measured terrain surface."
            ),
        ],
    }

    metadata_path = output_dir / "ground_meta.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "Ground estimation complete."
    )
    LOGGER.info(
        "Ground path : %s",
        _method_name(method_code),
    )
    LOGGER.info(
        "Ground      : %s",
        G_path,
    )
    LOGGER.info(
        "Relative AGL: %s",
        R_path,
    )

    return G_path, R_path


def _estimate_ground_from_semantic_mask(
    *,
    P: np.ndarray,
    semantic_mask: np.ndarray,
    smooth_sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate G from an eroded ground-class mask.

    G = gaussian_filter(P * mask, sigma) /
        gaussian_filter(mask, sigma)

    Where ground support is sparse, a wider blur is used locally.
    """
    mask = (
        np.asarray(
            semantic_mask,
            dtype=bool,
        )
    )

    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )

    if not np.any(eroded):
        raise ValueError(
            "Semantic ground mask contains no pixels after erosion."
        )

    sigma = max(
        0.5,
        float(smooth_sigma_px),
    )

    numerator = gaussian_filter(
        P * eroded.astype(np.float32),
        sigma=sigma,
        mode="nearest",
    )

    denominator = gaussian_filter(
        eroded.astype(np.float32),
        sigma=sigma,
        mode="nearest",
    )

    wide_sigma = max(
        sigma * 2.0,
        1.0,
    )

    wide_numerator = gaussian_filter(
        P * eroded.astype(np.float32),
        sigma=wide_sigma,
        mode="nearest",
    )

    wide_denominator = gaussian_filter(
        eroded.astype(np.float32),
        sigma=wide_sigma,
        mode="nearest",
    )

    support_threshold = 0.10

    G = np.full_like(
        P,
        np.nan,
        dtype=np.float32,
    )

    strong = denominator >= support_threshold

    G[strong] = (
        numerator[strong]
        / denominator[strong]
    )

    weak = ~strong
    wide_valid = wide_denominator >= 0.02

    G[weak & wide_valid] = (
        wide_numerator[weak & wide_valid]
        / wide_denominator[weak & wide_valid]
    )

    # Where there is effectively no ground support, use the global
    # lower-envelope proxy as a final local safety fallback.
    unresolved = ~np.isfinite(G)

    if np.any(unresolved):
        lower = grey_opening(
            gaussian_filter(
                P,
                sigma=sigma,
                mode="nearest",
            ),
            size=5,
            mode="nearest",
        )
        G[unresolved] = lower[unresolved]

    support = np.clip(
        np.maximum(
            denominator,
            wide_denominator,
        ),
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    return G, support


def _estimate_ground_opening(
    *,
    P: np.ndarray,
    window_px: int,
    smooth_sigma_px: float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback estimator using true morphological opening.

    A flat structuring element is used so the lower surface can survive
    gentle slopes better than a one-sided minimum filter.
    """
    height, width = shape

    if window_px < 3:
        raise ValueError(
            "Opening window must be at least 3 pixels."
        )

    max_reasonable = (
        2 * min(height, width)
        - 1
    )

    if max_reasonable < 3:
        raise ValueError(
            "Input grid is too small for a safe opening kernel."
        )

    effective_window = min(
        int(window_px),
        int(max_reasonable),
    )

    if effective_window % 2 == 0:
        effective_window -= 1

    if effective_window < 3:
        raise ValueError(
            "Opening kernel became smaller than 3 pixels."
        )

    pre_smoothed = gaussian_filter(
        P,
        sigma=max(
            0.5,
            float(smooth_sigma_px),
        ),
        mode="nearest",
    )

    lower = grey_opening(
        pre_smoothed,
        size=effective_window,
        mode="nearest",
    )

    G = gaussian_filter(
        lower,
        sigma=max(
            0.5,
            float(smooth_sigma_px),
        ),
        mode="nearest",
    ).astype(
        np.float32,
        copy=False,
    )

    support = np.ones_like(
        G,
        dtype=np.float32,
    )

    return G, support


def _estimate_ground_min_pool(
    *,
    P: np.ndarray,
    window_px: int,
    smooth_sigma_px: float,
) -> np.ndarray:
    lower = minimum_filter(
        P,
        size=window_px,
        mode="nearest",
    )

    return gaussian_filter(
        lower,
        sigma=max(
            0.5,
            float(smooth_sigma_px),
        ),
        mode="nearest",
    ).astype(
        np.float32,
        copy=False,
    )


def _method_name(code: int) -> str:
    return {
        METHOD_SEMANTIC_MASK: "semantic-mask",
        METHOD_OPENING_FALLBACK: "opening-fallback",
        METHOD_MIN_POOL_LAST_RESORT: "min-pool-last-resort",
    }.get(
        int(code),
        "unknown",
    )


def _load_semantic_mask(
    path: Path,
    *,
    shape: tuple[int, int],
    grid: Mapping[str, Any],
) -> np.ndarray | None:
    """
    Load a ground-class mask from raw uint8 .dat or a georeferenced raster.

    Nonzero values are treated as ground support.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Semantic ground mask not found: {path}"
        )

    height, width = shape

    if path.suffix.lower() == ".dat":
        raw = np.fromfile(
            path,
            dtype=np.uint8,
        )
        if raw.size != height * width:
            raise ValueError(
                f"Semantic .dat mask has {raw.size} pixels; "
                f"expected {height * width}."
            )
        return raw.reshape(
            height,
            width,
        ) > 0

    transform_value = grid.get("transform")
    crs_value = grid.get("crs")

    if transform_value is None or crs_value is None:
        raise ValueError(
            "A raster semantic mask requires canonical transform and CRS."
        )

    transform = _affine_from_value(
        transform_value
    )

    destination = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError(
                f"Semantic mask raster has no bands: {path}"
            )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=crs_value,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    return destination > 0


def _fill_invalid(
    P: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    result = np.asarray(
        P,
        dtype=np.float32,
    ).copy()

    if np.all(valid):
        return result

    values = result[valid]
    if values.size == 0:
        raise ValueError(
            "Cannot fill prediction: no valid pixels."
        )

    fallback = float(
        np.median(values)
    )

    result[~valid] = fallback
    return result


def _estimate_ground_confidence(
    *,
    P: np.ndarray,
    G: np.ndarray,
    valid: np.ndarray,
    gsd_m: float,
    support: np.ndarray | None = None,
) -> np.ndarray:
    residual = np.abs(
        P - G
    )

    local_scale = max(
        1.0,
        5.0 / max(gsd_m, 1e-6),
    )

    smooth_residual = gaussian_filter(
        residual.astype(np.float32, copy=False),
        sigma=local_scale,
        mode="nearest",
    )

    confidence = np.exp(
        -smooth_residual / 5.0
    ).astype(
        np.float32,
        copy=False,
    )

    if support is not None:
        confidence *= np.clip(
            support.astype(np.float32, copy=False),
            0.0,
            1.0,
        )

    return np.clip(
        confidence,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )


def _load_grid(
    inference_meta: Mapping[str, Any],
) -> dict[str, Any]:
    grid = (
        inference_meta.get("canonical_grid")
        or inference_meta.get("grid")
    )

    if not isinstance(grid, Mapping):
        raise ValueError(
            "inference_meta.json does not contain canonical_grid/grid."
        )

    width = int(grid["width"])
    height = int(grid["height"])
    gsd_m = float(grid["gsd_m"])

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid grid dimensions: {width}x{height}"
        )

    if not math.isfinite(gsd_m) or gsd_m <= 0.0:
        raise ValueError(
            f"Invalid grid GSD: {gsd_m}"
        )

    return {
        "width": width,
        "height": height,
        "gsd_m": gsd_m,
        "crs": grid.get("crs"),
        "transform": grid.get("transform"),
        "bounds": grid.get("bounds"),
    }


def _validate_grid_consistency(
    *,
    inference_grid: Mapping[str, Any],
    tiling_meta: Mapping[str, Any],
) -> None:
    tiling_grid = tiling_meta.get(
        "canonical_grid"
    )

    if not isinstance(tiling_grid, Mapping):
        return

    if int(tiling_grid.get("width", -1)) != int(
        inference_grid["width"]
    ):
        raise ValueError(
            "Tiling/inference grid width mismatch."
        )

    if int(tiling_grid.get("height", -1)) != int(
        inference_grid["height"]
    ):
        raise ValueError(
            "Tiling/inference grid height mismatch."
        )

    tiling_gsd = float(
        tiling_grid.get(
            "gsd_m",
            inference_grid["gsd_m"],
        )
    )

    if not math.isclose(
        tiling_gsd,
        float(inference_grid["gsd_m"]),
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Tiling/inference GSD mismatch."
        )


def _metres_to_odd_pixels(
    metres: float,
    gsd_m: float,
) -> int:
    pixels = max(
        1,
        int(
            round(
                metres / gsd_m
            )
        ),
    )

    if pixels % 2 == 0:
        pixels += 1

    return pixels


def _validate_parameters(
    *,
    ground_window_m: float,
    smooth_sigma_m: float,
) -> None:
    if (
        not math.isfinite(ground_window_m)
        or ground_window_m <= 0.0
    ):
        raise ValueError(
            "ground_window_m must be > 0."
        )

    if (
        not math.isfinite(smooth_sigma_m)
        or smooth_sigma_m <= 0.0
    ):
        raise ValueError(
            "smooth_sigma_m must be > 0."
        )


def _affine_from_value(value: Any):
    if hasattr(value, "a"):
        return value

    if isinstance(value, (list, tuple)) and len(value) == 6:
        from rasterio.transform import Affine

        return Affine(
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
            float(value[4]),
            float(value[5]),
        )

    if isinstance(value, Mapping):
        from rasterio.transform import Affine

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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object in {path}."
        )

    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate ground and relative AGL from a "
            "scene-relative surface prediction."
        )
    )

    parser.add_argument(
        "--inference-dir",
        required=True,
    )

    parser.add_argument(
        "--tiling-meta",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--ground-window-m",
        type=float,
        default=DEFAULT_GROUND_WINDOW_M,
    )

    parser.add_argument(
        "--smooth-sigma-m",
        type=float,
        default=DEFAULT_SMOOTH_SIGMA_M,
    )

    parser.add_argument(
        "--semantic-mask",
        default=None,
        help=(
            "Optional ground-class mask. Nonzero pixels are used as "
            "ground/road/low-vegetation/water support."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser


def main() -> int:
    parser = build_parser()
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

    estimate_ground(
        inference_dir=args.inference_dir,
        tiling_meta_path=args.tiling_meta,
        output_dir=args.output_dir,
        ground_window_m=args.ground_window_m,
        smooth_sigma_m=args.smooth_sigma_m,
        semantic_mask_path=args.semantic_mask,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
