# src/dw/core/infer.py

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject


LOGGER = logging.getLogger("depthwizard.infer")


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODEL_CONFIG = {
    "encoder": "vitb",
    "features": 128,
    "out_channels": [96, 192, 384, 768],
}

# Depth Anything V2 model input resolution.
INPUT_SIZE = 518

IMAGE_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGE_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def _load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def _affine_from_value(
    value: Any,
) -> Affine:
    """
    Convert a JSON transform representation into rasterio Affine.
    """
    if isinstance(
        value,
        (list, tuple),
    ) and len(value) == 6:
        return Affine(
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
            float(value[4]),
            float(value[5]),
        )

    if isinstance(
        value,
        dict,
    ):
        return Affine(
            float(value["a"]),
            float(value["b"]),
            float(value["c"]),
            float(value["d"]),
            float(value["e"]),
            float(value["f"]),
        )

    raise ValueError(
        "Unable to parse raster transform from metadata."
    )


def _transform_to_list(
    transform: Affine | None,
) -> list[float] | None:
    if transform is None:
        return None

    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


# ---------------------------------------------------------------------
# Grid metadata helpers
# ---------------------------------------------------------------------

def _normalise_grid_dict(
    grid: Any,
) -> dict[str, Any] | None:
    """
    Extract a grid dictionary while tolerating a few older metadata forms.

    Preferred form:
        {
            "width": ...,
            "height": ...,
            "crs": ...,
            "transform": ...
        }
    """
    if not isinstance(
        grid,
        dict,
    ):
        return None

    width = grid.get(
        "width"
    )

    height = grid.get(
        "height"
    )

    if width is None or height is None:
        return None

    return grid


def _extract_grid_pair(
    tiling_meta: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """
    Extract:

        inference_grid
        canonical_grid

    from tiling metadata.

    The current tiling format is expected to have them as top-level
    sibling dictionaries.

    Older nested forms are also tolerated where practical.
    """

    # -------------------------------------------------------------
    # Current preferred format
    # -------------------------------------------------------------

    inference_grid = _normalise_grid_dict(
        tiling_meta.get(
            "inference_grid"
        )
    )

    canonical_grid = _normalise_grid_dict(
        tiling_meta.get(
            "canonical_grid"
        )
    )

    # -------------------------------------------------------------
    # Backward-compatible nested form
    # -------------------------------------------------------------

    if canonical_grid is not None:
        nested_inference = _normalise_grid_dict(
            canonical_grid.get(
                "inference_grid"
            )
        )

        nested_canonical = _normalise_grid_dict(
            canonical_grid.get(
                "canonical_grid"
            )
        )

        if inference_grid is None:
            inference_grid = nested_inference

        if nested_canonical is not None:
            canonical_grid = nested_canonical

    # -------------------------------------------------------------
    # Older "grid" form
    # -------------------------------------------------------------

    if canonical_grid is None:
        canonical_grid = _normalise_grid_dict(
            tiling_meta.get(
                "grid"
            )
        )

    # -------------------------------------------------------------
    # Flat fallback
    # -------------------------------------------------------------

    if canonical_grid is None:
        width = tiling_meta.get(
            "canonical_width"
        )

        height = tiling_meta.get(
            "canonical_height"
        )

        transform = tiling_meta.get(
            "canonical_transform"
        )

        crs = tiling_meta.get(
            "canonical_crs"
        )

        if width is not None and height is not None:
            canonical_grid = {
                "width": int(width),
                "height": int(height),
                "transform": transform,
                "crs": crs,
            }

    # -------------------------------------------------------------
    # Derive inference grid from canonical metadata if necessary
    # -------------------------------------------------------------

    if inference_grid is None and canonical_grid is not None:
        native_width = canonical_grid.get(
            "native_width"
        )

        native_height = canonical_grid.get(
            "native_height"
        )

        native_transform = canonical_grid.get(
            "native_transform"
        )

        native_crs = canonical_grid.get(
            "native_crs"
        )

        if (
            native_width is not None
            and native_height is not None
            and native_transform is not None
        ):
            inference_grid = {
                "width": int(native_width),
                "height": int(native_height),
                "transform": native_transform,
                "crs": native_crs,
            }

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    if inference_grid is None:
        raise ValueError(
            "tiling_meta.json does not contain inference_grid."
        )

    if canonical_grid is None:
        raise ValueError(
            "tiling_meta.json does not contain canonical_grid."
        )

    return (
        inference_grid,
        canonical_grid,
    )


def _grid_values(
    grid: dict[str, Any],
    *,
    name: str,
) -> tuple[
    int,
    int,
    Affine | None,
    str | None,
]:
    """
    Read width, height, transform and CRS from one grid.
    """

    width = grid.get(
        "width"
    )

    height = grid.get(
        "height"
    )

    if width is None or height is None:
        raise ValueError(
            f"{name} is missing width/height."
        )

    transform_value = grid.get(
        "transform"
    )

    transform = None

    if transform_value is not None:
        transform = _affine_from_value(
            transform_value
        )

    crs = grid.get(
        "crs"
    )

    if crs is not None:
        crs = str(crs)

    return (
        int(width),
        int(height),
        transform,
        crs,
    )


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------

def _load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: str,
) -> None:
    LOGGER.info(
        "Loading checkpoint: %s",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    if not isinstance(
        state_dict,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint does not contain a valid state dictionary."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):
            new_key = new_key[
                len("module.") :
            ]

        cleaned_state_dict[
            new_key
        ] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Checkpoint is missing model parameters. "
            f"First missing keys: {missing[:10]}"
        )

    if unexpected:
        LOGGER.warning(
            "Checkpoint contains unexpected parameters. "
            f"First unexpected keys: {unexpected[:10]}"
        )

    LOGGER.info(
        "Depth Anything V2 V4 checkpoint loaded."
    )


# ---------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------

def _create_model(
    external_model_dir: Path,
    checkpoint_path: Path,
    device: str,
) -> torch.nn.Module:
    """
    Create the same ViT-B Depth Anything V2 architecture used by V4.
    """

    if not external_model_dir.exists():
        raise FileNotFoundError(
            "Depth Anything V2 directory not found: "
            f"{external_model_dir}"
        )

    external_model_dir_str = str(
        external_model_dir
    )

    if external_model_dir_str not in sys.path:
        sys.path.insert(
            0,
            external_model_dir_str,
        )

    try:
        from depth_anything_v2.dpt import (
            DepthAnythingV2,
        )

    except ImportError as exc:

        raise ImportError(
            "Could not import Depth Anything V2 from: "
            f"{external_model_dir}"
        ) from exc

    LOGGER.info(
        "Creating Depth Anything V2 ViT-B..."
    )

    model = DepthAnythingV2(
        encoder=MODEL_CONFIG["encoder"],
        features=MODEL_CONFIG["features"],
        out_channels=MODEL_CONFIG["out_channels"],
    )

    model.to(
        device
    )

    model.eval()

    _load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return model


# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

def _normalize_image(
    image_chw: np.ndarray,
) -> torch.Tensor:
    """
    Convert CHW float32 [0,1] image to normalized BCHW tensor.
    """

    image = np.asarray(
        image_chw,
        dtype=np.float32,
    )

    if image.ndim != 3:
        raise ValueError(
            f"Expected CHW image, got shape={image.shape}"
        )

    if image.shape[0] < 3:
        raise ValueError(
            "Inference requires at least 3 image channels."
        )

    image = image[
        :3
    ]

    mean = IMAGE_MEAN.reshape(
        3,
        1,
        1,
    )

    std = IMAGE_STD.reshape(
        3,
        1,
        1,
    )

    normalized = (
        image - mean
    ) / std

    return torch.from_numpy(
        normalized
    ).unsqueeze(
        0
    )


def _prepare_model_tensor(
    image_chw: np.ndarray,
) -> torch.Tensor:
    """
    Prepare one native-resolution tile for the network.

    Native tile:
        typically 1024 x 1024

    Network input:
        518 x 518
    """

    tensor = _normalize_image(
        image_chw
    )

    if tensor.shape[-2:] != (
        INPUT_SIZE,
        INPUT_SIZE,
    ):
        tensor = F.interpolate(
            tensor,
            size=(
                INPUT_SIZE,
                INPUT_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

    return tensor


# ---------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------

@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    image_chw: np.ndarray,
    device: str,
    fp16: bool,
) -> np.ndarray:
    """
    Run one native tile through Depth Anything V2.

    Input:
        CHW float32 [0,1]

    Network:
        518 x 518

    Output:
        Native tile H x W relative surface prediction.
    """

    original_height = int(
        image_chw.shape[1]
    )

    original_width = int(
        image_chw.shape[2]
    )

    tensor = _prepare_model_tensor(
        image_chw
    ).to(
        device,
        non_blocking=True,
    )

    use_amp = (
        fp16
        and device.startswith(
            "cuda"
        )
        and torch.cuda.is_available()
    )

    if use_amp:

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            prediction = model(
                tensor
            )

    else:

        prediction = model(
            tensor
        )

    if not torch.is_tensor(
        prediction
    ):
        raise RuntimeError(
            "Depth Anything V2 returned a non-tensor output."
        )

    if prediction.ndim == 4:

        prediction = prediction[
            0,
            0,
        ]

    elif prediction.ndim == 3:

        prediction = prediction[
            0
        ]

    elif prediction.ndim != 2:

        raise RuntimeError(
            "Unexpected model output shape: "
            f"{tuple(prediction.shape)}"
        )

    # -------------------------------------------------------------
    # Resize model prediction back to native tile geometry.
    # -------------------------------------------------------------

    if prediction.shape != (
        original_height,
        original_width,
    ):

        prediction = F.interpolate(
            prediction.unsqueeze(
                0
            ).unsqueeze(
                0
            ),
            size=(
                original_height,
                original_width,
            ),
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            0,
            1,
        )

    return prediction.float().cpu().numpy()


# ---------------------------------------------------------------------
# Test-time augmentation
# ---------------------------------------------------------------------

def _predict_with_tta(
    model: torch.nn.Module,
    image_chw: np.ndarray,
    device: str,
    use_flip_tta: bool,
    fp16: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Run normal prediction and optional horizontal-flip TTA.

    Returns:
        prediction,
        tta_sigma
    """

    prediction = _predict(
        model=model,
        image_chw=image_chw,
        device=device,
        fp16=fp16,
    )

    if not use_flip_tta:

        sigma = np.zeros_like(
            prediction,
            dtype=np.float32,
        )

        return (
            prediction,
            sigma,
        )

    flipped = np.flip(
        image_chw,
        axis=2,
    ).copy()

    prediction_flipped = _predict(
        model=model,
        image_chw=flipped,
        device=device,
        fp16=fp16,
    )

    prediction_flipped = np.flip(
        prediction_flipped,
        axis=1,
    ).copy()

    blended = (
        prediction
        + prediction_flipped
    ) / 2.0

    sigma = (
        np.abs(
            prediction
            - prediction_flipped
        )
        * 0.5
    ).astype(
        np.float32
    )

    return (
        blended.astype(
            np.float32
        ),
        sigma,
    )


# ---------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------

def _hann_squared(
    height: int,
    width: int,
) -> np.ndarray:
    """
    Build a 2D squared Hann overlap-blending window.

    A small positive floor avoids zero weight at exact tile edges.
    """

    if height <= 1:

        wy = np.ones(
            height,
            dtype=np.float32,
        )

    else:

        wy = np.hanning(
            height
        ).astype(
            np.float32
        )

    if width <= 1:

        wx = np.ones(
            width,
            dtype=np.float32,
        )

    else:

        wx = np.hanning(
            width
        ).astype(
            np.float32
        )

    window = (
        wy[:, None]
        * wx[None, :]
    )

    window = (
        window
        ** 2
    )

    window = np.maximum(
        window,
        1e-6,
    )

    return window.astype(
        np.float32
    )


# ---------------------------------------------------------------------
# Tile metadata helpers
# ---------------------------------------------------------------------

def _extract_tile_list(
    tiling_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    tiles = tiling_meta.get(
        "tiles"
    )

    if isinstance(
        tiles,
        list,
    ):
        return tiles

    raise KeyError(
        "No 'tiles' list found in tiling_meta.json."
    )


def _extract_tile_position(
    tile: dict[str, Any],
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """
    Extract tile placement information.

    Returns:

        y,
        x,
        valid_height,
        valid_width,
        tile_height,
        tile_width
    """

    y = tile.get(
        "y",
        tile.get(
            "row",
            tile.get(
                "top"
            ),
        ),
    )

    x = tile.get(
        "x",
        tile.get(
            "col",
            tile.get(
                "left"
            ),
        ),
    )

    if y is None or x is None:
        raise KeyError(
            "Could not determine tile position from metadata: "
            f"{tile}"
        )

    valid_height = int(
        tile.get(
            "valid_height",
            tile.get(
                "height",
                tile.get(
                    "h",
                    1024,
                ),
            ),
        )
    )

    valid_width = int(
        tile.get(
            "valid_width",
            tile.get(
                "width",
                tile.get(
                    "w",
                    1024,
                ),
            ),
        )
    )

    tile_height = int(
        tile.get(
            "tile_height",
            1024,
        )
    )

    tile_width = int(
        tile.get(
            "tile_width",
            1024,
        )
    )

    return (
        int(y),
        int(x),
        valid_height,
        valid_width,
        tile_height,
        tile_width,
    )


# ---------------------------------------------------------------------
# Array output helpers
# ---------------------------------------------------------------------

def _write_raw_array(
    path: Path,
    array: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.asarray(
        array,
        dtype=np.float32,
    ).tofile(
        path
    )


def _write_tif(
    *,
    path: Path,
    array: np.ndarray,
    transform: Affine | None,
    crs: str | None,
    description: str,
    product: str,
) -> None:
    """
    Write a Float32 GeoTIFF.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = np.asarray(
        array,
        dtype=np.float32,
    ).copy()

    invalid = ~np.isfinite(
        data
    )

    data[invalid] = np.nan

    height, width = data.shape

    if transform is None:
        transform = Affine.identity()

    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    if crs:
        profile[
            "crs"
        ] = crs

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            data,
            1,
        )

        dst.set_band_description(
            1,
            description,
        )

        dst.update_tags(
            PRODUCT=product,
            UNITS="relative",
            SOURCE="Depth Anything V2",
            METRIC="false",
            GEOREFERENCED=(
                "true"
                if crs
                else "false"
            ),
        )


# ---------------------------------------------------------------------
# Native -> canonical resampling
# ---------------------------------------------------------------------

def _resample_to_canonical(
    *,
    native_array: np.ndarray,
    native_transform: Affine | None,
    native_crs: str | None,
    canonical_width: int,
    canonical_height: int,
    canonical_transform: Affine | None,
    canonical_crs: str | None,
    resampling: Resampling,
) -> np.ndarray:
    """
    Reproject/resample a native stitched array onto the canonical grid.
    """

    destination = np.full(
        (
            canonical_height,
            canonical_width,
        ),
        np.nan,
        dtype=np.float32,
    )

    if native_transform is None:
        raise ValueError(
            "Native grid does not contain a valid transform."
        )

    if canonical_transform is None:
        raise ValueError(
            "Canonical grid does not contain a valid transform."
        )

    # Non-georeferenced scenes have no CRS to drive a map reprojection.
    # Their canonical grid is intentionally the same pixel grid as the
    # native inference grid, so preserve the stitched array directly.
    if native_crs is None or canonical_crs is None:
        if (
            native_array.shape
            == (
                canonical_height,
                canonical_width,
            )
        ):
            return np.asarray(
                native_array,
                dtype=np.float32,
            ).copy()

        raise ValueError(
            "Cannot resample a non-georeferenced prediction to a "
            "different-sized canonical grid because no CRS is available."
        )

    source = np.asarray(
        native_array,
        dtype=np.float32,
    )

    reproject(
        source=source,
        destination=destination,
        src_transform=native_transform,
        src_crs=native_crs,
        dst_transform=canonical_transform,
        dst_crs=canonical_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling,
    )

    return destination


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def _finite_stats(
    array: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:
    valid = np.asarray(
        array,
        dtype=np.float32,
    )

    mask = np.isfinite(
        valid
    )

    if not np.any(
        mask
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    values = valid[
        mask
    ]

    return (
        float(
            values.min()
        ),
        float(
            np.median(
                values
            )
        ),
        float(
            values.max()
        ),
    )


# ---------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------

def run_inference(
    tiles_dir: str | Path,
    tiling_meta_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    external_model_dir: str | Path = "external/Depth-Anything-V2",
    device: str = "cuda",
    use_flip_tta: bool = True,
    batch_size: int = 1,
    fp16: bool = True,
) -> Path:
    """
    Run native-resolution overlapping inference and then resample the
    stitched result onto the canonical processing grid.

    Pipeline:

        native RGB
            ↓
        native 1024x1024 overlapping tiles
            ↓
        resize each tile to 518x518
            ↓
        Depth Anything V2
            ↓
        resize prediction back to native tile size
            ↓
        native overlap blending
            ↓
        native full-scene P
            ↓
        resample to canonical 1 m grid
            ↓
        canonical P for downstream stages

    Outputs:

        P_native.dat
        P.dat

        relative_dsm_native.tif
        relative_dsm.tif

        sigma_tta_native.dat
        sigma_tta.dat

        weight_native.dat
        weight.dat

        inference_meta.json
    """

    tiles_dir = Path(
        tiles_dir
    ).expanduser().resolve()

    tiling_meta_path = Path(
        tiling_meta_path
    ).expanduser().resolve()

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    checkpoint_path = Path(
        checkpoint_path
    ).expanduser().resolve()

    external_model_dir = Path(
        external_model_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    if not tiles_dir.exists():
        raise FileNotFoundError(
            f"Tiles directory not found: {tiles_dir}"
        )

    if not tiling_meta_path.exists():
        raise FileNotFoundError(
            f"Tiling metadata not found: {tiling_meta_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1."
        )

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch reports "
                "CUDA is not available."
            )

    LOGGER.info(
        "Inference device: %s",
        device,
    )

    # -------------------------------------------------------------
    # Load tiling metadata
    # -------------------------------------------------------------

    tiling_meta = _load_json(
        tiling_meta_path
    )

    (
        inference_grid_meta,
        canonical_grid_meta,
    ) = _extract_grid_pair(
        tiling_meta
    )

    (
        native_width,
        native_height,
        native_transform,
        native_crs,
    ) = _grid_values(
        inference_grid_meta,
        name="inference_grid",
    )

    (
        canonical_width,
        canonical_height,
        canonical_transform,
        canonical_crs,
    ) = _grid_values(
        canonical_grid_meta,
        name="canonical_grid",
    )

    LOGGER.info(
        "Native inference grid: %dx%d",
        native_width,
        native_height,
    )

    LOGGER.info(
        "Canonical processing grid: %dx%d",
        canonical_width,
        canonical_height,
    )

    native_gsd_x = None
    native_gsd_y = None

    if native_transform is not None:
        native_gsd_x = abs(
            float(
                native_transform.a
            )
        )

        native_gsd_y = abs(
            float(
                native_transform.e
            )
        )

    canonical_gsd_x = None
    canonical_gsd_y = None

    if canonical_transform is not None:
        canonical_gsd_x = abs(
            float(
                canonical_transform.a
            )
        )

        canonical_gsd_y = abs(
            float(
                canonical_transform.e
            )
        )

    LOGGER.info(
        "Native GSD: %.6f x %.6f",
        (
            native_gsd_x
            if native_gsd_x is not None
            else float("nan")
        ),
        (
            native_gsd_y
            if native_gsd_y is not None
            else float("nan")
        ),
    )

    LOGGER.info(
        "Canonical GSD: %.6f x %.6f",
        (
            canonical_gsd_x
            if canonical_gsd_x is not None
            else float("nan")
        ),
        (
            canonical_gsd_y
            if canonical_gsd_y is not None
            else float("nan")
        ),
    )

    # -------------------------------------------------------------
    # Tiles
    # -------------------------------------------------------------

    tiles = _extract_tile_list(
        tiling_meta
    )

    total_tiles = len(
        tiles
    )

    if total_tiles == 0:
        raise ValueError(
            "Tiling metadata contains zero tiles."
        )

    tile_files = sorted(
        tiles_dir.glob(
            "*.npy"
        )
    )

    if not tile_files:
        raise FileNotFoundError(
            f"No .npy tiles found in {tiles_dir}"
        )

    if len(tile_files) != total_tiles:
        LOGGER.warning(
            "Tiling metadata contains %d tiles, "
            "but %d .npy files were found.",
            total_tiles,
            len(tile_files),
        )

    LOGGER.info(
        "Native tile size expected: 1024x1024"
    )

    LOGGER.info(
        "Network input size: %dx%d",
        INPUT_SIZE,
        INPUT_SIZE,
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = _create_model(
        external_model_dir=external_model_dir,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # -------------------------------------------------------------
    # Native accumulators
    # -------------------------------------------------------------

    prediction_sum_native = np.zeros(
        (
            native_height,
            native_width,
        ),
        dtype=np.float32,
    )

    prediction_weight_native = np.zeros(
        (
            native_height,
            native_width,
        ),
        dtype=np.float32,
    )

    sigma_sum_native = np.zeros(
        (
            native_height,
            native_width,
        ),
        dtype=np.float32,
    )

    processed = 0

    # -------------------------------------------------------------
    # Native tile inference
    # -------------------------------------------------------------

    for tile_index, tile_meta in enumerate(
        tiles
    ):

        tile_filename = tile_meta.get(
            "filename"
        )

        if tile_filename:

            tile_path = (
                tiles_dir
                / str(tile_filename)
            )

        else:

            tile_path = (
                tiles_dir
                / f"tile_{tile_index:06d}.npy"
            )

        if not tile_path.exists():

            if tile_index < len(
                tile_files
            ):

                tile_path = tile_files[
                    tile_index
                ]

            else:

                raise FileNotFoundError(
                    "Tile file not found for "
                    f"tile index {tile_index}: "
                    f"{tile_path}"
                )

        image = np.load(
            tile_path
        )

        if image.ndim != 3:
            raise ValueError(
                f"Tile {tile_path} must be CHW. "
                f"Got shape={image.shape}"
            )

        if image.shape[0] < 3:
            raise ValueError(
                f"Tile {tile_path} does not contain "
                "three RGB channels."
            )

        (
            y,
            x,
            valid_height,
            valid_width,
            tile_height,
            tile_width,
        ) = _extract_tile_position(
            tile_meta
        )

        actual_height = int(
            image.shape[1]
        )

        actual_width = int(
            image.shape[2]
        )

        tile_height = min(
            tile_height,
            actual_height,
        )

        tile_width = min(
            tile_width,
            actual_width,
        )

        LOGGER.debug(
            "Tile %d: x=%d y=%d "
            "valid=%dx%d "
            "actual=%dx%d",
            tile_index,
            x,
            y,
            valid_width,
            valid_height,
            actual_width,
            actual_height,
        )

        # ---------------------------------------------------------
        # Predict tile
        # ---------------------------------------------------------

        (
            prediction,
            sigma,
        ) = _predict_with_tta(
            model=model,
            image_chw=image,
            device=device,
            use_flip_tta=use_flip_tta,
            fp16=fp16,
        )

        # ---------------------------------------------------------
        # Clip valid region against native scene
        # ---------------------------------------------------------

        valid_height = min(
            valid_height,
            prediction.shape[0],
            native_height - y,
        )

        valid_width = min(
            valid_width,
            prediction.shape[1],
            native_width - x,
        )

        if (
            valid_height <= 0
            or valid_width <= 0
        ):
            LOGGER.warning(
                "Skipping tile %d because its valid region "
                "falls outside the native scene.",
                tile_index,
            )
            continue

        prediction_crop = prediction[
            :valid_height,
            :valid_width,
        ]

        sigma_crop = sigma[
            :valid_height,
            :valid_width,
        ]

        # ---------------------------------------------------------
        # Blend
        # ---------------------------------------------------------

        window = _hann_squared(
            valid_height,
            valid_width,
        )

        y0 = y
        y1 = (
            y
            + valid_height
        )

        x0 = x
        x1 = (
            x
            + valid_width
        )

        prediction_sum_native[
            y0:y1,
            x0:x1,
        ] += (
            prediction_crop
            * window
        )

        prediction_weight_native[
            y0:y1,
            x0:x1,
        ] += window

        sigma_sum_native[
            y0:y1,
            x0:x1,
        ] += (
            sigma_crop
            * window
        )

        processed += 1

        LOGGER.info(
            "Inference: %d/%d tiles",
            processed,
            total_tiles,
        )

    # -------------------------------------------------------------
    # Final native blend
    # -------------------------------------------------------------

    valid_native = (
        prediction_weight_native
        > 0
    )

    P_native = np.full(
        (
            native_height,
            native_width,
        ),
        np.nan,
        dtype=np.float32,
    )

    P_native[
        valid_native
    ] = (
        prediction_sum_native[
            valid_native
        ]
        /
        prediction_weight_native[
            valid_native
        ]
    )

    sigma_tta_native = np.full(
        (
            native_height,
            native_width,
        ),
        np.nan,
        dtype=np.float32,
    )

    sigma_tta_native[
        valid_native
    ] = (
        sigma_sum_native[
            valid_native
        ]
        /
        prediction_weight_native[
            valid_native
        ]
    )

    coverage_fraction = (
        float(
            np.count_nonzero(
                valid_native
            )
        )
        /
        float(
            native_width
            * native_height
        )
    )

    LOGGER.info(
        "Native prediction coverage: %.4f%%",
        coverage_fraction * 100.0,
    )

    # -------------------------------------------------------------
    # Native -> canonical resampling
    # -------------------------------------------------------------

    LOGGER.info(
        "Resampling stitched native prediction "
        "from %dx%d to %dx%d canonical grid.",
        native_width,
        native_height,
        canonical_width,
        canonical_height,
    )

    P = _resample_to_canonical(
        native_array=P_native,
        native_transform=native_transform,
        native_crs=native_crs,
        canonical_width=canonical_width,
        canonical_height=canonical_height,
        canonical_transform=canonical_transform,
        canonical_crs=canonical_crs,
        resampling=Resampling.bilinear,
    )

    sigma_tta = _resample_to_canonical(
        native_array=sigma_tta_native,
        native_transform=native_transform,
        native_crs=native_crs,
        canonical_width=canonical_width,
        canonical_height=canonical_height,
        canonical_transform=canonical_transform,
        canonical_crs=canonical_crs,
        resampling=Resampling.bilinear,
    )

    weight = _resample_to_canonical(
        native_array=prediction_weight_native,
        native_transform=native_transform,
        native_crs=native_crs,
        canonical_width=canonical_width,
        canonical_height=canonical_height,
        canonical_transform=canonical_transform,
        canonical_crs=canonical_crs,
        resampling=Resampling.bilinear,
    )

    # -------------------------------------------------------------
    # Log statistics
    # -------------------------------------------------------------

    (
        native_min,
        native_median,
        native_max,
    ) = _finite_stats(
        P_native
    )

    (
        canonical_min,
        canonical_median,
        canonical_max,
    ) = _finite_stats(
        P
    )

    LOGGER.info(
        "Native P range: %.6f to %.6f",
        native_min,
        native_max,
    )

    LOGGER.info(
        "Native P median: %.6f",
        native_median,
    )

    LOGGER.info(
        "Canonical P range: %.6f to %.6f",
        canonical_min,
        canonical_max,
    )

    LOGGER.info(
        "Canonical P median: %.6f",
        canonical_median,
    )

    # -------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------

    P_native_path = (
        output_dir
        / "P_native.dat"
    )

    P_path = (
        output_dir
        / "P.dat"
    )

    relative_dsm_native_dat = (
        output_dir
        / "relative_dsm_native.dat"
    )

    relative_dsm_dat = (
        output_dir
        / "relative_dsm.dat"
    )

    sigma_native_path = (
        output_dir
        / "sigma_tta_native.dat"
    )

    sigma_path = (
        output_dir
        / "sigma_tta.dat"
    )

    weight_native_path = (
        output_dir
        / "weight_native.dat"
    )

    weight_path = (
        output_dir
        / "weight.dat"
    )

    # -------------------------------------------------------------
    # Raw outputs
    # -------------------------------------------------------------

    _write_raw_array(
        P_native_path,
        P_native,
    )

    _write_raw_array(
        P_path,
        P,
    )

    _write_raw_array(
        relative_dsm_native_dat,
        P_native,
    )

    _write_raw_array(
        relative_dsm_dat,
        P,
    )

    _write_raw_array(
        sigma_native_path,
        sigma_tta_native,
    )

    _write_raw_array(
        sigma_path,
        sigma_tta,
    )

    _write_raw_array(
        weight_native_path,
        prediction_weight_native,
    )

    _write_raw_array(
        weight_path,
        weight,
    )

    # -------------------------------------------------------------
    # Relative DSM GeoTIFFs
    # -------------------------------------------------------------

    native_is_georeferenced = bool(
        native_crs
        and native_transform is not None
    )

    canonical_is_georeferenced = bool(
        canonical_crs
        and canonical_transform is not None
    )

    relative_dsm_native_tif = (
        output_dir
        / "relative_dsm_native.tif"
    )

    relative_dsm_tif = (
        output_dir
        / "relative_dsm.tif"
    )

    _write_tif(
        path=relative_dsm_native_tif,
        array=P_native,
        transform=native_transform,
        crs=(
            native_crs
            if native_is_georeferenced
            else None
        ),
        description=(
            "Native-resolution relative DSM / relative surface"
        ),
        product="RELATIVE_DSM_NATIVE",
    )

    _write_tif(
        path=relative_dsm_tif,
        array=P,
        transform=canonical_transform,
        crs=(
            canonical_crs
            if canonical_is_georeferenced
            else None
        ),
        description=(
            "Canonical relative DSM / relative surface"
        ),
        product="RELATIVE_DSM",
    )

    # -------------------------------------------------------------
    # Metadata
    #
    # IMPORTANT:
    # inference_grid and canonical_grid are TOP-LEVEL siblings.
    # canonical_grid directly contains width/height/crs/transform.
    # This is required by ground.py.
    # -------------------------------------------------------------

    native_bounds = None

    if (
        native_transform is not None
    ):
        native_bounds = [
            float(value)
            for value in array_bounds(
                native_height,
                native_width,
                native_transform,
            )
        ]

    canonical_bounds = None

    if (
        canonical_transform is not None
    ):
        canonical_bounds = [
            float(value)
            for value in array_bounds(
                canonical_height,
                canonical_width,
                canonical_transform,
            )
        ]

    inference_meta = {

        "stage": "inference",

        "model": {
            "name": "Depth Anything V2",
            "encoder": MODEL_CONFIG[
                "encoder"
            ],
            "features": MODEL_CONFIG[
                "features"
            ],
            "out_channels": MODEL_CONFIG[
                "out_channels"
            ],
            "input_size": INPUT_SIZE,
        },

        "checkpoint": str(
            checkpoint_path
        ),

        "device": device,

        "settings": {

            "flip_tta": bool(
                use_flip_tta
            ),

            "batch_size": int(
                batch_size
            ),

            "fp16": bool(
                fp16
            ),

            "tile_geometry": {
                "native_tile_size": (
                    int(
                        tiles[0].get(
                            "tile_width",
                            1024,
                        )
                    )
                    if tiles
                    else 1024
                ),

                "overlap": float(
                    tiling_meta.get(
                        "overlap",
                        0.5,
                    )
                ),

                "stride": int(
                    tiling_meta.get(
                        "stride",
                        512,
                    )
                ),
            },
        },

        # ---------------------------------------------------------
        # Native inference grid
        # ---------------------------------------------------------

        "inference_grid": {

            "width": int(
                native_width
            ),

            "height": int(
                native_height
            ),

            "crs": native_crs,

            "transform": _transform_to_list(
                native_transform
            ),

            "gsd_m": (
                float(
                    native_gsd_x
                )
                if native_gsd_x is not None
                else None
            ),

            "bounds": native_bounds,

            "purpose": (
                "Native source grid used for RGB tile extraction, "
                "Depth Anything V2 inference and native stitching."
            ),
        },

        # ---------------------------------------------------------
        # Canonical processing grid
        # ---------------------------------------------------------

        "canonical_grid": {

            "width": int(
                canonical_width
            ),

            "height": int(
                canonical_height
            ),

            "crs": canonical_crs,

            "transform": _transform_to_list(
                canonical_transform
            ),

            "gsd_m": (
                float(
                    canonical_gsd_x
                )
                if canonical_gsd_x is not None
                else None
            ),

            "bounds": canonical_bounds,

            "purpose": (
                "Canonical processing grid used by downstream "
                "FABDEM, ground, calibration and DSM stages."
            ),
        },

        "grid_roles": {

            "inference_grid": (
                "native"
            ),

            "canonical_grid": (
                "downstream_processing"
            ),
        },

        "tiles": {

            "total": int(
                total_tiles
            ),

            "processed": int(
                processed
            ),

            "coverage_fraction_native": float(
                coverage_fraction
            ),
        },

        "products": {

            "P_native": {

                "path": str(
                    P_native_path
                ),

                "description": (
                    "Native-resolution stitched relative "
                    "surface prediction."
                ),

                "units": "relative",

                "metric": False,

                "georeferenced": bool(
                    native_is_georeferenced
                ),
            },

            "P": {

                "path": str(
                    P_path
                ),

                "description": (
                    "Canonical-grid relative surface prediction "
                    "used by downstream stages."
                ),

                "units": "relative",

                "metric": False,

                "georeferenced": bool(
                    canonical_is_georeferenced
                ),
            },

            "relative_dsm_native": {

                "path": str(
                    relative_dsm_native_tif
                ),

                "dat": str(
                    relative_dsm_native_dat
                ),

                "description": (
                    "Native-resolution relative DSM represented "
                    "directly by P_native."
                ),

                "units": "relative",

                "metric": False,

                "georeferenced": bool(
                    native_is_georeferenced
                ),
            },

            "relative_dsm": {

                "path": str(
                    relative_dsm_tif
                ),

                "dat": str(
                    relative_dsm_dat
                ),

                "description": (
                    "Canonical relative DSM represented "
                    "directly by P."
                ),

                "units": "relative",

                "metric": False,

                "georeferenced": bool(
                    canonical_is_georeferenced
                ),
            },

            "sigma_tta_native": {

                "path": str(
                    sigma_native_path
                ),
            },

            "sigma_tta": {

                "path": str(
                    sigma_path
                ),
            },

            "weight_native": {

                "path": str(
                    weight_native_path
                ),
            },

            "weight": {

                "path": str(
                    weight_path
                ),
            },
        },

        "notes": [

            (
                "RGB inference is performed on the native-resolution "
                "source grid using overlapping tiles."
            ),

            (
                "Each native tile is internally resized to 518x518 "
                "for Depth Anything V2."
            ),

            (
                "Predictions are resized back to native tile geometry "
                "before overlap blending."
            ),

            (
                "The stitched native prediction is resampled to the "
                "canonical processing grid before FABDEM, ground, "
                "calibration and DSM composition."
            ),

            (
                "P is an uncalibrated relative surface prediction."
            ),

            (
                "No FABDEM or metric calibration is applied in inference."
            ),

            (
                "Metric DSM composition is performed later for "
                "georeferenced scenes."
            ),
        ],
    }

    inference_meta_path = (
        output_dir
        / "inference_meta.json"
    )

    with inference_meta_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            inference_meta,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final logging
    # -------------------------------------------------------------

    LOGGER.info(
        "Inference complete."
    )

    LOGGER.info(
        "Native P: %s",
        P_native_path,
    )

    LOGGER.info(
        "Canonical P: %s",
        P_path,
    )

    LOGGER.info(
        "Native relative DSM: %s",
        relative_dsm_native_tif,
    )

    LOGGER.info(
        "Canonical relative DSM: %s",
        relative_dsm_tif,
    )

    LOGGER.info(
        "Canonical relative depth map: %s",
        P_path,
    )

    LOGGER.info(
        "Native relative depth map: %s",
        P_native_path,
    )

    return P_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Depth Anything V2 native-resolution inference "
            "for the DepthWizard prototype."
        )
    )

    parser.add_argument(
        "--tiles-dir",
        required=True,
        help=(
            "Directory containing prepared native-resolution "
            ".npy tiles."
        ),
    )

    parser.add_argument(
        "--tiling-meta",
        required=True,
        help=(
            "Path to tiling_meta.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Inference output directory."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to trained V4 checkpoint."
        ),
    )

    parser.add_argument(
        "--external-model-dir",
        default=(
            "external/Depth-Anything-V2"
        ),
        help=(
            "Path to copied Depth Anything V2 repository."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=[
            "cuda",
            "cpu",
        ],
        help=(
            "Inference device."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Inference batch size. "
            "The prototype currently processes tiles serially."
        ),
    )

    parser.add_argument(
        "--no-flip-tta",
        action="store_true",
        help=(
            "Disable horizontal flip test-time augmentation."
        ),
    )

    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help=(
            "Disable FP16 inference on CUDA."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    run_inference(
        tiles_dir=args.tiles_dir,
        tiling_meta_path=args.tiling_meta,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        external_model_dir=args.external_model_dir,
        device=args.device,
        use_flip_tta=not args.no_flip_tta,
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )