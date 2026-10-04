"""
DepthWizard / GeoSculpt prototype tiling.

Pipeline role
-------------
    Original RGB image
        |
        +--> native inference grid
        |       |
        |       +--> overlapping 1024x1024 RGB tiles
        |               |
        |               +--> Depth Anything V2
        |
        +--> canonical 1 m processing grid
                |
                +--> FABDEM / ground / metric DSM

Important design
----------------
The neural network must see the original-resolution RGB imagery.

For georeferenced scenes we therefore maintain TWO grids:

1. inference_grid
   - original source pixel grid
   - original width/height
   - original transform
   - original CRS
   - used for RGB tile extraction and model inference

2. canonical_grid
   - projected metric grid
   - normally 1 m/pixel for this prototype
   - used later for FABDEM alignment, ground estimation,
     calibration and final DSM composition

The stitched model prediction will later be resampled from
the native inference grid to the canonical processing grid.

The model expects RGB values in [0, 1]. ImageNet normalization is
applied later in infer.py immediately before the neural network.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

# IMPORTANT:
# Native RGB tiles are now 1024x1024.
#
# The Depth Anything implementation may internally resize the tile to
# its network input size (518), but the source crop remains 1024x1024.
DEFAULT_TILE_SIZE = 1024

DEFAULT_OVERLAP = 0.50

# This remains the canonical processing GSD for downstream FABDEM,
# ground estimation and DSM composition.
DEFAULT_TRAIN_GSD_M = 1.0

STRETCH_LOW = 2.0
STRETCH_HIGH = 98.0

SUPPORTED_KINDS = {
    "GEO_PROJECTED",
    "GEO_GEOGRAPHIC",
    "GEO_WEBMERC",
}


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TileWindow:
    """
    One tile's location in the native inference grid.

    x/y refer to the upper-left pixel of the tile in the native
    source pixel grid.

    valid_* describe the non-padded portion of the tile.
    """

    index: int

    x: int
    y: int

    width: int
    height: int

    valid_x: int
    valid_y: int
    valid_width: int
    valid_height: int


@dataclass
class CanonicalGrid:
    """
    Downstream metric processing grid.

    For georeferenced scenes this is a projected metric grid,
    normally at 1 m/pixel for the prototype.
    """

    width: int
    height: int

    crs: str
    transform: List[float]

    gsd_m: float

    bounds: List[float]

    native_width: int
    native_height: int

    native_crs: Optional[str]
    native_transform: List[float]

    native_gsd_m: Optional[float]

    canonical: bool


@dataclass
class InferenceGrid:
    """
    Native source grid used by the neural-network inference stage.
    """

    width: int
    height: int

    crs: Optional[str]
    transform: List[float]

    gsd_m: Optional[float]

    bounds: List[float]

    native: bool


@dataclass
class TilingResult:
    output_dir: Path

    inference_grid: InferenceGrid
    canonical_grid: CanonicalGrid

    tile_size: int
    stride: int
    tile_count: int

    stretch_low: List[float]
    stretch_high: List[float]

    metadata_path: Path


# ---------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------


def prepare_tiles(
    *,
    input_path: str | Path,
    scene_meta_path: str | Path,
    output_dir: str | Path,
    train_gsd_m: float = DEFAULT_TRAIN_GSD_M,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP,
) -> TilingResult:
    """
    Prepare native-resolution overlapping RGB tiles.

    Parameters
    ----------
    input_path:
        Original RGB image.

    scene_meta_path:
        JSON produced by core/meta.py.

    output_dir:
        Directory in which tiles and metadata are written.

    train_gsd_m:
        Canonical downstream processing GSD in metres/pixel.
        The name is retained for compatibility with the current
        run.py interface.

        IMPORTANT:
        This value does NOT resample the RGB image before inference.

    tile_size:
        Native RGB tile size.

        GeoSculpt deployment uses 1024x1024 source crops.

    overlap:
        Fractional overlap between neighboring tiles.

    Returns
    -------
    TilingResult
    """

    input_path = Path(
        input_path
    ).expanduser().resolve()

    scene_meta_path = Path(
        scene_meta_path
    ).expanduser().resolve()

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    _validate_arguments(
        train_gsd_m=train_gsd_m,
        tile_size=tile_size,
        overlap=overlap,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    if not scene_meta_path.exists():
        raise FileNotFoundError(
            scene_meta_path
        )

    scene_meta = _read_json(
        scene_meta_path
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tiles_dir = (
        output_dir
        / "tiles"
    )

    tiles_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Open source image
    # ---------------------------------------------------------------

    with rasterio.open(
        input_path
    ) as src:

        band_map = _resolve_band_map(
            scene_meta,
            src.count,
        )

        bands = [
            band_map["red"],
            band_map["green"],
            band_map["blue"],
        ]

        native_width = int(
            src.width
        )

        native_height = int(
            src.height
        )

        native_crs = _resolve_native_crs(
            scene_meta,
            src.crs,
        )

        native_transform = _resolve_native_transform(
            scene_meta,
            src.transform,
        )

        native_gsd_m = _optional_float(
            scene_meta.get(
                "gsd_true_m"
            )
        )

        scene_kind = str(
            scene_meta.get(
                "kind",
                "NONE",
            )
        ).upper()

        # -----------------------------------------------------------
        # Scene-level radiometric normalization
        # -----------------------------------------------------------

        stretch_low, stretch_high = _compute_scene_stretch(
            src,
            bands=bands,
        )

        # -----------------------------------------------------------
        # Build inference grid
        #
        # This is ALWAYS the native source pixel grid.
        #
        # No 5 cm -> 1 m RGB resampling occurs here.
        # -----------------------------------------------------------

        inference_grid = _build_inference_grid(
            width=native_width,
            height=native_height,
            transform=native_transform,
            crs=native_crs,
            gsd_m=native_gsd_m,
        )

        LOGGER.info(
            "Inference grid: %dx%d native pixels",
            inference_grid.width,
            inference_grid.height,
        )

        LOGGER.info(
            "Inference GSD: %s m/pixel",
            (
                f"{native_gsd_m:.6f}"
                if native_gsd_m is not None
                else "unknown"
            ),
        )

        # -----------------------------------------------------------
        # Build canonical downstream grid separately.
        #
        # This is where the 1 m processing grid remains.
        # -----------------------------------------------------------

        if (
            scene_kind in SUPPORTED_KINDS
            and native_crs is not None
        ):
            canonical_grid = _build_canonical_grid(
                src=src,
                native_crs=native_crs,
                native_transform=native_transform,
                train_gsd_m=train_gsd_m,
                native_gsd_m=native_gsd_m,
            )

        else:
            # Non-georeferenced fallback.
            #
            # This preserves the native image geometry for relative
            # inference. It is NOT claimed to be a metric canonical grid.
            canonical_grid = _build_relative_grid(
                width=native_width,
                height=native_height,
                transform=native_transform,
                crs=native_crs,
                gsd_m=native_gsd_m,
            )

        LOGGER.info(
            "Canonical processing grid: %dx%d @ %.3f m/pixel",
            canonical_grid.width,
            canonical_grid.height,
            canonical_grid.gsd_m,
        )

        # -----------------------------------------------------------
        # IMPORTANT:
        #
        # Read RGB directly from src.
        #
        # Do NOT use a WarpedVRT here.
        #
        # Therefore the source crop remains native-resolution.
        # -----------------------------------------------------------

        tiles = _create_tiles(
            reader=src,
            bands=bands,
            grid=inference_grid,
            output_dir=tiles_dir,
            tile_size=tile_size,
            overlap=overlap,
            stretch_low=stretch_low,
            stretch_high=stretch_high,
        )

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    stride = int(
        round(
            tile_size
            * (1.0 - overlap)
        )
    )

    metadata = {
        "stage": "S1_TILING",

        "input_path": str(
            input_path
        ),

        "tile_size": int(
            tile_size
        ),

        "overlap": float(
            overlap
        ),

        "stride": stride,

        "tile_count": len(
            tiles
        ),

        "bands": band_map,

        "radiometry": {
            "input_range": "[0,1]",

            "stretch_type": (
                "none"
                if all(
                    low == 0.0
                    and high == 1.0
                    for low, high in zip(
                        stretch_low,
                        stretch_high,
                    )
                )
                else "scene_percentile"
            ),

            "percentile_low": STRETCH_LOW,
            "percentile_high": STRETCH_HIGH,

            "channel_low": stretch_low,
            "channel_high": stretch_high,
        },

        # -----------------------------------------------------------
        # TWO-GRID DESIGN
        # -----------------------------------------------------------

        "grid_roles": {
            "inference_grid": (
                "Native source pixel grid used for RGB tile "
                "extraction and neural-network inference."
            ),

            "canonical_grid": (
                "Metric processing grid used downstream for "
                "FABDEM alignment, ground estimation and DSM composition."
            ),

            "prediction_resampling": (
                "Native stitched prediction is resampled to the "
                "canonical grid after model inference."
            ),
        },

        "inference_grid": asdict(
            inference_grid
        ),

        "canonical_grid": asdict(
            canonical_grid
        ),

        "tiles": [
            asdict(
                tile
            )
            for tile in tiles
        ],
    }

    metadata_path = (
        output_dir
        / "tiling_meta.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "Prepared %d native-resolution tiles.",
        len(tiles),
    )

    LOGGER.info(
        "Tile geometry: %dx%d, %.1f%% overlap, stride=%d",
        tile_size,
        tile_size,
        overlap * 100.0,
        stride,
    )

    LOGGER.info(
        "Inference grid: %dx%d",
        inference_grid.width,
        inference_grid.height,
    )

    LOGGER.info(
        "Canonical grid: %dx%d @ %.3f m",
        canonical_grid.width,
        canonical_grid.height,
        canonical_grid.gsd_m,
    )

    return TilingResult(
        output_dir=output_dir,

        inference_grid=inference_grid,

        canonical_grid=canonical_grid,

        tile_size=tile_size,

        stride=stride,

        tile_count=len(
            tiles
        ),

        stretch_low=stretch_low,

        stretch_high=stretch_high,

        metadata_path=metadata_path,
    )


# ---------------------------------------------------------------------
# Inference grid
# ---------------------------------------------------------------------


def _build_inference_grid(
    *,
    width: int,
    height: int,
    transform: Affine,
    crs: Optional[CRS],
    gsd_m: Optional[float],
) -> InferenceGrid:
    """
    Build the native source grid used for RGB inference.

    No reprojection or resampling is performed here.
    """

    bounds = _bounds_from_transform(
        transform,
        width,
        height,
    )

    return InferenceGrid(
        width=int(
            width
        ),

        height=int(
            height
        ),

        crs=(
            str(crs)
            if crs is not None
            else None
        ),

        transform=[
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ],

        gsd_m=(
            float(gsd_m)
            if gsd_m is not None
            else None
        ),

        bounds=[
            float(value)
            for value in bounds
        ],

        native=True,
    )


# ---------------------------------------------------------------------
# Canonical grid
# ---------------------------------------------------------------------


def _build_canonical_grid(
    *,
    src: rasterio.io.DatasetReader,
    native_crs: CRS,
    native_transform: Affine,
    train_gsd_m: float,
    native_gsd_m: Optional[float],
) -> CanonicalGrid:
    """
    Build a projected metric grid for downstream processing.

    For georeferenced imagery the scene center determines the UTM CRS.

    IMPORTANT:
        This grid is NOT used to read the RGB inference tiles.

        It exists for downstream FABDEM/ground/DSM processing.
    """

    bounds_native = _bounds_from_transform(
        native_transform,
        src.width,
        src.height,
    )

    bounds_wgs84 = transform_bounds(
        native_crs,
        "EPSG:4326",
        *bounds_native,
        densify_pts=21,
    )

    centre_lon = (
        bounds_wgs84[0]
        + bounds_wgs84[2]
    ) / 2.0

    centre_lat = (
        bounds_wgs84[1]
        + bounds_wgs84[3]
    ) / 2.0

    utm_crs = CRS.from_user_input(
        _utm_crs_for_centre(
            centre_lon,
            centre_lat,
        )
    )

    projected_bounds = transform_bounds(
        native_crs,
        utm_crs,
        *bounds_native,
        densify_pts=21,
    )

    left = projected_bounds[0]
    bottom = projected_bounds[1]
    right = projected_bounds[2]
    top = projected_bounds[3]

    width = max(
        1,
        int(
            math.ceil(
                (right - left)
                / train_gsd_m
            )
        ),
    )

    height = max(
        1,
        int(
            math.ceil(
                (top - bottom)
                / train_gsd_m
            )
        ),
    )

    # Snap the canonical origin to the requested resolution.
    origin_x = (
        math.floor(
            left
            / train_gsd_m
        )
        * train_gsd_m
    )

    origin_y = (
        math.ceil(
            top
            / train_gsd_m
        )
        * train_gsd_m
    )

    transform = Affine(
        train_gsd_m,
        0.0,
        origin_x,
        0.0,
        -train_gsd_m,
        origin_y,
    )

    bounds = _bounds_from_transform(
        transform,
        width,
        height,
    )

    return CanonicalGrid(
        width=width,

        height=height,

        crs=str(
            utm_crs
        ),

        transform=[
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ],

        gsd_m=float(
            train_gsd_m
        ),

        bounds=[
            float(value)
            for value in bounds
        ],

        native_width=int(
            src.width
        ),

        native_height=int(
            src.height
        ),

        native_crs=str(
            native_crs
        ),

        native_transform=[
            float(native_transform.a),
            float(native_transform.b),
            float(native_transform.c),
            float(native_transform.d),
            float(native_transform.e),
            float(native_transform.f),
        ],

        native_gsd_m=(
            float(native_gsd_m)
            if native_gsd_m is not None
            else None
        ),

        canonical=True,
    )


def _build_relative_grid(
    *,
    width: int,
    height: int,
    transform: Affine,
    crs: Optional[CRS],
    gsd_m: Optional[float],
) -> CanonicalGrid:
    """
    Non-georeferenced fallback.

    Keeps the native raster geometry.

    This is not a metric processing grid.
    """

    if crs is None:
        crs_text = ""
    else:
        crs_text = str(
            crs
        )

    bounds = _bounds_from_transform(
        transform,
        width,
        height,
    )

    return CanonicalGrid(
        width=int(
            width
        ),

        height=int(
            height
        ),

        crs=crs_text,

        transform=[
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ],

        gsd_m=(
            float(gsd_m)
            if gsd_m is not None
            else 1.0
        ),

        bounds=[
            float(value)
            for value in bounds
        ],

        native_width=int(
            width
        ),

        native_height=int(
            height
        ),

        native_crs=(
            str(crs)
            if crs is not None
            else None
        ),

        native_transform=[
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ],

        native_gsd_m=(
            float(gsd_m)
            if gsd_m is not None
            else None
        ),

        canonical=False,
    )


# ---------------------------------------------------------------------
# Tile generation
# ---------------------------------------------------------------------


def _create_tiles(
    *,
    reader: rasterio.io.DatasetReader,
    bands: Sequence[int],
    grid: InferenceGrid,
    output_dir: Path,
    tile_size: int,
    overlap: float,
    stretch_low: Sequence[float],
    stretch_high: Sequence[float],
) -> List[TileWindow]:
    """
    Create overlapping RGB tiles on the native inference grid.
    """

    stride = int(
        round(
            tile_size
            * (1.0 - overlap)
        )
    )

    x_starts = _tile_starts(
        grid.width,
        tile_size,
        stride,
    )

    y_starts = _tile_starts(
        grid.height,
        tile_size,
        stride,
    )

    LOGGER.info(
        "Native tile layout: %d columns x %d rows",
        len(x_starts),
        len(y_starts),
    )

    tiles: List[TileWindow] = []

    index = 0

    for y in y_starts:

        for x in x_starts:

            valid_width = min(
                tile_size,
                max(
                    0,
                    grid.width - x,
                ),
            )

            valid_height = min(
                tile_size,
                max(
                    0,
                    grid.height - y,
                ),
            )

            data = _read_tile(
                reader=reader,
                bands=bands,
                x=x,
                y=y,
                tile_size=tile_size,
                valid_width=valid_width,
                valid_height=valid_height,
                stretch_low=stretch_low,
                stretch_high=stretch_high,
            )

            tile_path = (
                output_dir
                / f"tile_{index:06d}.npy"
            )

            np.save(
                tile_path,
                data,
                allow_pickle=False,
            )

            tile = TileWindow(
                index=index,

                x=int(x),

                y=int(y),

                width=tile_size,

                height=tile_size,

                valid_x=0,

                valid_y=0,

                valid_width=int(
                    valid_width
                ),

                valid_height=int(
                    valid_height
                ),
            )

            tiles.append(
                tile
            )

            index += 1

    return tiles


def _read_tile(
    *,
    reader: rasterio.io.DatasetReader,
    bands: Sequence[int],
    x: int,
    y: int,
    tile_size: int,
    valid_width: int,
    valid_height: int,
    stretch_low: Sequence[float],
    stretch_high: Sequence[float],
) -> np.ndarray:
    """
    Read one native-resolution RGB tile.

    Output:
        float32
        shape=(3, tile_size, tile_size)
        values in [0, 1]
    """

    if (
        valid_width <= 0
        or valid_height <= 0
    ):
        raise ValueError(
            "Tile has no valid pixels."
        )

    window = rasterio.windows.Window(
        col_off=x,
        row_off=y,
        width=valid_width,
        height=valid_height,
    )

    raw = reader.read(
        indexes=list(
            bands
        ),
        window=window,
        out_dtype="float32",
        boundless=False,
    )

    if raw.shape[0] != 3:
        raise ValueError(
            f"Expected three RGB bands, got {raw.shape}."
        )

    # Convert to [0,1] while preserving the source spatial resolution.
    data = _radiometric_normalize(
        raw,
        low=stretch_low,
        high=stretch_high,
    )

    pad_bottom = (
        tile_size
        - valid_height
    )

    pad_right = (
        tile_size
        - valid_width
    )

    if (
        pad_bottom > 0
        or pad_right > 0
    ):
        data = _pad_reflect(
            data,
            pad_bottom=pad_bottom,
            pad_right=pad_right,
        )

    if data.shape != (
        3,
        tile_size,
        tile_size,
    ):
        raise RuntimeError(
            f"Unexpected tile shape: {data.shape}"
        )

    return np.asarray(
        data,
        dtype=np.float32,
    )


def _pad_reflect(
    data: np.ndarray,
    *,
    pad_bottom: int,
    pad_right: int,
) -> np.ndarray:
    """
    Reflect-pad spatial dimensions.

    A one-pixel raster cannot use NumPy's reflect mode, so it falls
    back to edge replication for that axis.
    """

    mode_y = (
        "reflect"
        if data.shape[1] > 1
        else "edge"
    )

    mode_x = (
        "reflect"
        if data.shape[2] > 1
        else "edge"
    )

    # Pad Y first, then X because NumPy does not support a different
    # mode per axis in one np.pad call.
    padded = np.pad(
        data,
        (
            (
                0,
                0,
            ),
            (
                0,
                pad_bottom,
            ),
            (
                0,
                0,
            ),
        ),
        mode=mode_y,
    )

    padded = np.pad(
        padded,
        (
            (
                0,
                0,
            ),
            (
                0,
                0,
            ),
            (
                0,
                pad_right,
            ),
        ),
        mode=mode_x,
    )

    return padded


def _tile_starts(
    size: int,
    tile_size: int,
    stride: int,
) -> List[int]:
    """
    Return deterministic tile starts covering the complete dimension.

    The final tile is shifted back to the boundary so no pixels are
    omitted.
    """

    if size <= tile_size:
        return [0]

    starts: List[int] = []

    current = 0

    while current + tile_size < size:

        starts.append(
            current
        )

        current += stride

    final_start = (
        size
        - tile_size
    )

    if (
        not starts
        or starts[-1] != final_start
    ):
        starts.append(
            final_start
        )

    return starts


# ---------------------------------------------------------------------
# Radiometry
# ---------------------------------------------------------------------


def _compute_scene_stretch(
    src: rasterio.io.DatasetReader,
    *,
    bands: Sequence[int],
) -> Tuple[List[float], List[float]]:
    """
    Compute one scene-level 2-98 percentile stretch per RGB channel.

    For uint8 input we preserve the natural [0,255] range.

    For other integer/float imagery, the stretch is computed once
    from a decimated scene overview and then reused for every tile.
    """

    dtype = np.dtype(
        src.dtypes[
            bands[0] - 1
        ]
    )

    if dtype == np.dtype(
        "uint8"
    ):
        return (
            [0.0, 0.0, 0.0],
            [255.0, 255.0, 255.0],
        )

    # Read a small scene overview.
    scale = min(
        1.0,
        1024.0
        / max(
            src.width,
            src.height,
        ),
    )

    out_width = max(
        1,
        int(
            round(
                src.width
                * scale
            )
        ),
    )

    out_height = max(
        1,
        int(
            round(
                src.height
                * scale
            )
        ),
    )

    sample = src.read(
        indexes=list(
            bands
        ),
        out_shape=(
            3,
            out_height,
            out_width,
        ),
        resampling=Resampling.average,
        out_dtype="float32",
    )

    lows: List[float] = []
    highs: List[float] = []

    for channel in range(3):

        values = sample[
            channel
        ]

        values = values[
            np.isfinite(
                values
            )
        ]

        if values.size == 0:

            lows.append(
                0.0
            )

            highs.append(
                1.0
            )

            continue

        low = float(
            np.percentile(
                values,
                STRETCH_LOW,
            )
        )

        high = float(
            np.percentile(
                values,
                STRETCH_HIGH,
            )
        )

        if not math.isfinite(
            low
        ):
            low = 0.0

        if not math.isfinite(
            high
        ):
            high = low + 1.0

        if high <= low:
            high = low + 1.0

        lows.append(
            low
        )

        highs.append(
            high
        )

    return lows, highs


def _radiometric_normalize(
    data: np.ndarray,
    *,
    low: Sequence[float],
    high: Sequence[float],
) -> np.ndarray:
    """
    Normalize RGB to [0,1].
    """

    result = np.empty_like(
        data,
        dtype=np.float32,
    )

    for channel in range(3):

        channel_data = data[
            channel
        ]

        lo = float(
            low[channel]
        )

        hi = float(
            high[channel]
        )

        if hi <= lo:

            result[
                channel
            ] = 0.0

            continue

        channel_norm = (
            channel_data
            - lo
        ) / (
            hi - lo
        )

        result[
            channel
        ] = np.clip(
            channel_norm,
            0.0,
            1.0,
        )

    result[
        ~np.isfinite(
            result
        )
    ] = 0.0

    return result


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------


def _resolve_band_map(
    scene_meta: Mapping[str, Any],
    count: int,
) -> Dict[str, int]:
    """
    Resolve RGB band numbers.
    """

    value = scene_meta.get(
        "band_map"
    )

    if isinstance(
        value,
        Mapping,
    ):

        result: Dict[str, int] = {}

        for key in (
            "red",
            "green",
            "blue",
        ):

            raw = value.get(
                key
            )

            if raw is not None:

                result[key] = int(
                    raw
                )

        if all(
            key in result
            for key in (
                "red",
                "green",
                "blue",
            )
        ):

            _validate_band_numbers(
                result,
                count,
            )

            return result

    if count < 3:
        raise ValueError(
            "Prototype requires at least three RGB bands."
        )

    return {
        "red": 1,
        "green": 2,
        "blue": 3,
    }


def _validate_band_numbers(
    bands: Mapping[str, int],
    count: int,
) -> None:

    for name in (
        "red",
        "green",
        "blue",
    ):

        index = int(
            bands[name]
        )

        if not (
            1
            <= index
            <= count
        ):

            raise ValueError(
                f"Band mapping for {name}={index} "
                f"is outside image band count {count}."
            )


def _resolve_native_crs(
    scene_meta: Mapping[str, Any],
    source_crs: Optional[CRS],
) -> Optional[CRS]:
    """
    Resolve CRS from scene metadata first, then raster metadata.
    """

    value = scene_meta.get(
        "crs"
    )

    if value:

        try:

            return CRS.from_user_input(
                value
            )

        except Exception:

            LOGGER.warning(
                "Could not parse CRS from scene metadata; "
                "falling back to raster CRS."
            )

    return (
        CRS.from_user_input(
            source_crs
        )
        if source_crs is not None
        else None
    )


def _resolve_native_transform(
    scene_meta: Mapping[str, Any],
    source_transform: Affine,
) -> Affine:
    """
    Resolve affine transform from scene metadata first,
    then raster transform.
    """

    value = scene_meta.get(
        "transform"
    )

    if value is None:
        return source_transform

    try:

        values = [
            float(v)
            for v in value
        ]

        if len(values) >= 6:

            return Affine(
                *values[:6]
            )

    except (
        TypeError,
        ValueError,
    ):

        LOGGER.warning(
            "Could not parse transform from scene metadata; "
            "falling back to raster transform."
        )

    return source_transform


def _optional_float(
    value: Any,
) -> Optional[float]:
    """
    Parse an optional finite float.
    """

    if value is None:
        return None

    try:

        number = float(
            value
        )

        if math.isfinite(
            number
        ):
            return number

    except (
        TypeError,
        ValueError,
    ):
        pass

    return None


def _utm_crs_for_centre(
    lon: float,
    lat: float,
) -> str:
    """
    Select the UTM CRS containing the scene centre.
    """

    zone = int(
        math.floor(
            (lon + 180.0)
            / 6.0
        )
        + 1
    )

    zone = max(
        1,
        min(
            60,
            zone,
        ),
    )

    epsg = (
        32600 + zone
        if lat >= 0.0
        else 32700 + zone
    )

    return f"EPSG:{epsg}"


def _bounds_from_transform(
    transform: Affine,
    width: int,
    height: int,
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    """
    Calculate raster bounds from affine transform.
    """

    corners = [
        transform * (
            0,
            0,
        ),

        transform * (
            width,
            0,
        ),

        transform * (
            0,
            height,
        ),

        transform * (
            width,
            height,
        ),
    ]

    xs = [
        point[0]
        for point in corners
    ]

    ys = [
        point[1]
        for point in corners
    ]

    return (
        float(
            min(xs)
        ),

        float(
            min(ys)
        ),

        float(
            max(xs)
        ),

        float(
            max(ys)
        ),
    )


def _read_json(
    path: Path,
) -> Dict[str, Any]:
    """
    Read a JSON object.
    """

    try:

        value = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Invalid JSON: {path}: {exc}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _validate_arguments(
    *,
    train_gsd_m: float,
    tile_size: int,
    overlap: float,
) -> None:

    if (
        not math.isfinite(
            train_gsd_m
        )
        or train_gsd_m <= 0.0
    ):

        raise ValueError(
            f"train_gsd_m must be > 0, got {train_gsd_m}"
        )

    if tile_size <= 0:

        raise ValueError(
            "tile_size must be positive."
        )

    if not (
        0.0
        <= overlap
        < 1.0
    ):

        raise ValueError(
            "overlap must satisfy 0 <= overlap < 1."
        )

    stride = int(
        round(
            tile_size
            * (1.0 - overlap)
        )
    )

    if stride <= 0:

        raise ValueError(
            "Overlap produces zero stride."
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "GeoSculpt native-resolution RGB tiling. "
            "Creates overlapping source-resolution tiles while "
            "retaining a separate canonical processing grid."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Path to the input RGB image."
        ),
    )

    parser.add_argument(
        "--scene-meta",
        required=True,
        type=Path,
        help=(
            "Path to scene_meta.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory for tiles and tiling_meta.json."
        ),
    )

    parser.add_argument(
        "--train-gsd-m",
        type=float,
        default=DEFAULT_TRAIN_GSD_M,
        help=(
            "Canonical downstream processing GSD in metres/pixel. "
            "This does NOT resample RGB before inference."
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=(
            "Native RGB tile size. Default: 1024."
        ),
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help=(
            "Tile overlap fraction. Default: 0.5."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:

    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.debug
            else logging.INFO
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    try:

        result = prepare_tiles(
            input_path=args.input,
            scene_meta_path=args.scene_meta,
            output_dir=args.output_dir,
            train_gsd_m=args.train_gsd_m,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )

        print(
            json.dumps(
                {
                    "output_dir": str(
                        result.output_dir
                    ),

                    "tile_count": result.tile_count,

                    "tile_size": result.tile_size,

                    "stride": result.stride,

                    "inference_grid": asdict(
                        result.inference_grid
                    ),

                    "canonical_grid": asdict(
                        result.canonical_grid
                    ),

                    "metadata": str(
                        result.metadata_path
                    ),
                },
                indent=2,
            )
        )

        return 0

    except Exception as exc:

        LOGGER.error(
            "%s",
            exc,
        )

        if args.debug:

            LOGGER.exception(
                "Tiling failed."
            )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )