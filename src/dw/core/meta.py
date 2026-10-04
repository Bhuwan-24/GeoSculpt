# src/dw/core/meta.py

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import ColorInterp
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds


# ---------------------------------------------------------------------
# Prototype limits
# ---------------------------------------------------------------------

MIN_SUPPORTED_GSD_M = 0.05
MAX_SUPPORTED_GSD_M = 30.0


# ---------------------------------------------------------------------
# Scene metadata
# ---------------------------------------------------------------------

@dataclass
class SceneMeta:
    path: str
    filename: str

    kind: str

    width: int
    height: int
    count: int

    crs: str | None
    utm_crs: str | None

    transform: list[float]

    bounds_native: dict[str, float] | None
    bounds_wgs84: dict[str, float] | None

    centre: dict[str, float] | None

    gsd_true_m: float | None

    dtype: str | None
    bit_depth: int | None
    dtypes: list[str]

    nodata: float | int | None
    has_alpha: bool
    has_nodata: bool
    valid_fraction: float

    band_map: dict[str, int]
    colorinterp: list[str]

    overview_levels: list[int]

    has_rpcs: bool
    has_gcps: bool
    gcp_count: int

    sidecars: dict[str, str | None]

    acquisition_metadata: dict[str, Any]

    use_dem: bool
    relative_only: bool

    warnings: list[str]

    # -------------------------------------------------------------
    # Compatibility property used by run.py
    # -------------------------------------------------------------

    @property
    def classification(self) -> str:
        return self.kind

    # -------------------------------------------------------------
    # JSON representation
    # -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        # Keep both names for compatibility.
        data["classification"] = self.kind

        return data


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _find_sidecar(
    path: Path,
    extensions: list[str],
) -> Path | None:
    """
    Find a sidecar file case-insensitively.
    """
    for extension in extensions:
        candidate = path.with_suffix(extension)

        if candidate.exists():
            return candidate

    # Explicit case-insensitive fallback.
    parent = path.parent
    stem = path.stem.lower()

    for child in parent.iterdir():
        if (
            child.is_file()
            and child.stem.lower() == stem
            and child.suffix.lower() in {
                ext.lower()
                for ext in extensions
            }
        ):
            return child

    return None


def _read_world_file(path: Path) -> Affine:
    """
    Read a standard ESRI world file.

    World-file order:

        A
        D
        B
        E
        C
        F

    where C/F are the map coordinates of the CENTER of the
    upper-left pixel.

    Rasterio transforms use the upper-left pixel corner, so we
    shift by half a pixel.
    """
    values: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            values.append(float(line))

    if len(values) != 6:
        raise ValueError(
            f"World file must contain exactly 6 values: {path}"
        )

    a, d, b, e, c, f = values

    world_transform = Affine(
        a,
        b,
        c,
        d,
        e,
        f,
    )

    # World file C/F refer to the upper-left pixel CENTER.
    # Rasterio transform refers to the upper-left pixel CORNER.
    return world_transform * Affine.translation(
        -0.5,
        -0.5,
    )


def _is_identity_transform(transform: Affine) -> bool:
    return (
        abs(transform.a - 1.0) < 1e-12
        and abs(transform.b) < 1e-12
        and abs(transform.c) < 1e-12
        and abs(transform.d) < 1e-12
        and abs(transform.e - 1.0) < 1e-12
        and abs(transform.f) < 1e-12
    )


def _transform_georeferencing_issue(
    transform: Affine,
) -> str | None:
    """
    Return a reason why an affine transform cannot be trusted as
    raster georeferencing, or None when the transform is structurally
    usable.

    This deliberately catches placeholder/identity transforms before
    CRS-based bounds/GSD calculations can make them look legitimate.
    """
    if _is_identity_transform(transform):
        return "identity transform"

    determinant = (
        float(transform.a) * float(transform.e)
        - float(transform.b) * float(transform.d)
    )

    if not np.isfinite(determinant):
        return "non-finite affine determinant"

    if abs(determinant) <= 1e-15:
        return "degenerate affine transform (determinant is approximately zero)"

    coefficients = np.array(
        [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(coefficients).all():
        return "non-finite affine coefficients"

    return None


def _resolve_transform(
    src: rasterio.io.DatasetReader,
    world_file: Path | None,
) -> Affine:
    """
    Prefer an existing valid raster transform.

    If the raster has no useful transform and a world file exists,
    use the world file.
    """
    src_transform = src.transform

    if world_file is not None:
        if _is_identity_transform(src_transform):
            return _read_world_file(world_file)

    return src_transform


def _resolve_crs(
    src: rasterio.io.DatasetReader,
    prj_file: Path | None,
) -> CRS | None:
    """
    Resolve CRS.

    Priority:
      1. CRS stored in the raster itself
      2. .prj sidecar

    A .tfw/world file does NOT define CRS.
    """
    if src.crs is not None:
        return CRS.from_user_input(src.crs)

    if prj_file is not None:
        text = prj_file.read_text(
            encoding="utf-8",
            errors="ignore",
        ).strip()

        if text:
            try:
                return CRS.from_wkt(text)
            except Exception:
                try:
                    return CRS.from_user_input(text)
                except Exception:
                    pass

    return None


def _classify_scene(
    crs: CRS | None,
    has_rpcs: bool,
    has_gcps: bool,
) -> str:
    """
    Classify the scene for the prototype pipeline.
    """
    if crs is not None:
        epsg = crs.to_epsg()

        if epsg == 3857:
            return "GEO_WEBMERC"

        if crs.is_projected:
            return "GEO_PROJECTED"

        if crs.is_geographic:
            return "GEO_GEOGRAPHIC"

    if has_rpcs:
        return "RPC_ONLY"

    if has_gcps:
        return "GCP_ONLY"

    return "NONE"


def _utm_crs_from_lon_lat(
    lon: float,
    lat: float,
) -> CRS:
    """
    Return the UTM CRS appropriate for a WGS84 coordinate.
    """
    zone = int(
        np.floor(
            (lon + 180.0) / 6.0
        ) + 1
    )

    zone = max(
        1,
        min(60, zone),
    )

    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return CRS.from_epsg(epsg)


def _get_scene_centre(
    transform: Affine,
    width: int,
    height: int,
) -> tuple[float, float]:
    """
    Get center coordinate in native raster CRS.
    """
    col = width / 2.0
    row = height / 2.0

    x, y = transform * (
        col,
        row,
    )

    return float(x), float(y)


def _compute_bounds(
    transform: Affine,
    width: int,
    height: int,
) -> dict[str, float]:
    left, bottom, right, top = array_bounds(
        height,
        width,
        transform,
    )

    return {
        "left": float(left),
        "bottom": float(bottom),
        "right": float(right),
        "top": float(top),
    }


def _compute_true_gsd(
    transform: Affine,
    width: int,
    height: int,
    native_crs: CRS,
    utm_crs: CRS,
) -> float:
    """
    Compute local pixel spacing in metres.

    One-pixel offsets are transformed into UTM and measured.
    """
    centre_x, centre_y = _get_scene_centre(
        transform,
        width,
        height,
    )

    x1, y1 = transform * (
        width / 2.0,
        height / 2.0,
    )

    x2, y2 = transform * (
        width / 2.0 + 1.0,
        height / 2.0,
    )

    x3, y3 = transform * (
        width / 2.0,
        height / 2.0 + 1.0,
    )

    transformer = Transformer.from_crs(
        native_crs,
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

    rx, ry = transformer.transform(
        x3,
        y3,
    )

    # In practice x1/y1 and centre are the same nominal pixel center
    # for our purpose, while x2/x3 represent one-pixel offsets.
    # Use local pixel-vector lengths.
    gsd_x = float(
        np.hypot(
            qx - px,
            qy - py,
        )
    )

    gsd_y = float(
        np.hypot(
            rx - px,
            ry - py,
        )
    )

    # Keep the average pixel spacing as the reported GSD.
    gsd = (
        gsd_x + gsd_y
    ) / 2.0

    # Explicit rounding avoids floating-point values such as
    # 0.049999999901 being rejected as less than 0.05.
    return round(
        float(gsd),
        6,
    )


def _transform_bounds_to_wgs84(
    bounds: dict[str, float],
    crs: CRS,
) -> dict[str, float]:
    left, bottom, right, top = transform_bounds(
        crs,
        CRS.from_epsg(4326),
        bounds["left"],
        bounds["bottom"],
        bounds["right"],
        bounds["top"],
        densify_pts=21,
    )

    return {
        "left": float(left),
        "bottom": float(bottom),
        "right": float(right),
        "top": float(top),
    }


def _centre_wgs84(
    transform: Affine,
    width: int,
    height: int,
    crs: CRS,
) -> dict[str, float]:
    x, y = _get_scene_centre(
        transform,
        width,
        height,
    )

    transformer = Transformer.from_crs(
        crs,
        CRS.from_epsg(4326),
        always_xy=True,
    )

    lon, lat = transformer.transform(
        x,
        y,
    )

    return {
        "lon": float(lon),
        "lat": float(lat),
    }


def _get_band_map(
    colorinterp: tuple[ColorInterp, ...] | list[ColorInterp],
    count: int,
) -> dict[str, int]:
    """
    Resolve RGB band numbers from color interpretation.
    """
    result: dict[str, int] = {}

    for index, interp in enumerate(
        colorinterp,
        start=1,
    ):
        if interp == ColorInterp.red:
            result["red"] = index

        elif interp == ColorInterp.green:
            result["green"] = index

        elif interp == ColorInterp.blue:
            result["blue"] = index

    # Fallback for ordinary 3-band RGB rasters.
    if count >= 3:
        result.setdefault("red", 1)
        result.setdefault("green", 2)
        result.setdefault("blue", 3)

    return result


def _sample_valid_fraction(
    src: rasterio.io.DatasetReader,
    band_map: dict[str, int],
) -> float:
    """
    Estimate valid pixel fraction using a small decimated sample.
    """
    if src.width <= 0 or src.height <= 0:
        return 0.0

    sample_width = min(
        512,
        src.width,
    )

    sample_height = min(
        512,
        src.height,
    )

    indexes = [
        band_map[name]
        for name in (
            "red",
            "green",
            "blue",
        )
        if name in band_map
    ]

    if not indexes:
        indexes = [1]

    data = src.read(
        indexes=indexes,
        out_shape=(
            len(indexes),
            sample_height,
            sample_width,
        ),
        masked=True,
        resampling=rasterio.enums.Resampling.nearest,
    )

    mask = np.ma.getmaskarray(data)

    if mask.ndim == 3:
        invalid = np.any(
            mask,
            axis=0,
        )
    else:
        invalid = mask

    if src.nodata is not None:
        raw = np.ma.getdata(data)

        nodata_mask = np.any(
            raw == src.nodata,
            axis=0,
        )

        invalid = np.logical_or(
            invalid,
            nodata_mask,
        )

    if not np.isfinite(
        np.asarray(
            np.ma.getdata(data),
            dtype=np.float32,
        )
    ).all():
        finite_mask = np.isfinite(
            np.asarray(
                np.ma.getdata(data),
                dtype=np.float32,
            )
        )

        invalid = np.logical_or(
            invalid,
            np.any(
                ~finite_mask,
                axis=0,
            ),
        )

    valid = ~invalid

    return float(
        valid.mean()
    )


# ---------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------

def resolve_scene(
    path: str | Path,
) -> SceneMeta:
    """
    Resolve scene metadata from raster headers and sidecars.

    The prototype supports:
      - GeoTIFF/internal georeferencing
      - world-file georeferencing
      - .prj CRS sidecars
      - projected/geographic/WebMercator CRS
      - RPC/GCP detection

    No full image is loaded.
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Input raster not found: {path}"
        )

    world_file = _find_sidecar(
        path,
        [
            ".tfw",
            ".TFW",
            ".wld",
            ".WLD",
            ".jgw",
            ".JGW",
            ".pgw",
            ".PGW",
        ],
    )

    prj_file = _find_sidecar(
        path,
        [
            ".prj",
            ".PRJ",
        ],
    )

    aux_xml_file = _find_sidecar(
        path,
        [
            ".aux.xml",
            ".AUX.XML",
        ],
    )

    with rasterio.open(path) as src:
        transform = _resolve_transform(
            src,
            world_file,
        )

        crs = _resolve_crs(
            src,
            prj_file,
        )

        has_rpcs = bool(
            getattr(
                src,
                "rpcs",
                None,
            )
        )

        gcps, gcp_crs = src.get_gcps()

        has_gcps = bool(
            gcps
        )

        gcp_count = (
            len(gcps)
            if gcps
            else 0
        )

        # If raster CRS itself is absent but GCP CRS exists,
        # preserve the GCP information but do not treat it as
        # standard projected georeferencing.
        if crs is None and gcp_crs is not None:
            crs = CRS.from_user_input(
                gcp_crs
            )

        warnings: list[str] = []

        transform_issue = _transform_georeferencing_issue(
            transform
        )

        if transform_issue is not None:
            kind = "NONE"
            warnings.append(
                "Georeferencing transform rejected: "
                f"{transform_issue}. Scene is treated as non-georeferenced."
            )
        else:
            kind = _classify_scene(
                crs,
                has_rpcs,
                has_gcps,
            )

        bounds_native = _compute_bounds(
            transform,
            src.width,
            src.height,
        )

        bounds_wgs84 = None
        centre = None
        utm_crs = None
        gsd_true_m = None

        # ---------------------------------------------------------
        # Georeferenced scene
        # ---------------------------------------------------------

        if crs is not None and kind in {
            "GEO_PROJECTED",
            "GEO_GEOGRAPHIC",
            "GEO_WEBMERC",
        }:
            centre = _centre_wgs84(
                transform,
                src.width,
                src.height,
                crs,
            )

            bounds_wgs84 = _transform_bounds_to_wgs84(
                bounds_native,
                crs,
            )

            utm = _utm_crs_from_lon_lat(
                centre["lon"],
                centre["lat"],
            )

            utm_crs = utm.to_string()

            gsd_true_m = _compute_true_gsd(
                transform,
                src.width,
                src.height,
                crs,
                utm,
            )

            # -----------------------------------------------------
            # GSD validation
            # -----------------------------------------------------

            # A transform can be syntactically non-degenerate while still
            # being an obvious placeholder or physically implausible image
            # scale. Keep a broad structural gate here (0.02-50 m/pixel);
            # the stricter prototype DEM gate below remains 0.05-30 m/pixel.
            if (
                not np.isfinite(gsd_true_m)
                or gsd_true_m < 0.02
                or gsd_true_m > 50.0
            ):
                warnings.append(
                    "Implied raster GSD is outside the georeferencing "
                    "sanity range 0.02-50 m/pixel: "
                    f"{gsd_true_m:.6f} m/pixel. "
                    "Scene is treated as non-georeferenced."
                )
                kind = "NONE"
                bounds_wgs84 = None
                centre = None
                utm_crs = None
                gsd_true_m = None

        else:
            if has_rpcs:
                warnings.append(
                    "Scene has RPC metadata. "
                    "RPC orthorectification is not implemented "
                    "in the prototype."
                )

            if has_gcps:
                warnings.append(
                    "Scene has GCP metadata. "
                    "GCP-based georeferencing is not implemented "
                    "in the prototype."
                )

            if world_file is not None and crs is None:
                warnings.append(
                    "World file transform is available, but no CRS "
                    "could be resolved. A .tfw file does not define "
                    "the CRS."
                )

            warnings.append(
                "Scene is not reliably georeferenced. "
                "Metric DEM/DSM claims are disabled."
            )

        # ---------------------------------------------------------
        # Validity / bands
        # ---------------------------------------------------------

        colorinterp = [
            str(item.name).lower()
            for item in src.colorinterp
        ]

        band_map = _get_band_map(
            src.colorinterp,
            src.count,
        )

        valid_fraction = _sample_valid_fraction(
            src,
            band_map,
        )

        # TIFF generally exposes itemsize in bytes.
        bit_depth = None

        if src.count > 0:
            dtype = np.dtype(
                src.dtypes[0]
            )

            bit_depth = (
                dtype.itemsize * 8
            )
        else:
            dtype = None

        overview_levels: list[int] = []

        if src.count > 0:
            overview_levels = list(
                src.overviews(1)
            )

        has_alpha = (
            ColorInterp.alpha
            in src.colorinterp
        )

        # ---------------------------------------------------------
        # DEM eligibility
        # ---------------------------------------------------------

        # Standard projected/geographic georeferencing is enough
        # for the prototype DEM path.
        #
        # We intentionally still reject scenes whose GSD is outside
        # the supported range.
        gsd_supported = (
            gsd_true_m is not None
            and (
                MIN_SUPPORTED_GSD_M
                <= gsd_true_m
                <= MAX_SUPPORTED_GSD_M
            )
        )

        use_dem = (
            kind
            in {
                "GEO_PROJECTED",
                "GEO_GEOGRAPHIC",
                "GEO_WEBMERC",
            }
            and gsd_supported
            and crs is not None
        )

        relative_only = not use_dem

        if not gsd_supported and gsd_true_m is not None:
            warnings.append(
                "Metric DEM/DSM processing is disabled because "
                "the scene GSD is outside the supported prototype range."
            )

        # ---------------------------------------------------------
        # Acquisition metadata
        # ---------------------------------------------------------

        acquisition_metadata: dict[str, Any] = {}

        # ---------------------------------------------------------
        # Construct metadata object
        # ---------------------------------------------------------

        scene_meta = SceneMeta(
            path=str(path),
            filename=path.name,

            kind=kind,

            width=int(src.width),
            height=int(src.height),
            count=int(src.count),

            crs=crs.to_string()
            if crs is not None
            else None,

            utm_crs=utm_crs,

            transform=[
                float(transform.a),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e),
                float(transform.f),
            ],

            bounds_native=bounds_native,
            bounds_wgs84=bounds_wgs84,

            centre=centre,

            gsd_true_m=gsd_true_m,

            dtype=str(
                src.dtypes[0]
            )
            if src.count > 0
            else None,

            bit_depth=bit_depth,

            dtypes=[
                str(dtype)
                for dtype in src.dtypes
            ],

            nodata=src.nodata,

            has_alpha=has_alpha,
            has_nodata=(
                src.nodata is not None
            ),

            valid_fraction=valid_fraction,

            band_map=band_map,

            colorinterp=colorinterp,

            overview_levels=overview_levels,

            has_rpcs=has_rpcs,
            has_gcps=has_gcps,
            gcp_count=gcp_count,

            sidecars={
                "world_file": (
                    str(world_file)
                    if world_file is not None
                    else None
                ),
                "prj_file": (
                    str(prj_file)
                    if prj_file is not None
                    else None
                ),
                "aux_xml_file": (
                    str(aux_xml_file)
                    if aux_xml_file is not None
                    else None
                ),
            },

            acquisition_metadata=acquisition_metadata,

            use_dem=use_dem,
            relative_only=relative_only,

            warnings=warnings,
        )

    return scene_meta


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_scene_meta(
    scene_meta: SceneMeta,
    output_path: str | Path,
) -> None:
    """
    Save scene metadata as JSON.
    """
    output_path = Path(
        output_path
    ).expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            scene_meta.to_dict(),
            f,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve remote-sensing raster metadata "
            "for the DepthWizard prototype."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input raster path.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output scene_meta.json path.",
    )

    args = parser.parse_args()

    scene_meta = resolve_scene(
        args.input
    )

    save_scene_meta(
        scene_meta,
        args.output,
    )

    print(
        json.dumps(
            scene_meta.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )