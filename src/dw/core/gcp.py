# src/dw/core/gcp.py

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine


SUPPORTED_GCP_EXTENSIONS = {
    ".csv",
    ".json",
}


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class GCP:
    """
    One ground control point.

    x/y:
        Source-coordinate values in source_crs.

    z:
        Reference elevation in metres.

    id:
        Stable identifier from the source file.
    """

    id: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class GCPSet:
    """
    Validated collection of GCPs sharing one source CRS.
    """

    source_path: str
    source_crs: str

    gcps: tuple[GCP, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_crs": self.source_crs,
            "count": len(self.gcps),
            "gcps": [
                asdict(gcp)
                for gcp in self.gcps
            ],
        }


# ---------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------


def _normalise_crs(value: Any) -> CRS:
    if value is None:
        raise ValueError(
            "GCP CRS is missing. Provide --gcp-crs or include a "
            "top-level/per-record CRS in the GCP JSON."
        )

    text = str(value).strip()

    if not text:
        raise ValueError(
            "GCP CRS is empty. Provide --gcp-crs or a CRS in the file."
        )

    try:
        return CRS.from_user_input(text)

    except Exception as exc:
        raise ValueError(
            f"Invalid GCP CRS: {text}"
        ) from exc


def _crs_equal(
    left: CRS,
    right: CRS,
) -> bool:
    return bool(
        left == right
    )


# ---------------------------------------------------------------------
# Primitive validation
# ---------------------------------------------------------------------


def _parse_float(
    value: Any,
    field_name: str,
    gcp_id: str,
) -> float:
    try:
        number = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"GCP '{gcp_id}' has non-numeric {field_name}: {value!r}"
        ) from exc

    if not np.isfinite(number):
        raise ValueError(
            f"GCP '{gcp_id}' has non-finite {field_name}: {value!r}"
        )

    return number


def _parse_id(
    value: Any,
    row_number: int,
) -> str:
    if value is None:
        raise ValueError(
            f"GCP row {row_number} is missing 'id'."
        )

    text = str(value).strip()

    if not text:
        raise ValueError(
            f"GCP row {row_number} has an empty 'id'."
        )

    return text


def _validate_unique_ids(
    gcps: list[GCP],
) -> None:
    seen: set[str] = set()

    for gcp in gcps:
        if gcp.id in seen:
            raise ValueError(
                f"Duplicate GCP id: '{gcp.id}'."
            )

        seen.add(gcp.id)


# ---------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------


def _load_csv_records(
    path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"GCP CSV has no header: {path}"
            )

        fields = {
            str(field).strip().lower()
            for field in reader.fieldnames
            if field is not None
        }

        required = {
            "id",
            "x",
            "y",
            "z",
        }

        missing = sorted(
            required - fields
        )

        if missing:
            raise ValueError(
                "GCP CSV is missing required columns: "
                + ", ".join(missing)
                + ". Expected at least: id,x,y,z"
            )

        rows: list[dict[str, Any]] = []

        for row_number, raw_row in enumerate(
            reader,
            start=2,
        ):
            normalised: dict[str, Any] = {}

            for key, value in raw_row.items():
                if key is None:
                    continue

                normalised[
                    str(key).strip().lower()
                ] = value

            # Preserve an optional CRS column if supplied.
            rows.append(
                normalised
            )

    embedded_crs = None

    crs_values = {
        str(row["crs"]).strip()
        for row in rows
        if row.get("crs") not in {
            None,
            "",
        }
    }

    if len(crs_values) > 1:
        raise ValueError(
            "GCP CSV contains multiple different CRS values. "
            "All GCPs must use one common CRS."
        )

    if crs_values:
        embedded_crs = next(
            iter(crs_values)
        )

    return (
        rows,
        embedded_crs,
    )


# ---------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------


def _load_json_records(
    path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid GCP JSON: {path}: {exc}"
        ) from exc

    embedded_crs = None

    if isinstance(
        value,
        dict,
    ):
        embedded_crs = value.get(
            "crs"
        )

        records = value.get(
            "gcps"
        )

        if records is None:
            raise ValueError(
                "GCP JSON object must contain a 'gcps' array."
            )

    elif isinstance(
        value,
        list,
    ):
        records = value

    else:
        raise ValueError(
            "GCP JSON must be either a list of GCP records or "
            "an object containing 'gcps'."
        )

    if not isinstance(
        records,
        list,
    ):
        raise ValueError(
            "GCP JSON 'gcps' must be an array."
        )

    normalised_records: list[dict[str, Any]] = []

    for item in records:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Each GCP JSON record must be an object."
            )

        normalised_records.append(
            {
                str(key).strip().lower(): value
                for key, value in item.items()
            }
        )

    return (
        normalised_records,
        str(embedded_crs).strip()
        if embedded_crs not in {None, ""}
        else None,
    )


# ---------------------------------------------------------------------
# Public GCP loader
# ---------------------------------------------------------------------


def load_gcps(
    path: str | Path,
    *,
    default_crs: str | None = None,
) -> GCPSet:
    """
    Load and validate a GCP CSV or JSON file.

    CSV:
        Required columns:
            id,x,y,z

        CRS:
            supplied with default_crs / --gcp-crs

            or an optional 'crs' column.

    JSON:
        Either:

            [
                {
                    "id": "1",
                    "x": ...,
                    "y": ...,
                    "z": ...
                }
            ]

        or:

            {
                "crs": "EPSG:25830",
                "gcps": [...]
            }

    Returns
    -------
    GCPSet
    """

    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"GCP file not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_GCP_EXTENSIONS:
        raise ValueError(
            f"Unsupported GCP file type: {path.suffix}. "
            "Expected .csv or .json."
        )

    if path.suffix.lower() == ".csv":
        records, embedded_crs = _load_csv_records(
            path
        )

    else:
        records, embedded_crs = _load_json_records(
            path
        )

    selected_crs = (
        default_crs
        if default_crs not in {None, ""}
        else embedded_crs
    )

    if (
        default_crs not in {None, ""}
        and embedded_crs not in {None, ""}
    ):
        default_obj = _normalise_crs(
            default_crs
        )

        embedded_obj = _normalise_crs(
            embedded_crs
        )

        if not _crs_equal(
            default_obj,
            embedded_obj,
        ):
            raise ValueError(
                "GCP CRS conflict: --gcp-crs does not match "
                "the CRS embedded in the GCP file."
            )

        selected_crs = default_crs

    source_crs = _normalise_crs(
        selected_crs
    )

    gcps: list[GCP] = []

    for row_number, record in enumerate(
        records,
        start=1,
    ):
        gcp_id = _parse_id(
            record.get("id"),
            row_number,
        )

        x = _parse_float(
            record.get("x"),
            "x",
            gcp_id,
        )

        y = _parse_float(
            record.get("y"),
            "y",
            gcp_id,
        )

        z = _parse_float(
            record.get("z"),
            "z",
            gcp_id,
        )

        gcps.append(
            GCP(
                id=gcp_id,
                x=x,
                y=y,
                z=z,
            )
        )

    if not gcps:
        raise ValueError(
            f"GCP file contains no GCP records: {path}"
        )

    _validate_unique_ids(
        gcps
    )

    return GCPSet(
        source_path=str(path),
        source_crs=source_crs.to_string(),
        gcps=tuple(gcps),
    )


# ---------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------


def _affine_from_value(
    value: Any,
) -> Affine:
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
        "Invalid affine transform in grid metadata."
    )


def _read_required_grid(
    tiling_meta: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    grid = tiling_meta.get(
        name
    )

    if not isinstance(
        grid,
        dict,
    ):
        raise ValueError(
            f"Tiling metadata is missing '{name}'."
        )

    required = {
        "width",
        "height",
        "crs",
        "transform",
    }

    missing = [
        key
        for key in required
        if grid.get(key) is None
    ]

    if missing:
        raise ValueError(
            f"Tiling metadata '{name}' is missing: "
            + ", ".join(missing)
        )

    return grid


# ---------------------------------------------------------------------
# Coordinate → pixel conversion
# ---------------------------------------------------------------------


def map_gcps_to_grid(
    gcps: GCPSet,
    *,
    target_crs: str,
    transform: Affine,
    width: int,
    height: int,
    grid_name: str,
) -> list[dict[str, Any]]:
    """
    Reproject GCPs to target_crs and map them to pixel coordinates.

    Pixel coordinates use the NumPy/raster convention where:
        col=0,row=0
    refers to the first pixel center.

    The raster affine transform maps from pixel-corner coordinates,
    so a 0.5 pixel correction is applied.
    """

    if width <= 0 or height <= 0:
        raise ValueError(
            f"{grid_name}: invalid grid dimensions "
            f"{width}x{height}."
        )

    target_crs_obj = _normalise_crs(
        target_crs
    )

    source_crs_obj = _normalise_crs(
        gcps.source_crs
    )

    try:
        transformer = Transformer.from_crs(
            source_crs_obj,
            target_crs_obj,
            always_xy=True,
        )

    except Exception as exc:
        raise ValueError(
            f"Unable to create GCP CRS transformation from "
            f"{source_crs_obj} to {target_crs_obj}."
        ) from exc

    try:
        inverse = ~transform

    except Exception as exc:
        raise ValueError(
            f"{grid_name}: raster transform is not invertible."
        ) from exc

    mapped: list[dict[str, Any]] = []

    for gcp in gcps.gcps:
        try:
            target_x, target_y = transformer.transform(
                gcp.x,
                gcp.y,
            )

        except Exception as exc:
            raise ValueError(
                f"Failed to transform GCP '{gcp.id}' "
                f"from {gcps.source_crs} to {target_crs}."
            ) from exc

        corner_col, corner_row = inverse * (
            target_x,
            target_y,
        )

        # Raster affine coordinates refer to the upper-left
        # pixel corner. Convert to pixel-center index coordinates.
        pixel_col = (
            float(corner_col) - 0.5
        )

        pixel_row = (
            float(corner_row) - 0.5
        )

        inside = (
            0.0
            <= pixel_col
            <= float(width - 1)
            and
            0.0
            <= pixel_row
            <= float(height - 1)
        )

        mapped.append(
            {
                "id": gcp.id,
                "source": {
                    "x": float(gcp.x),
                    "y": float(gcp.y),
                    "z": float(gcp.z),
                    "crs": gcps.source_crs,
                },
                "target": {
                    "x": float(target_x),
                    "y": float(target_y),
                    "crs": target_crs_obj.to_string(),
                },
                "pixel": {
                    "col": pixel_col,
                    "row": pixel_row,
                    "inside": bool(inside),
                },
            }
        )

    return mapped


# ---------------------------------------------------------------------
# Full scene GCP context
# ---------------------------------------------------------------------


def build_gcp_context(
    gcps: GCPSet,
    tiling_meta_path: str | Path,
) -> dict[str, Any]:
    """
    Map each GCP to both:
        1. native inference grid
        2. canonical metric grid

    The context is the Phase 2 handoff artifact for calibration.
    """

    tiling_meta_path = (
        Path(tiling_meta_path)
        .expanduser()
        .resolve()
    )

    if not tiling_meta_path.exists():
        raise FileNotFoundError(
            f"Tiling metadata not found: {tiling_meta_path}"
        )

    try:
        tiling_meta = json.loads(
            tiling_meta_path.read_text(
                encoding="utf-8",
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid tiling metadata: {tiling_meta_path}"
        ) from exc

    if not isinstance(
        tiling_meta,
        dict,
    ):
        raise ValueError(
            "Tiling metadata must be a JSON object."
        )

    inference_grid = _read_required_grid(
        tiling_meta,
        "inference_grid",
    )

    canonical_grid = _read_required_grid(
        tiling_meta,
        "canonical_grid",
    )

    native_mapped = map_gcps_to_grid(
        gcps,
        target_crs=str(
            inference_grid["crs"]
        ),
        transform=_affine_from_value(
            inference_grid["transform"]
        ),
        width=int(
            inference_grid["width"]
        ),
        height=int(
            inference_grid["height"]
        ),
        grid_name="inference_grid",
    )

    canonical_mapped = map_gcps_to_grid(
        gcps,
        target_crs=str(
            canonical_grid["crs"]
        ),
        transform=_affine_from_value(
            canonical_grid["transform"]
        ),
        width=int(
            canonical_grid["width"]
        ),
        height=int(
            canonical_grid["height"]
        ),
        grid_name="canonical_grid",
    )

    native_by_id = {
        item["id"]: item
        for item in native_mapped
    }

    canonical_by_id = {
        item["id"]: item
        for item in canonical_mapped
    }

    outside_native = [
        item["id"]
        for item in native_mapped
        if not item["pixel"]["inside"]
    ]

    outside_canonical = [
        item["id"]
        for item in canonical_mapped
        if not item["pixel"]["inside"]
    ]

    if outside_native:
        raise ValueError(
            "GCPs outside native image bounds: "
            + ", ".join(outside_native)
        )

    if outside_canonical:
        raise ValueError(
            "GCPs outside canonical processing bounds: "
            + ", ".join(outside_canonical)
        )

    merged_gcps: list[dict[str, Any]] = []

    for gcp in gcps.gcps:
        merged_gcps.append(
            {
                "id": gcp.id,
                "x": float(gcp.x),
                "y": float(gcp.y),
                "z": float(gcp.z),
                "crs": gcps.source_crs,
                "native_pixel": native_by_id[
                    gcp.id
                ]["pixel"],
                "canonical_pixel": canonical_by_id[
                    gcp.id
                ]["pixel"],
                "native_target": native_by_id[
                    gcp.id
                ]["target"],
                "canonical_target": canonical_by_id[
                    gcp.id
                ]["target"],
            }
        )

    return {
        "stage": "S2_GCP_ROUTING",
        "status": "READY_FOR_PHASE3_CALIBRATION",
        "source_file": gcps.source_path,
        "source_crs": gcps.source_crs,
        "count": len(
            gcps.gcps
        ),
        "gcps": merged_gcps,
        "inference_grid": {
            "width": int(
                inference_grid["width"]
            ),
            "height": int(
                inference_grid["height"]
            ),
            "crs": str(
                inference_grid["crs"]
            ),
            "transform": inference_grid[
                "transform"
            ],
        },
        "canonical_grid": {
            "width": int(
                canonical_grid["width"]
            ),
            "height": int(
                canonical_grid["height"]
            ),
            "crs": str(
                canonical_grid["crs"]
            ),
            "transform": canonical_grid[
                "transform"
            ],
            "gsd_m": canonical_grid.get(
                "gsd_m"
            ),
        },
        "phase3_handoff": {
            "reference_elevation_field": "z",
            "z_units": "metres",
            "fit_variables": [
                "a",
                "b",
            ],
            "future_residual_model": (
                "z_ref - DTM = a * R + b"
            ),
        },
    }


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------


def save_gcp_context(
    context: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            context,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path