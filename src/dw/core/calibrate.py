# src/dw/core/calibrate.py

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from pyproj import CRS, Transformer


LOGGER = logging.getLogger(__name__)


# =====================================================================
# Defaults
# =====================================================================

DEFAULT_SCALE_PRIOR = 7.12

# Keep run.py --scale-sigma aligned with this module-level default.
# run.py imports this constant and passes the CLI value explicitly.
DEFAULT_SCALE_SIGMA = 2.0

DEFAULT_OFFSET_PRIOR = 0.0
DEFAULT_OFFSET_SIGMA = 10.0

DEFAULT_GCP_TAU_M = 2.0

# GCP quality-gate defaults. These are intentionally conservative for the
# prototype: warn on weak relative spread, and reject an unstable LOO fit
# rather than silently composing a metric DSM from an unreliable calibration.
DEFAULT_GCP_LOO_RATIO_LIMIT = 3.0
DEFAULT_GCP_LOO_ACTION = "error"
DEFAULT_GCP_MIN_R_SPREAD = 0.05

MIN_RELATIVE_SPREAD = 1.0e-12

# GCP Z handling.  ``as_supplied`` preserves the legacy behaviour;
# ``ellipsoidal`` converts ellipsoidal ETRS89 heights to orthometric-like
# heights using the supplied EGM08-REDNAP geoid undulation N via H = h - N.
GCP_Z_REFERENCE_OPTIONS = {
    "as_supplied",
    "ellipsoidal",
    "orthometric",
}
DEFAULT_GCP_Z_REFERENCE = "as_supplied"
DEFAULT_GCP_GEOID_GRID = Path(
    "data/geoids/es_ign_EGM08_REDNAP.tif"
)


# =====================================================================
# JSON helpers
# =====================================================================

def _load_json(
    path: str | Path,
) -> dict[str, Any]:
    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        value = json.load(
            f
        )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"Expected a JSON object in {path}."
        )

    return value


def _json_safe(
    value: Any,
) -> Any:
    """
    Convert numpy values and non-finite floats into JSON-safe values.
    """

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        (
            np.floating,
            np.integer,
        ),
    ):
        value = value.item()

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

    return value


def _write_json(
    path: str | Path,
    value: dict[str, Any],
) -> None:
    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            _json_safe(
                value
            ),
            f,
            indent=2,
        )


# =====================================================================
# Statistics
# =====================================================================

def _rmse(
    residuals: Sequence[float] | np.ndarray,
) -> float:
    values = np.asarray(
        residuals,
        dtype=np.float64,
    ).reshape(
        -1
    )

    mask = np.isfinite(
        values
    )

    if not np.any(
        mask
    ):
        return float(
            "nan"
        )

    return float(
        np.sqrt(
            np.mean(
                values[mask] ** 2
            )
        )
    )


def _mae(
    residuals: Sequence[float] | np.ndarray,
) -> float:
    values = np.asarray(
        residuals,
        dtype=np.float64,
    ).reshape(
        -1
    )

    mask = np.isfinite(
        values
    )

    if not np.any(
        mask
    ):
        return float(
            "nan"
        )

    return float(
        np.mean(
            np.abs(
                values[mask]
            )
        )
    )


def _median(
    values: Sequence[float] | np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(
        -1
    )

    mask = np.isfinite(
        values
    )

    if not np.any(
        mask
    ):
        return float(
            "nan"
        )

    return float(
        np.median(
            values[mask]
        )
    )


def _safe_float(
    value: Any,
) -> float:
    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return float(
            "nan"
        )

    if not math.isfinite(
        result
    ):
        return float(
            "nan"
        )

    return result


# =====================================================================
# Raw raster helpers
# =====================================================================

def _load_raw_array(
    path: str | Path,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Load a Float32 raster written with ndarray.tofile().
    """

    path = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Raw raster not found: {path}"
        )

    height = int(
        shape[0]
    )

    width = int(
        shape[1]
    )

    expected_count = (
        height
        * width
    )

    data = np.fromfile(
        path,
        dtype=np.float32,
    )

    if data.size != expected_count:
        raise ValueError(
            f"Raw raster size mismatch for {path}: "
            f"expected {expected_count} Float32 values, "
            f"found {data.size}."
        )

    return data.reshape(
        (
            height,
            width,
        )
    )


# =====================================================================
# Canonical grid helpers
# =====================================================================

def _extract_canonical_grid(
    tiling_meta: dict[str, Any],
) -> tuple[int, int]:
    """
    Extract canonical width and height.

    Current schema:

        canonical_grid.width
        canonical_grid.height

    Compatibility layouts are also supported.
    """

    candidates: list[
        dict[str, Any]
    ] = []

    for key in (
        "canonical_grid",
        "canonical",
        "grid",
    ):
        value = tiling_meta.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):
            candidates.append(
                value
            )

    candidates.append(
        tiling_meta
    )

    for candidate in candidates:

        width = candidate.get(
            "width"
        )

        height = candidate.get(
            "height"
        )

        if (
            width is not None
            and height is not None
        ):

            width_i = int(
                width
            )

            height_i = int(
                height
            )

            if (
                width_i <= 0
                or height_i <= 0
            ):
                raise ValueError(
                    "Canonical grid width and height "
                    "must both be positive."
                )

            return (
                width_i,
                height_i,
            )

    width = tiling_meta.get(
        "canonical_width"
    )

    height = tiling_meta.get(
        "canonical_height"
    )

    if (
        width is not None
        and height is not None
    ):

        width_i = int(
            width
        )

        height_i = int(
            height
        )

        if (
            width_i <= 0
            or height_i <= 0
        ):
            raise ValueError(
                "Canonical grid width and height "
                "must both be positive."
            )

        return (
            width_i,
            height_i,
        )

    raise KeyError(
        "Could not determine canonical width/height "
        "from tiling_meta.json."
    )


# =====================================================================
# Bilinear sampling
# =====================================================================

def _bilinear_sample(
    array: np.ndarray,
    row: float,
    col: float,
) -> float:
    """
    Bilinearly sample a 2-D raster at floating-point row/column.
    """

    data = np.asarray(
        array,
        dtype=np.float64,
    )

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2-D raster, got shape={data.shape}."
        )

    height, width = data.shape

    if (
        height <= 0
        or width <= 0
    ):
        raise ValueError(
            "Cannot sample an empty raster."
        )

    row_value = float(
        row
    )

    col_value = float(
        col
    )

    if not (
        math.isfinite(
            row_value
        )
        and math.isfinite(
            col_value
        )
    ):
        return float(
            "nan"
        )

    row_value = float(
        np.clip(
            row_value,
            0.0,
            float(
                height - 1
            ),
        )
    )

    col_value = float(
        np.clip(
            col_value,
            0.0,
            float(
                width - 1
            ),
        )
    )

    r0 = int(
        math.floor(
            row_value
        )
    )

    c0 = int(
        math.floor(
            col_value
        )
    )

    r1 = min(
        r0 + 1,
        height - 1,
    )

    c1 = min(
        c0 + 1,
        width - 1,
    )

    fr = (
        row_value
        - float(r0)
    )

    fc = (
        col_value
        - float(c0)
    )

    q00 = float(
        data[
            r0,
            c0,
        ]
    )

    q01 = float(
        data[
            r0,
            c1,
        ]
    )

    q10 = float(
        data[
            r1,
            c0,
        ]
    )

    q11 = float(
        data[
            r1,
            c1,
        ]
    )

    values = np.asarray(
        [
            q00,
            q01,
            q10,
            q11,
        ],
        dtype=np.float64,
    )

    finite = np.isfinite(
        values
    )

    if not np.all(
        finite
    ):
        finite_values = values[
            finite
        ]

        if finite_values.size == 0:
            return float(
                "nan"
            )

        return float(
            np.mean(
                finite_values
            )
        )

    top = (
        q00
        * (1.0 - fc)
        + q01
        * fc
    )

    bottom = (
        q10
        * (1.0 - fc)
        + q11
        * fc
    )

    return float(
        top
        * (1.0 - fr)
        + bottom
        * fr
    )


# =====================================================================
# GCP vertical-reference helpers
# =====================================================================

def _resolve_gcp_vertical_reference(
    value: str,
) -> str:
    reference = str(
        value
    ).strip().lower()

    aliases = {
        "raw": "as_supplied",
        "as-supplied": "as_supplied",
        "as_supplied": "as_supplied",
        "ellipsoid": "ellipsoidal",
        "ellipsoidal": "ellipsoidal",
        "orthometric": "orthometric",
    }

    reference = aliases.get(
        reference,
        reference,
    )

    if reference not in GCP_Z_REFERENCE_OPTIONS:
        raise ValueError(
            "gcp_z_reference must be one of "
            f"{sorted(GCP_Z_REFERENCE_OPTIONS)}; got {value!r}."
        )

    return reference


def _sample_geoid_undulation(
    geoid_grid_path: str | Path,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    source_crs: str | CRS,
) -> np.ndarray:
    """
    Sample a geoid-undulation raster at projected GCP coordinates.

    The geoid raster is sampled in its native CRS after transforming the
    source GCP X/Y coordinates.  Continuous geoid values are bilinearly
    interpolated by rasterio's point sampler through the four surrounding
    pixels implemented below.
    """

    path = (
        Path(geoid_grid_path)
        .expanduser()
        .resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"GCP geoid grid not found: {path}"
        )

    xs = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)
    ys = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    if xs.size != ys.size:
        raise ValueError(
            "Geoid sample X/Y arrays must have the same size."
        )

    if xs.size == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    if not (
        np.all(np.isfinite(xs))
        and np.all(np.isfinite(ys))
    ):
        raise ValueError(
            "GCP X/Y coordinates must be finite for geoid sampling."
        )

    source = CRS.from_user_input(
        source_crs
    )

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(
                f"Geoid grid has no CRS: {path}"
            )

        transformer = Transformer.from_crs(
            source,
            src.crs,
            always_xy=True,
        )

        data = src.read(
            1,
            masked=False,
        ).astype(
            np.float64,
            copy=False,
        )

        nodata = src.nodata
        values: list[float] = []
        inv = ~src.transform

        for x_value, y_value in zip(xs, ys):
            lon_or_x, lat_or_y = transformer.transform(
                float(x_value),
                float(y_value),
            )
            col_value, row_value = inv * (
                float(lon_or_x),
                float(lat_or_y),
            )
            if not (
                math.isfinite(float(row_value))
                and math.isfinite(float(col_value))
            ):
                values.append(float("nan"))
                continue

            r0 = int(math.floor(float(row_value)))
            c0 = int(math.floor(float(col_value)))
            r1 = r0 + 1
            c1 = c0 + 1

            if (
                r0 < 0
                or c0 < 0
                or r1 >= src.height
                or c1 >= src.width
            ):
                values.append(float("nan"))
                continue

            fr = float(row_value) - float(r0)
            fc = float(col_value) - float(c0)

            q = np.asarray(
                [
                    data[r0, c0],
                    data[r0, c1],
                    data[r1, c0],
                    data[r1, c1],
                ],
                dtype=np.float64,
            )

            finite = np.isfinite(q)
            if nodata is not None:
                finite &= ~np.isclose(
                    q,
                    float(nodata),
                    rtol=0.0,
                    atol=1e-12,
                )

            if not np.all(finite):
                q = q[finite]
                if q.size == 0:
                    values.append(float("nan"))
                    continue
                values.append(float(np.mean(q)))
                continue

            top = q[0] * (1.0 - fc) + q[1] * fc
            bottom = q[2] * (1.0 - fc) + q[3] * fc
            values.append(float(top * (1.0 - fr) + bottom * fr))

    result = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(result)):
        bad = np.flatnonzero(~np.isfinite(result))
        raise ValueError(
            "EGM08-REDNAP geoid sampling failed for GCP indices: "
            f"{bad.tolist()}. Check the grid CRS/coverage and GCP coordinates."
        )

    return result


def _convert_gcp_heights(
    records: list[dict[str, Any]],
    *,
    z_reference: str,
    geoid_grid: str | Path | None,
    gcp_crs: str | CRS | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Convert GCP elevations into the vertical reference used for DTM/DSM.

    For ``ellipsoidal`` GCP heights:

        H = h - N

    where N is sampled from the EGM08-REDNAP geoid grid.
    """

    reference = _resolve_gcp_vertical_reference(
        z_reference
    )

    if reference == "as_supplied":
        converted = []
        for record in records:
            item = dict(record)
            item["z_raw_m"] = float(record["z_ref"])
            item["geoid_undulation_m"] = None
            item["z_ref_m"] = float(record["z_ref"])
            converted.append(item)

        return (
            converted,
            {
                "input_reference": "as_supplied",
                "output_reference": "unchanged",
                "geoid_grid": None,
                "formula": "H = Z (no vertical conversion)",
            },
        )

    if reference == "orthometric":
        converted = []
        for record in records:
            item = dict(record)
            item["z_raw_m"] = float(record["z_ref"])
            item["geoid_undulation_m"] = None
            item["z_ref_m"] = float(record["z_ref"])
            converted.append(item)

        return (
            converted,
            {
                "input_reference": "orthometric",
                "output_reference": "orthometric",
                "geoid_grid": None,
                "formula": "H = Z (orthometric heights already supplied)",
            },
        )

    if gcp_crs is None:
        raise ValueError(
            "gcp_crs is required when converting ellipsoidal GCP heights."
        )

    if geoid_grid is None:
        geoid_grid = DEFAULT_GCP_GEOID_GRID

    xs = [
        float(record["x"])
        for record in records
        if "x" in record
    ]
    ys = [
        float(record["y"])
        for record in records
        if "y" in record
    ]

    if len(xs) != len(records) or len(ys) != len(records):
        missing = [
            str(record["id"])
            for record in records
            if "x" not in record or "y" not in record
        ]
        raise ValueError(
            "Ellipsoidal GCP height conversion requires original X/Y "
            "coordinates in gcp_context.json. Missing for GCPs: "
            f"{missing}"
        )

    geoid_n = _sample_geoid_undulation(
        geoid_grid,
        xs,
        ys,
        source_crs=gcp_crs,
    )

    converted = []
    for record, n_value in zip(records, geoid_n):
        item = dict(record)
        z_raw = float(record["z_ref"])
        n_m = float(n_value)
        item["z_raw_m"] = z_raw
        item["geoid_undulation_m"] = n_m
        item["z_ref_m"] = z_raw - n_m
        converted.append(item)

    return (
        converted,
        {
            "input_reference": "ellipsoidal",
            "output_reference": "EGM08-REDNAP orthometric-like H",
            "geoid_grid": str(
                Path(geoid_grid).expanduser().resolve()
            ),
            "formula": "H = h - N",
        },
    )


def _fit_offset_only(
    observed_agl: Sequence[float] | np.ndarray,
    *,
    tau: float = DEFAULT_GCP_TAU_M,
) -> dict[str, Any]:
    """
    Fit only the vertical offset b while keeping scale fixed at the prior.

    The robust point estimate is the median of the observed ground offsets.
    """

    y = np.asarray(
        observed_agl,
        dtype=np.float64,
    ).reshape(-1)

    y = y[np.isfinite(y)]

    if y.size < 2:
        raise ValueError(
            "At least 2 finite observations are required for offset-only calibration."
        )

    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(
            "tau must be finite and > 0."
        )

    b = float(np.median(y))
    residuals = y - b
    rmse_m = _rmse(residuals)
    mae_m = _mae(residuals)
    median_error_m = _median(residuals)

    mad = float(np.median(np.abs(residuals)))
    robust_sigma = 1.4826 * mad
    sigma_b = max(
        tau / math.sqrt(float(y.size)),
        robust_sigma / math.sqrt(float(y.size)),
    )

    return {
        "b": b,
        "sigma_b": float(sigma_b),
        "rmse_m": rmse_m,
        "mae_m": mae_m,
        "median_error_m": median_error_m,
        "n": int(y.size),
    }


def _leave_one_out_offset(
    y: np.ndarray,
    ids: Sequence[str],
    *,
    tau: float,
) -> dict[str, Any]:
    """Leave one GCP out for the fixed-scale/offset-only calibration."""

    n = int(y.size)
    if n < 3:
        return {
            "enabled": False,
            "reason": "At least 3 GCPs are required for leave-one-out validation.",
            "per_gcp": [],
            "summary": {
                "count": 0,
                "rmse_m": float("nan"),
                "mae_m": float("nan"),
            },
        }

    per_gcp: list[dict[str, Any]] = []
    residuals: list[float] = []

    for index in range(n):
        mask = np.ones(n, dtype=bool)
        mask[index] = False
        fit = _fit_offset_only(
            y[mask],
            tau=tau,
        )
        prediction = float(fit["b"])
        residual = float(y[index] - prediction)
        residuals.append(residual)
        per_gcp.append({
            "id": str(ids[index]),
            "predicted_agl_m": prediction,
            "observed_agl_m": float(y[index]),
            "residual_m": residual,
            "absolute_error_m": abs(residual),
        })

    return {
        "enabled": True,
        "per_gcp": per_gcp,
        "summary": {
            "count": n,
            "rmse_m": _rmse(residuals),
            "mae_m": _mae(residuals),
        },
    }


# =====================================================================
# GCP extraction
# =====================================================================

def _extract_gcp_records(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Normalize supported GCP-context schemas.

    Current Phase 2:

        z
        canonical_pixel.row
        canonical_pixel.col

    Phase 3 synthetic tests:

        z_ref
        canonical.row
        canonical.col

    Older flat:

        z_ref
        row
        col
    """

    if not isinstance(
        context,
        dict,
    ):
        raise ValueError(
            "GCP context must be a JSON object."
        )

    raw_gcps = context.get(
        "gcps"
    )

    if not isinstance(
        raw_gcps,
        list,
    ):
        raise ValueError(
            "gcp_context.json does not contain "
            "a valid 'gcps' list."
        )

    records: list[
        dict[str, Any]
    ] = []

    for index, item in enumerate(
        raw_gcps
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        # ---------------------------------------------------------
        # ID
        # ---------------------------------------------------------

        gcp_id = item.get(
            "id",
            index + 1,
        )

        gcp_id = str(
            gcp_id
        )

        # ---------------------------------------------------------
        # Reference elevation
        # ---------------------------------------------------------

        z_ref = item.get(
            "z_ref"
        )

        if z_ref is None:
            z_ref = item.get(
                "z"
            )

        # ---------------------------------------------------------
        # Canonical pixel
        # ---------------------------------------------------------

        canonical = item.get(
            "canonical_pixel"
        )

        if not isinstance(
            canonical,
            dict,
        ):
            canonical = item.get(
                "canonical"
            )

        if isinstance(
            canonical,
            dict,
        ):

            row = canonical.get(
                "row"
            )

            col = canonical.get(
                "col"
            )

        else:

            row = item.get(
                "row"
            )

            col = item.get(
                "col"
            )

        if (
            z_ref is None
            or row is None
            or col is None
        ):
            continue

        try:

            z_value = float(
                z_ref
            )

            row_value = float(
                row
            )

            col_value = float(
                col
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if not (
            np.isfinite(
                z_value
            )
            and np.isfinite(
                row_value
            )
            and np.isfinite(
                col_value
            )
        ):
            continue

        record: dict[str, Any] = {
            "id": gcp_id,
            "z_ref": z_value,
            "row": row_value,
            "col": col_value,
        }

        # Preserve original projected coordinates when available.

        x = item.get(
            "x"
        )

        y = item.get(
            "y"
        )

        if x is not None:

            x_value = _safe_float(
                x
            )

            if math.isfinite(
                x_value
            ):
                record[
                    "x"
                ] = x_value

        if y is not None:

            y_value = _safe_float(
                y
            )

            if math.isfinite(
                y_value
            ):
                record[
                    "y"
                ] = y_value

        records.append(
            record
        )

    if not records:
        raise ValueError(
            "No canonical GCP records with "
            "id, z_ref, row and col could be extracted "
            "from gcp_context.json."
        )

    return records


# =====================================================================
# Bayesian affine fitting
# =====================================================================

def fit_affine_map(
    relative_agl: Sequence[float] | np.ndarray,
    observed_agl: Sequence[float] | np.ndarray,
    *,
    scale_prior: float = DEFAULT_SCALE_PRIOR,
    scale_sigma: float = DEFAULT_SCALE_SIGMA,
    offset_prior: float = DEFAULT_OFFSET_PRIOR,
    offset_sigma: float = DEFAULT_OFFSET_SIGMA,
    tau: float = DEFAULT_GCP_TAU_M,
) -> dict[str, Any]:
    """
    Fit:

        observed_agl = a * relative_agl + b

    using Gaussian priors on a and b.
    """

    x = np.asarray(
        relative_agl,
        dtype=np.float64,
    ).reshape(
        -1
    )

    y = np.asarray(
        observed_agl,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if x.size != y.size:
        raise ValueError(
            "relative_agl and observed_agl must have "
            "the same number of values."
        )

    if x.size < 2:
        raise ValueError(
            "At least 2 observations are required "
            "for affine calibration."
        )

    finite = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    if np.count_nonzero(
        finite
    ) < 2:
        raise ValueError(
            "At least 2 finite observations are required "
            "for affine calibration."
        )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    # -------------------------------------------------------------
    # Relative spread validation
    # -------------------------------------------------------------

    spread = float(
        np.max(
            x
        )
        - np.min(
            x
        )
    )

    if (
        not math.isfinite(
            spread
        )
        or spread <= MIN_RELATIVE_SPREAD
    ):
        raise ValueError(
            "zero spread in relative_agl; "
            "cannot estimate affine scale."
        )

    # -------------------------------------------------------------
    # Validate priors
    # -------------------------------------------------------------

    scale_prior = float(
        scale_prior
    )

    scale_sigma = float(
        scale_sigma
    )

    offset_prior = float(
        offset_prior
    )

    offset_sigma = float(
        offset_sigma
    )

    tau = float(
        tau
    )

    if not math.isfinite(
        scale_prior
    ):
        raise ValueError(
            "scale_prior must be finite."
        )

    if (
        not math.isfinite(
            scale_sigma
        )
        or scale_sigma < 0.0
    ):
        raise ValueError(
            "scale_sigma must be finite and >= 0."
        )

    if not math.isfinite(
        offset_prior
    ):
        raise ValueError(
            "offset_prior must be finite."
        )

    if (
        not math.isfinite(
            offset_sigma
        )
        or offset_sigma < 0.0
    ):
        raise ValueError(
            "offset_sigma must be finite and >= 0."
        )

    if (
        not math.isfinite(
            tau
        )
        or tau <= 0.0
    ):
        raise ValueError(
            "tau must be finite and > 0."
        )

    # -------------------------------------------------------------
    # Design matrix
    # -------------------------------------------------------------

    X = np.column_stack(
        [
            x,
            np.ones_like(
                x
            ),
        ]
    )

    likelihood_precision = (
        X.T
        @ X
        / (
            tau
            ** 2
        )
    )

    likelihood_rhs = (
        X.T
        @ y
        / (
            tau
            ** 2
        )
    )

    prior_precision = np.zeros(
        (
            2,
            2,
        ),
        dtype=np.float64,
    )

    prior_rhs = np.zeros(
        2,
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # Scale prior
    # -------------------------------------------------------------

    if scale_sigma > 0.0:

        precision_a = (
            1.0
            / (
                scale_sigma
                ** 2
            )
        )

    else:

        precision_a = 1.0e12

    prior_precision[
        0,
        0,
    ] = precision_a

    prior_rhs[
        0
    ] = (
        precision_a
        * scale_prior
    )

    # -------------------------------------------------------------
    # Offset prior
    # -------------------------------------------------------------

    if offset_sigma > 0.0:

        precision_b = (
            1.0
            / (
                offset_sigma
                ** 2
            )
        )

    else:

        precision_b = 1.0e12

    prior_precision[
        1,
        1,
    ] = precision_b

    prior_rhs[
        1
    ] = (
        precision_b
        * offset_prior
    )

    precision = (
        likelihood_precision
        + prior_precision
    )

    rhs = (
        likelihood_rhs
        + prior_rhs
    )

    try:

        theta = np.linalg.solve(
            precision,
            rhs,
        )

        covariance = np.linalg.inv(
            precision
        )

    except np.linalg.LinAlgError as exc:

        raise ValueError(
            "Could not solve Bayesian affine calibration."
        ) from exc

    a = float(
        theta[0]
    )

    b = float(
        theta[1]
    )

    # -------------------------------------------------------------
    # Scale validation
    # -------------------------------------------------------------

    if (
        not math.isfinite(
            a
        )
        or a <= 0.0
    ):
        raise ValueError(
            "non-positive scale estimated by affine calibration."
        )

    if not math.isfinite(
        b
    ):
        raise ValueError(
            "non-finite offset estimated by affine calibration."
        )

    sigma_a = float(
        math.sqrt(
            max(
                0.0,
                float(
                    covariance[
                        0,
                        0,
                    ]
                ),
            )
        )
    )

    sigma_b = float(
        math.sqrt(
            max(
                0.0,
                float(
                    covariance[
                        1,
                        1,
                    ]
                ),
            )
        )
    )

    prediction = (
        a
        * x
        + b
    )

    residuals = (
        y
        - prediction
    )

    rmse_m = _rmse(
        residuals
    )

    mae_m = _mae(
        residuals
    )

    median_error_m = _median(
        residuals
    )

    return {
        "a": a,
        "b": b,

        "sigma_a": sigma_a,
        "sigma_b": sigma_b,

        "n": int(
            x.size
        ),

        "tau": tau,

        "scale_prior": scale_prior,
        "scale_sigma": scale_sigma,

        "offset_prior": offset_prior,
        "offset_sigma": offset_sigma,

        # Primary Phase 3 metrics.
        "rmse_m": rmse_m,
        "mae_m": mae_m,

        # Compatibility aliases.
        "fit_rmse": rmse_m,
        "fit_mae": mae_m,

        "median_error_m": median_error_m,
        "fit_median_error": median_error_m,
    }


# =====================================================================
# Leave-one-out validation
# =====================================================================

def _leave_one_out(
    x: np.ndarray,
    y: np.ndarray,
    ids: Sequence[str],
    *,
    scale_prior: float,
    scale_sigma: float,
    offset_prior: float,
    offset_sigma: float,
    tau: float,
) -> dict[str, Any]:
    """
    Leave one GCP out, refit, and predict the omitted GCP.
    """

    n = int(
        x.size
    )

    if n < 3:

        return {
            "enabled": False,

            "reason": (
                "At least 3 GCPs are required "
                "for leave-one-out validation."
            ),

            "per_gcp": [],

            "summary": {
                "count": 0,
                "rmse_m": float(
                    "nan"
                ),
                "mae_m": float(
                    "nan"
                ),
            },

            "rmse_m": float(
                "nan"
            ),

            "mae_m": float(
                "nan"
            ),
        }

    per_gcp: list[
        dict[str, Any]
    ] = []

    residuals: list[
        float
    ] = []

    for index in range(
        n
    ):

        mask = np.ones(
            n,
            dtype=bool,
        )

        mask[
            index
        ] = False

        try:

            fit = fit_affine_map(
                x[
                    mask
                ],
                y[
                    mask
                ],
                scale_prior=scale_prior,
                scale_sigma=scale_sigma,
                offset_prior=offset_prior,
                offset_sigma=offset_sigma,
                tau=tau,
            )

            a_loo = float(
                fit[
                    "a"
                ]
            )

            b_loo = float(
                fit[
                    "b"
                ]
            )

            observed = float(
                y[index]
            )

            predicted = (
                a_loo
                * float(
                    x[index]
                )
                + b_loo
            )

            residual = (
                observed
                - predicted
            )

            residuals.append(
                residual
            )

            per_gcp.append(
                {
                    "id": str(
                        ids[index]
                    ),

                    "a_loo": a_loo,

                    "b_loo": b_loo,

                    "relative_r": float(
                        x[index]
                    ),

                    "observed_agl_m": observed,

                    "predicted_agl_m": float(
                        predicted
                    ),

                    "residual_m": float(
                        residual
                    ),

                    "absolute_error_m": abs(
                        residual
                    ),
                }
            )

        except Exception as exc:

            per_gcp.append(
                {
                    "id": str(
                        ids[index]
                    ),

                    "error": str(
                        exc
                    ),
                }
            )

    residual_array = np.asarray(
        residuals,
        dtype=np.float64,
    )

    rmse_m = _rmse(
        residual_array
    )

    mae_m = _mae(
        residual_array
    )

    count = int(
        residual_array.size
    )

    return {
        "enabled": True,

        "per_gcp": per_gcp,

        "summary": {
            "count": count,
            "rmse_m": rmse_m,
            "mae_m": mae_m,
        },

        "rmse_m": rmse_m,

        "mae_m": mae_m,
    }


# =====================================================================
# GCP calibration
# =====================================================================

def calibrate_with_gcps(
    *,
    gcp_context_path: str | Path,
    ground_dir: str | Path,
    dem_dir: str | Path,
    tiling_meta_path: str | Path,
    output_dir: str | Path,
    scale_prior: float = DEFAULT_SCALE_PRIOR,
    scale_sigma: float = DEFAULT_SCALE_SIGMA,
    offset_prior: float = DEFAULT_OFFSET_PRIOR,
    offset_sigma: float = DEFAULT_OFFSET_SIGMA,
    tau: float = DEFAULT_GCP_TAU_M,
    max_agl_m: float | None = None,
    gcp_loo_ratio_limit: float = DEFAULT_GCP_LOO_RATIO_LIMIT,
    gcp_loo_action: str = DEFAULT_GCP_LOO_ACTION,
    gcp_min_r_spread: float = DEFAULT_GCP_MIN_R_SPREAD,
    gcp_z_reference: str = DEFAULT_GCP_Z_REFERENCE,
    gcp_geoid_grid: str | Path | None = None,
    gcp_crs: str | CRS | None = None,
) -> Path:
    """
    Phase 3 GCP calibration.

    Model:

        z_ref - DTM = a * R + b
    """

    gcp_context_path = (
        Path(
            gcp_context_path
        )
        .expanduser()
        .resolve()
    )

    ground_dir = (
        Path(
            ground_dir
        )
        .expanduser()
        .resolve()
    )

    dem_dir = (
        Path(
            dem_dir
        )
        .expanduser()
        .resolve()
    )

    tiling_meta_path = (
        Path(
            tiling_meta_path
        )
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(
            output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gcp_loo_ratio_limit = float(gcp_loo_ratio_limit)
    gcp_loo_action = str(gcp_loo_action).strip().lower()
    gcp_min_r_spread = float(gcp_min_r_spread)

    if (
        not math.isfinite(gcp_loo_ratio_limit)
        or gcp_loo_ratio_limit <= 0.0
    ):
        raise ValueError(
            "gcp_loo_ratio_limit must be finite and > 0."
        )

    if gcp_loo_action not in {"error", "fallback_to_prior"}:
        raise ValueError(
            "gcp_loo_action must be 'error' or 'fallback_to_prior'."
        )

    if (
        not math.isfinite(gcp_min_r_spread)
        or gcp_min_r_spread < 0.0
    ):
        raise ValueError(
            "gcp_min_r_spread must be finite and >= 0."
        )

    # -------------------------------------------------------------
    # Load GCP context
    # -------------------------------------------------------------

    context = _load_json(
        gcp_context_path
    )

    records = _extract_gcp_records(
        context
    )

    LOGGER.info(
        "Extracted %d GCP records.",
        len(
            records
        ),
    )

    records, vertical_reference = _convert_gcp_heights(
        records,
        z_reference=gcp_z_reference,
        geoid_grid=gcp_geoid_grid,
        gcp_crs=gcp_crs,
    )

    LOGGER.info(
        "GCP vertical reference: %s -> %s",
        vertical_reference["input_reference"],
        vertical_reference["output_reference"],
    )

    if vertical_reference.get("geoid_grid"):
        LOGGER.info(
            "GCP geoid grid: %s",
            vertical_reference["geoid_grid"],
        )

    # -------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------

    tiling_meta = _load_json(
        tiling_meta_path
    )

    width, height = (
        _extract_canonical_grid(
            tiling_meta
        )
    )

    shape = (
        height,
        width,
    )

    # -------------------------------------------------------------
    # Load maps
    # -------------------------------------------------------------

    R_path = (
        ground_dir
        / "R.dat"
    )

    DTM_path = (
        dem_dir
        / "dtm.dat"
    )

    R = _load_raw_array(
        R_path,
        shape,
    )

    DTM = _load_raw_array(
        DTM_path,
        shape,
    )

    # -------------------------------------------------------------
    # Gather usable GCP observations
    # -------------------------------------------------------------

    usable: list[
        dict[str, Any]
    ] = []

    excluded: list[
        dict[str, Any]
    ] = []

    relative_values: list[
        float
    ] = []

    observed_values: list[
        float
    ] = []

    ids: list[
        str
    ] = []

    warnings: list[
        str
    ] = []

    for record in records:

        gcp_id = str(
            record[
                "id"
            ]
        )

        row = float(
            record[
                "row"
            ]
        )

        col = float(
            record[
                "col"
            ]
        )

        z_ref = float(
            record.get(
                "z_ref_m",
                record["z_ref"],
            )
        )

        # ---------------------------------------------------------
        # Bounds
        # ---------------------------------------------------------

        if (
            row < 0.0
            or row > float(
                height - 1
            )
            or col < 0.0
            or col > float(
                width - 1
            )
        ):

            message = (
                f"GCP {gcp_id} falls outside "
                "the canonical grid."
            )

            warnings.append(
                message
            )

            excluded.append(
                {
                    "id": gcp_id,
                    "reason": message,
                }
            )

            continue

        # ---------------------------------------------------------
        # Bilinear sample
        # ---------------------------------------------------------

        r_value = _bilinear_sample(
            R,
            row,
            col,
        )

        dtm_value = _bilinear_sample(
            DTM,
            row,
            col,
        )

        if not (
            math.isfinite(
                r_value
            )
            and math.isfinite(
                dtm_value
            )
            and math.isfinite(
                z_ref
            )
        ):

            message = (
                f"GCP {gcp_id} produced non-finite "
                "R/DTM/z values."
            )

            warnings.append(
                message
            )

            excluded.append(
                {
                    "id": gcp_id,
                    "reason": message,
                }
            )

            continue

        # ---------------------------------------------------------
        # Reference AGL
        # ---------------------------------------------------------

        observed_agl = (
            z_ref
            - dtm_value
        )

        # ---------------------------------------------------------
        # Optional sanity limit
        # ---------------------------------------------------------

        if (
            max_agl_m is not None
            and math.isfinite(
                float(
                    max_agl_m
                )
            )
            and float(
                max_agl_m
            ) > 0.0
        ):

            limit = float(
                max_agl_m
            )

            if (
                observed_agl < -limit
                or observed_agl > limit
            ):

                message = (
                    f"GCP {gcp_id} observed AGL "
                    f"{observed_agl:.6f} m is outside "
                    f"[-{limit:.6f}, {limit:.6f}] m."
                )

                warnings.append(
                    message
                )

                excluded.append(
                    {
                        "id": gcp_id,
                        "reason": message,
                    }
                )

                continue

        relative_values.append(
            r_value
        )

        observed_values.append(
            observed_agl
        )

        ids.append(
            gcp_id
        )

        usable.append(
            {
                "id": gcp_id,

                "row": row,

                "col": col,

                "z_ref_m": z_ref,

                "z_raw_m": float(
                    record.get(
                        "z_raw_m",
                        z_ref,
                    )
                ),

                "geoid_undulation_m": record.get(
                    "geoid_undulation_m"
                ),

                "dtm_m": dtm_value,

                "relative_r": r_value,

                "observed_agl_m": observed_agl,
            }
        )

    LOGGER.info(
        "Usable GCPs: %d / %d",
        len(
            usable
        ),
        len(
            records
        ),
    )

    if len(
        usable
    ) < 2:

        raise ValueError(
            "At least 2 usable GCPs are required "
            "for Phase 3 calibration."
        )

    x = np.asarray(
        relative_values,
        dtype=np.float64,
    )

    y = np.asarray(
        observed_values,
        dtype=np.float64,
    )

    r_spread = float(np.max(x) - np.min(x))

    # Warn when the available relative-height spread is too small to
    # strongly constrain the metric scale. The relative threshold is tied
    # to a modest expected AGL span rather than to the fitted result.
    if max_agl_m is not None and math.isfinite(float(max_agl_m)):
        expected_agl_span_m = max(
            10.0,
            min(float(max_agl_m), 50.0),
        )
    else:
        expected_agl_span_m = 50.0

    expected_r_span = (
        expected_agl_span_m
        / max(abs(float(scale_prior)), 1e-6)
    )
    spread_warning_threshold = max(
        float(gcp_min_r_spread),
        0.01 * expected_r_span,
    )

    if r_spread < spread_warning_threshold:
        message = (
            "GCP relative-height spread is small: "
            f"{r_spread:.6f} < {spread_warning_threshold:.6f}. "
            "The GCPs may not contain enough height variation to "
            "constrain scale reliably; add GCPs on known-height "
            "objects such as building/roof corners when available."
        )
        warnings.append(message)
        LOGGER.warning(message)

    # -------------------------------------------------------------
    # Ground-only / zero-spread GCP branch
    # -------------------------------------------------------------

    if r_spread <= MIN_RELATIVE_SPREAD:
        message = (
            "All usable GCPs have effectively zero relative AGL spread; "
            "using GCP_OFFSET_ONLY instead of forcing an affine scale fit. "
            "The scene scale remains the supplied prior."
        )
        warnings.append(message)
        LOGGER.warning(message)

        offset_fit = _fit_offset_only(
            y,
            tau=tau,
        )

        a = float(scale_prior)
        b = float(offset_fit["b"])
        sigma_a = float(scale_sigma)
        sigma_b = float(offset_fit["sigma_b"])
        fit_rmse = float(offset_fit["rmse_m"])
        fit_mae = float(offset_fit["mae_m"])
        fit_median = float(offset_fit["median_error_m"])

        predicted = np.full_like(
            y,
            b,
            dtype=np.float64,
        )
        residuals = y - predicted

        per_gcp = []
        for index, base in enumerate(usable):
            residual = float(residuals[index])
            per_gcp.append({
                "id": str(base["id"]),
                "row": float(base["row"]),
                "col": float(base["col"]),
                "z_ref_m": float(base["z_ref_m"]),
                "z_raw_m": float(base["z_raw_m"]),
                "geoid_undulation_m": base.get("geoid_undulation_m"),
                "dtm_m": float(base["dtm_m"]),
                "relative_r": float(base["relative_r"]),
                "observed_agl_m": float(base["observed_agl_m"]),
                "predicted_agl_m": float(predicted[index]),
                "residual_m": residual,
                "absolute_error_m": abs(residual),
            })

        loo = _leave_one_out_offset(
            y,
            ids,
            tau=tau,
        )

        loo_per_gcp = loo.get(
            "per_gcp",
            [],
        )
        loo_summary = loo.get(
            "summary",
            {"count": 0, "rmse_m": float("nan"), "mae_m": float("nan")},
        )
        loo_rmse = _safe_float(loo_summary.get("rmse_m"))
        loo_mae = _safe_float(loo_summary.get("mae_m"))
        loo_count = int(loo_summary.get("count", 0))

        quality_gate = {
            "loo_ratio_limit": float(gcp_loo_ratio_limit),
            "loo_action": gcp_loo_action,
            "fit_rmse_floor_m": 0.05,
            "r_spread": r_spread,
            "r_spread_warning_threshold": spread_warning_threshold,
            "status": "PASS",
            "loo_fit_rmse_ratio": None,
            "scale_estimation": "SKIPPED_ZERO_R_SPREAD",
        }

        if loo_count > 0 and math.isfinite(loo_rmse):
            ratio = float(
                loo_rmse
                / max(fit_rmse, 0.05)
            )
            quality_gate["loo_fit_rmse_ratio"] = ratio

            if ratio > gcp_loo_ratio_limit:
                message = (
                    "GCP offset-only calibration failed the leave-one-out "
                    f"stability gate: LOO/fit RMSE ratio={ratio:.3f} exceeds "
                    f"limit={gcp_loo_ratio_limit:.3f}."
                )
                warnings.append(message)
                quality_gate["status"] = "FAIL"

                if gcp_loo_action == "error":
                    raise ValueError(message)

                fallback_path = calibrate_relative_height(
                    ground_dir=ground_dir,
                    output_dir=output_dir,
                    scale_prior=scale_prior,
                    scale_sigma=scale_sigma,
                    scale_override=None,
                )
                fallback = _load_json(fallback_path)
                fallback["stage"] = "S3_GCP_CALIBRATION"
                fallback["status"] = "FALLBACK"
                fallback["mode"] = "PRIOR_FALLBACK_FROM_GCP"
                fallback["contract"] = "z_ref - DTM = a * R + b"
                fallback["gcp_quality_gate"] = quality_gate
                fallback["gcp_quality_gate"]["fallback_reason"] = message
                fallback["gcp_vertical_reference"] = vertical_reference
                fallback["gcp_residuals"] = {
                    "per_gcp": per_gcp,
                    "n_gcps": int(len(per_gcp)),
                    "rmse_m": fit_rmse,
                    "mae_m": fit_mae,
                    "median_error_m": fit_median,
                    "leave_one_out": {
                        "enabled": bool(loo.get("enabled", False)),
                        "per_gcp": loo_per_gcp,
                        "summary": {
                            "count": loo_count,
                            "rmse_m": loo_rmse,
                            "mae_m": loo_mae,
                        },
                    },
                }
                fallback["loo_rmse"] = loo_rmse
                fallback["loo_mae"] = loo_mae
                fallback["excluded_gcps"] = excluded
                fallback["warnings"] = warnings
                _write_json(output_dir / "calibration.json", fallback)
                return output_dir / "calibration.json"

        calibration = {
            "stage": "S3_GCP_CALIBRATION",
            "status": "COMPLETE",
            "mode": "GCP_OFFSET_ONLY",
            "contract": "z_ref - DTM = a * R + b",
            "formula": "z_ref - DTM = a * R + b",
            "a": a,
            "b": b,
            "scale": a,
            "offset": b,
            "sigma_a": sigma_a,
            "sigma_b": sigma_b,
            "tau": float(tau),
            "n_gcps": int(len(per_gcp)),
            "priors": {
                "scale_prior": float(scale_prior),
                "scale_sigma": float(scale_sigma),
                "offset_prior": float(offset_prior),
                "offset_sigma": float(offset_sigma),
            },
            "scale_source": "prior",
            "offset_source": "gcp_ground_alignment",
            "gcp_vertical_reference": vertical_reference,
            "fit": {
                "rmse_m": fit_rmse,
                "mae_m": fit_mae,
                "median_error_m": fit_median,
            },
            "fit_rmse": fit_rmse,
            "fit_mae": fit_mae,
            "gcp_residuals": {
                "per_gcp": per_gcp,
                "n_gcps": int(len(per_gcp)),
                "rmse_m": fit_rmse,
                "mae_m": fit_mae,
                "median_error_m": fit_median,
                "leave_one_out": {
                    "enabled": bool(loo.get("enabled", False)),
                    "per_gcp": loo_per_gcp,
                    "summary": {
                        "count": loo_count,
                        "rmse_m": loo_rmse,
                        "mae_m": loo_mae,
                    },
                },
            },
            "loo": {
                "enabled": bool(loo.get("enabled", False)),
                "per_gcp": loo_per_gcp,
                "summary": {
                    "count": loo_count,
                    "rmse_m": loo_rmse,
                    "mae_m": loo_mae,
                },
            },
            "loo_rmse": loo_rmse,
            "loo_mae": loo_mae,
            "gcp_quality_gate": quality_gate,
            "excluded_gcps": excluded,
            "dem_evidence": {
                "status": "not_evaluated_in_phase3",
                "method": None,
                "notes": [
                    "Ground-only GCPs constrain vertical offset but not relative scale.",
                    "Scene scale remains the supplied prior because R has zero spread.",
                ],
            },
            "warnings": warnings,
        }

        calibration_path = output_dir / "calibration.json"
        _write_json(calibration_path, calibration)

        report = {
            "stage": "S3_GCP_CALIBRATION_REPORT",
            "status": "COMPLETE",
            "mode": "GCP_OFFSET_ONLY",
            "gcp_context": str(gcp_context_path),
            "canonical_grid": {"width": int(width), "height": int(height)},
            "vertical_reference": vertical_reference,
            "fit": {
                "a": a,
                "b": b,
                "sigma_a": sigma_a,
                "sigma_b": sigma_b,
                "tau_m": float(tau),
                "n_gcps": int(len(per_gcp)),
                "rmse_m": fit_rmse,
                "mae_m": fit_mae,
                "median_error_m": fit_median,
            },
            "gcp_residuals": calibration["gcp_residuals"],
            "gcp_quality_gate": quality_gate,
            "priors": calibration["priors"],
            "excluded_gcps": excluded,
            "warnings": warnings,
            "next_stage_contract": {
                "metric_agl_formula": "AGL_metric = a_prior * R + b_gcp",
                "metric_dsm_formula": "DSM = DTM + AGL_metric",
                "calibration_file": str(calibration_path),
            },
        }
        _write_json(output_dir / "phase3_calibration_report.json", report)

        LOGGER.info("Phase 3 GCP offset-only calibration complete.")
        LOGGER.info("Scale a (prior) : %.9f", a)
        LOGGER.info("Offset b (GCP)  : %.9f m", b)
        LOGGER.info("Fit RMSE        : %.6f m", fit_rmse)
        LOGGER.info("LOO RMSE        : %.6f m", loo_rmse)
        LOGGER.info("Calibration JSON: %s", calibration_path)
        return calibration_path

    # -------------------------------------------------------------
    # Fit affine map
    # -------------------------------------------------------------

    fit = fit_affine_map(
        x,
        y,
        scale_prior=scale_prior,
        scale_sigma=scale_sigma,
        offset_prior=offset_prior,
        offset_sigma=offset_sigma,
        tau=tau,
    )

    a = float(
        fit[
            "a"
        ]
    )

    b = float(
        fit[
            "b"
        ]
    )

    sigma_a = float(
        fit[
            "sigma_a"
        ]
    )

    sigma_b = float(
        fit[
            "sigma_b"
        ]
    )

    fit_rmse = float(
        fit[
            "rmse_m"
        ]
    )

    fit_mae = float(
        fit[
            "mae_m"
        ]
    )

    fit_median = float(
        fit[
            "median_error_m"
        ]
    )

    # -------------------------------------------------------------
    # Full-fit predictions and GCP residuals
    # -------------------------------------------------------------

    predicted = (
        a
        * x
        + b
    )

    residuals = (
        y
        - predicted
    )

    per_gcp: list[
        dict[str, Any]
    ] = []

    for index, base in enumerate(
        usable
    ):

        residual = float(
            residuals[
                index
            ]
        )

        predicted_value = float(
            predicted[
                index
            ]
        )

        per_gcp.append(
            {
                "id": str(
                    base[
                        "id"
                    ]
                ),

                "row": float(
                    base[
                        "row"
                    ]
                ),

                "col": float(
                    base[
                        "col"
                    ]
                ),

                "z_ref_m": float(
                    base[
                        "z_ref_m"
                    ]
                ),

                "z_raw_m": float(
                    base.get(
                        "z_raw_m",
                        base["z_ref_m"],
                    )
                ),

                "geoid_undulation_m": base.get(
                    "geoid_undulation_m"
                ),

                "dtm_m": float(
                    base[
                        "dtm_m"
                    ]
                ),

                "relative_r": float(
                    base[
                        "relative_r"
                    ]
                ),

                "observed_agl_m": float(
                    base[
                        "observed_agl_m"
                    ]
                ),

                "predicted_agl_m": predicted_value,

                "residual_m": residual,

                "absolute_error_m": abs(
                    residual
                ),
            }
        )

    # -------------------------------------------------------------
    # Leave-one-out
    # -------------------------------------------------------------

    loo = _leave_one_out(
        x,
        y,
        ids,
        scale_prior=scale_prior,
        scale_sigma=scale_sigma,
        offset_prior=offset_prior,
        offset_sigma=offset_sigma,
        tau=tau,
    )

    loo_per_gcp = loo.get(
        "per_gcp",
        [],
    )

    loo_summary = loo.get(
        "summary",
        {
            "count": 0,
            "rmse_m": float(
                "nan"
            ),
            "mae_m": float(
                "nan"
            ),
        },
    )

    loo_rmse = _safe_float(
        loo_summary.get(
            "rmse_m"
        )
    )

    loo_mae = _safe_float(
        loo_summary.get(
            "mae_m"
        )
    )

    loo_count = int(
        loo_summary.get(
            "count",
            0,
        )
    )

    # -------------------------------------------------------------
    # GCP quality gate
    # -------------------------------------------------------------

    quality_gate: dict[str, Any] = {
        "loo_ratio_limit": float(gcp_loo_ratio_limit),
        "loo_action": gcp_loo_action,
        "fit_rmse_floor_m": 0.05,
        "r_spread": r_spread,
        "r_spread_warning_threshold": spread_warning_threshold,
        "status": "PASS",
        "loo_fit_rmse_ratio": None,
    }

    calibration_mode = "GCP_FITTED"

    if loo_count > 0 and math.isfinite(loo_rmse):
        ratio = float(
            loo_rmse
            / max(
                fit_rmse,
                0.05,
            )
        )
        quality_gate["loo_fit_rmse_ratio"] = ratio

        if ratio > gcp_loo_ratio_limit:
            message = (
                "GCP calibration failed the leave-one-out stability "
                f"gate: LOO/fit RMSE ratio={ratio:.3f} exceeds "
                f"limit={gcp_loo_ratio_limit:.3f}. "
                "The GCPs may lack height-object coverage; add GCPs "
                "on known-height objects such as building corners."
            )
            warnings.append(message)
            quality_gate["status"] = "FAIL"

            if gcp_loo_action == "error":
                raise ValueError(message)

            # Graceful fallback: replace the unstable GCP result with
            # the normal Mode-2 prior calibration and preserve the reason
            # loudly in calibration.json.
            fallback_path = calibrate_relative_height(
                ground_dir=ground_dir,
                output_dir=output_dir,
                scale_prior=scale_prior,
                scale_sigma=scale_sigma,
                scale_override=None,
            )

            fallback = _load_json(fallback_path)
            fallback["stage"] = "S3_GCP_CALIBRATION"
            fallback["status"] = "FALLBACK"
            fallback["mode"] = "PRIOR_FALLBACK_FROM_GCP"
            fallback["contract"] = "z_ref - DTM = a * R + b"
            fallback["gcp_quality_gate"] = quality_gate
            fallback["gcp_quality_gate"]["fallback_reason"] = message
            fallback["gcp_residuals"] = {
                "per_gcp": per_gcp,
                "n_gcps": int(len(per_gcp)),
                "rmse_m": fit_rmse,
                "mae_m": fit_mae,
                "median_error_m": fit_median,
                "leave_one_out": {
                    "enabled": bool(loo.get("enabled", False)),
                    "per_gcp": loo_per_gcp,
                    "summary": {
                        "count": loo_count,
                        "rmse_m": loo_rmse,
                        "mae_m": loo_mae,
                    },
                },
            }
            fallback["loo"] = {
                "enabled": bool(loo.get("enabled", False)),
                "per_gcp": loo_per_gcp,
                "summary": {
                    "count": loo_count,
                    "rmse_m": loo_rmse,
                    "mae_m": loo_mae,
                },
            }
            fallback["loo_rmse"] = loo_rmse
            fallback["loo_mae"] = loo_mae
            fallback["excluded_gcps"] = excluded
            fallback["warnings"] = warnings

            _write_json(
                output_dir / "calibration.json",
                fallback,
            )

            LOGGER.error("%s", message)
            LOGGER.warning(
                "GCP calibration is unstable. Falling back to "
                "Mode-2 prior calibration."
            )
            return output_dir / "calibration.json"

    elif len(per_gcp) < 3:
        calibration_mode = "GCP_FITTED_UNVALIDATED"

        message = (
            "Fewer than 3 usable GCPs were available, so no "
            "independent leave-one-out validation of the fit was possible."
        )
        warnings.append(message)
        quality_gate["status"] = "UNVALIDATED"
        LOGGER.warning(message)

    # -------------------------------------------------------------
    # DEM evidence placeholder
    # -------------------------------------------------------------

    dem_evidence = {
        "status": "not_evaluated_in_phase3",

        "method": None,

        "notes": [
            (
                "Phase 3 fits the explicit GCP relation "
                "z_ref - DTM = a * R + b."
            ),

            (
                "Global DEM-evidence calibration remains "
                "a separate validation task."
            ),
        ],
    }

    # -------------------------------------------------------------
    # GCP residual contract
    #
    # IMPORTANT:
    #
    # The test contract expects:
    #
    # gcp_residuals.per_gcp
    # gcp_residuals.leave_one_out.summary.count
    # -------------------------------------------------------------

    gcp_residuals = {
        "per_gcp": per_gcp,

        "n_gcps": int(
            len(
                per_gcp
            )
        ),

        "rmse_m": fit_rmse,

        "mae_m": fit_mae,

        "median_error_m": fit_median,

        "leave_one_out": {
            "enabled": bool(
                loo.get(
                    "enabled",
                    False,
                )
            ),

            "per_gcp": loo_per_gcp,

            "summary": {
                "count": loo_count,

                "rmse_m": loo_rmse,

                "mae_m": loo_mae,
            },
        },
    }

    # -------------------------------------------------------------
    # Main calibration contract
    # -------------------------------------------------------------

    calibration = {
        "stage": "S3_GCP_CALIBRATION",

        "status": "COMPLETE",

        "mode": calibration_mode,

        "contract": "z_ref - DTM = a * R + b",

        "formula": (
            "z_ref - DTM = a * R + b"
        ),

        "a": a,

        "b": b,

        # compose.py compatibility
        "scale": a,

        "offset": b,

        "sigma_a": sigma_a,

        "sigma_b": sigma_b,

        "tau": float(
            tau
        ),

        "gcp_vertical_reference": vertical_reference,
        "scale_source": "gcp_affine_fit",
        "offset_source": "gcp_affine_fit",

        "n_gcps": int(
            len(
                per_gcp
            )
        ),

        "priors": {
            "scale_prior": float(
                scale_prior
            ),

            "scale_sigma": float(
                scale_sigma
            ),

            "offset_prior": float(
                offset_prior
            ),

            "offset_sigma": float(
                offset_sigma
            ),
        },

        "fit": {
            "rmse_m": fit_rmse,

            "mae_m": fit_mae,

            "median_error_m": fit_median,
        },

        "fit_rmse": fit_rmse,

        "fit_mae": fit_mae,

        "gcp_residuals": gcp_residuals,

        # Also expose LOO at top level for convenient access.
        "loo": {
            "enabled": bool(
                loo.get(
                    "enabled",
                    False,
                )
            ),

            "per_gcp": loo_per_gcp,

            "summary": {
                "count": loo_count,

                "rmse_m": loo_rmse,

                "mae_m": loo_mae,
            },
        },

        "loo_rmse": loo_rmse,

        "loo_mae": loo_mae,

        "gcp_quality_gate": quality_gate,

        "excluded_gcps": excluded,

        "dem_evidence": dem_evidence,

        "warnings": warnings,
    }

    calibration_path = (
        output_dir
        / "calibration.json"
    )

    _write_json(
        calibration_path,
        calibration,
    )

    # -------------------------------------------------------------
    # Human-readable Phase 3 report
    # -------------------------------------------------------------

    report = {
        "stage": "S3_GCP_CALIBRATION_REPORT",

        "status": "COMPLETE",

        "gcp_context": str(
            gcp_context_path
        ),

        "canonical_grid": {
            "width": int(
                width
            ),

            "height": int(
                height
            ),
        },

        "vertical_reference": vertical_reference,

        "fit": {
            "a": a,

            "b": b,

            "sigma_a": sigma_a,

            "sigma_b": sigma_b,

            "tau_m": float(
                tau
            ),

            "n_gcps": int(
                len(
                    per_gcp
                )
            ),

            "rmse_m": fit_rmse,

            "mae_m": fit_mae,

            "median_error_m": fit_median,
        },

        "gcp_residuals": gcp_residuals,

        "leave_one_out": {
            "enabled": bool(
                loo.get(
                    "enabled",
                    False,
                )
            ),

            "per_gcp": loo_per_gcp,

            "summary": {
                "count": loo_count,

                "rmse_m": loo_rmse,

                "mae_m": loo_mae,
            },
        },

        "priors": {
            "scale_prior": float(
                scale_prior
            ),

            "scale_sigma": float(
                scale_sigma
            ),

            "offset_prior": float(
                offset_prior
            ),

            "offset_sigma": float(
                offset_sigma
            ),
        },

        "excluded_gcps": excluded,

        "dem_evidence": dem_evidence,

        "warnings": warnings,

        "next_stage_contract": {
            "metric_agl_formula": (
                "AGL_metric = a * R + b"
            ),

            "metric_dsm_formula": (
                "DSM = DTM + AGL_metric"
            ),

            "calibration_file": str(
                calibration_path
            ),
        },
    }

    report_path = (
        output_dir
        / "phase3_calibration_report.json"
    )

    _write_json(
        report_path,
        report,
    )

    # -------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------

    LOGGER.info(
        "Phase 3 GCP calibration complete."
    )

    LOGGER.info(
        "GCPs used : %d",
        len(
            per_gcp
        ),
    )

    LOGGER.info(
        "Scale a  : %.9f",
        a,
    )

    LOGGER.info(
        "Offset b : %.9f m",
        b,
    )

    LOGGER.info(
        "sigma_a  : %.9f",
        sigma_a,
    )

    LOGGER.info(
        "sigma_b  : %.9f m",
        sigma_b,
    )

    LOGGER.info(
        "Fit RMSE : %.6f m",
        fit_rmse,
    )

    LOGGER.info(
        "Fit MAE  : %.6f m",
        fit_mae,
    )

    if bool(
        loo.get(
            "enabled",
            False,
        )
    ):

        LOGGER.info(
            "LOO count : %d",
            loo_count,
        )

        LOGGER.info(
            "LOO RMSE  : %.6f m",
            loo_rmse,
        )

        LOGGER.info(
            "LOO MAE   : %.6f m",
            loo_mae,
        )

    if warnings:

        LOGGER.warning(
            "Calibration warnings: %d",
            len(
                warnings
            ),
        )

    LOGGER.info(
        "Calibration JSON : %s",
        calibration_path,
    )

    LOGGER.info(
        "Phase 3 report   : %s",
        report_path,
    )

    return calibration_path


# =====================================================================
# Legacy Mode 2 calibration
# =====================================================================

def calibrate_relative_height(
    *,
    ground_dir: str | Path,
    output_dir: str | Path,
    scale_prior: float = DEFAULT_SCALE_PRIOR,
    scale_sigma: float = 0.0,
    scale_override: float | None = None,
) -> Path:
    """
    Legacy Mode 2 calibration.

    Formula:

        AGL_metric = scale * R + offset

    No scene-specific GCP calibration is performed here.
    """

    ground_dir = (
        Path(
            ground_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(
            output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if scale_override is not None:

        scale = float(
            scale_override
        )

        source = (
            "explicit_override"
        )

    else:

        scale = float(
            scale_prior
        )

        source = (
            "global_prior"
        )

    if (
        not math.isfinite(
            scale
        )
        or scale <= 0.0
    ):
        raise ValueError(
            "non-positive scale supplied to Mode 2 calibration."
        )

    scale_sigma_value = float(
        scale_sigma
    )

    if (
        not math.isfinite(
            scale_sigma_value
        )
        or scale_sigma_value < 0.0
    ):
        scale_sigma_value = 0.0

    calibration = {
        "stage": "S3_CALIBRATION",

        "status": "COMPLETE",

        "mode": "PRIOR_OR_DEM_EVIDENCE",

        "formula": (
            "AGL_metric = scale * R + offset"
        ),

        "a": scale,

        "b": 0.0,

        "scale": scale,

        "offset": 0.0,

        "sigma_a": scale_sigma_value,

        "sigma_b": 0.0,

        "tau": None,

        "n_gcps": 0,

        "source": source,

        "priors": {
            "scale_prior": float(
                scale_prior
            ),

            "scale_sigma": scale_sigma_value,

            "offset_prior": 0.0,

            "offset_sigma": 0.0,
        },

        "fit": {
            "rmse_m": None,

            "mae_m": None,

            "median_error_m": None,
        },

        "fit_rmse": None,

        "fit_mae": None,

        "gcp_residuals": {
            "per_gcp": [],

            "n_gcps": 0,

            "rmse_m": None,

            "mae_m": None,

            "median_error_m": None,

            "leave_one_out": {
                "enabled": False,

                "per_gcp": [],

                "summary": {
                    "count": 0,

                    "rmse_m": None,

                    "mae_m": None,
                },
            },
        },

        "loo": {
            "enabled": False,

            "per_gcp": [],

            "summary": {
                "count": 0,

                "rmse_m": None,

                "mae_m": None,
            },
        },

        "loo_rmse": None,

        "loo_mae": None,

        "dem_evidence": {
            "status": "not_evaluated",
        },

        "warnings": [
            (
                "No GCPs were supplied. "
                "This is not scene-specific absolute "
                "vertical calibration."
            ),
        ],
    }

    calibration_path = (
        output_dir
        / "calibration.json"
    )

    _write_json(
        calibration_path,
        calibration,
    )

    LOGGER.info(
        "Prior calibration complete."
    )

    LOGGER.info(
        "Calibration source : %s",
        source,
    )

    LOGGER.info(
        "Scale a            : %.9f",
        scale,
    )

    LOGGER.info(
        "Offset b           : %.9f m",
        0.0,
    )

    return calibration_path


# =====================================================================
# Calibration loader
# =====================================================================

def load_calibration(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load calibration.json and normalize a/b and scale/offset aliases.
    """

    calibration = _load_json(
        path
    )

    if (
        "a" not in calibration
        and "scale" in calibration
    ):

        calibration[
            "a"
        ] = calibration[
            "scale"
        ]

    if (
        "b" not in calibration
        and "offset" in calibration
    ):

        calibration[
            "b"
        ] = calibration[
            "offset"
        ]

    if (
        "scale" not in calibration
        and "a" in calibration
    ):

        calibration[
            "scale"
        ] = calibration[
            "a"
        ]

    if (
        "offset" not in calibration
        and "b" in calibration
    ):

        calibration[
            "offset"
        ] = calibration[
            "b"
        ]

    return calibration


# =====================================================================
# Apply calibration
# =====================================================================

def apply_calibration(
    relative_agl: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    """
    Apply:

        metric_agl = a * relative_agl + b
    """

    if not isinstance(
        calibration,
        dict,
    ):
        raise ValueError(
            "Calibration must be a dictionary."
        )

    a_value = calibration.get(
        "a"
    )

    if a_value is None:
        a_value = calibration.get(
            "scale"
        )

    b_value = calibration.get(
        "b"
    )

    if b_value is None:
        b_value = calibration.get(
            "offset",
            0.0,
        )

    if a_value is None:
        raise ValueError(
            "Calibration is missing a/scale."
        )

    a = float(
        a_value
    )

    b = float(
        b_value
    )

    if not (
        math.isfinite(
            a
        )
        and math.isfinite(
            b
        )
    ):
        raise ValueError(
            "Calibration a/b must be finite."
        )

    if a <= 0.0:
        raise ValueError(
            "non-positive scale in calibration."
        )

    result = (
        np.asarray(
            relative_agl,
            dtype=np.float32,
        )
        * np.float32(
            a
        )
        + np.float32(
            b
        )
    )

    return result.astype(
        np.float32
    )


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GeoSculpt relative-height calibration."
        )
    )

    parser.add_argument(
        "--ground-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--scale-prior",
        type=float,
        default=DEFAULT_SCALE_PRIOR,
    )

    parser.add_argument(
        "--scale-sigma",
        type=float,
        # Keep this aligned with DEFAULT_SCALE_SIGMA above and run.py.
        default=DEFAULT_SCALE_SIGMA,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--gcp-z-reference",
        choices=sorted(GCP_Z_REFERENCE_OPTIONS),
        default=DEFAULT_GCP_Z_REFERENCE,
        help=(
            "Vertical reference of GCP Z values. Use 'ellipsoidal' "
            "with an EGM08-REDNAP geoid grid for H=Z-N conversion."
        ),
    )

    parser.add_argument(
        "--gcp-geoid-grid",
        type=Path,
        default=DEFAULT_GCP_GEOID_GRID,
        help=(
            "EGM08-REDNAP geoid GeoTIFF used when --gcp-z-reference "
            "is ellipsoidal."
        ),
    )

    parser.add_argument(
        "--gcp-crs",
        default=None,
        help=(
            "Projected CRS of GCP X/Y coordinates for geoid sampling, "
            "for example EPSG:25830."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser


# =====================================================================
# CLI main
# =====================================================================

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
            "%(name)s | "
            "%(message)s"
        ),
    )

    try:

        calibration_path = (
            calibrate_relative_height(
                ground_dir=args.ground_dir,
                output_dir=args.output_dir,
                scale_prior=args.scale_prior,
                scale_sigma=args.scale_sigma,
                scale_override=args.scale,
            )
        )

        print(
            json.dumps(
                {
                    "calibration": str(
                        calibration_path
                    ),
                },
                indent=2,
            )
        )

        return 0

    except Exception as exc:

        LOGGER.exception(
            "Calibration failed: %s",
            exc,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )