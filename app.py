"""
GeoSculpt - public Gradio deployment wrapper.

This file intentionally keeps the existing GeoSculpt processing pipeline intact.
The web layer only:
    1. accepts an RGB aerial/orthomosaic image,
    2. optionally accepts a GCP CSV,
    3. calls the existing run.py CLI,
    4. collects the generated products,
    5. exposes downloads and a GLB preview when reconstruction is available.

Designed for Hugging Face Spaces + ZeroGPU.

Repository layout expected by this app:
    app.py
    run.py
    models/best.pth
    external/Depth-Anything-V2/
    data/geoids/es_ign_EGM08_REDNAP.tif
    data/dem_cache/fabdem/       (created automatically)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Hugging Face ZeroGPU integration.
# The real package is available in the Space environment.
try:
    import spaces
except ImportError:  # local development fallback
    class _SpacesFallback:
        @staticmethod
        def GPU(*args: Any, **kwargs: Any):
            def decorator(fn):
                return fn
            return decorator

    spaces = _SpacesFallback()  # type: ignore[assignment]

import gradio as gr


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

CHECKPOINT = ROOT / "models" / "best.pth"
EXTERNAL_MODEL_DIR = ROOT / "external" / "Depth-Anything-V2"
GEOID_GRID = ROOT / "data" / "geoids" / "es_ign_EGM08_REDNAP.tif"
FABDEM_CACHE = ROOT / "data" / "dem_cache" / "fabdem"

WEB_RUN_ROOT = ROOT / "web_runs"
WEB_RUN_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_GCP_CRS = os.getenv("DEFAULT_GCP_CRS", "EPSG:25830")

# ZeroGPU free-account quota is small, so keep a hard upper bound per call.
GPU_DURATION_SECONDS = int(os.getenv("ZEROGPU_DURATION", "300"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "250"))

# Accepted input extensions. GeoTIFF is the primary production input.
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
GCP_EXTENSIONS = {".csv", ".json"}


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Return a filesystem-safe basename."""
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "upload"


def _file_size_ok(path: Path) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"Input file is {size_mb:.1f} MB. "
            f"Maximum allowed is {MAX_UPLOAD_MB} MB."
        )


def _locate_file(scene_dir: Path, filename: str) -> Path | None:
    """Find an expected output anywhere below the scene directory."""
    direct = scene_dir / filename
    if direct.exists():
        return direct

    matches = list(scene_dir.rglob(filename))
    return matches[0] if matches else None


def _locate_extension(scene_dir: Path, extension: str) -> Path | None:
    matches = sorted(scene_dir.rglob(f"*{extension}"))
    return matches[0] if matches else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _make_preview(
    raster_path: Path,
    output_path: Path,
) -> Path | None:
    """
    Create a lightweight PNG preview from a single-band raster.

    This deliberately uses rasterio + PIL only; it does not alter the
    source GeoTIFF.
    """
    try:
        import numpy as np
        import rasterio
        from PIL import Image

        with rasterio.open(raster_path) as src:
            arr = src.read(1, masked=True)

        if arr.size == 0:
            return None

        data = np.asarray(arr.filled(np.nan), dtype=np.float32)

        finite = np.isfinite(data)
        if not finite.any():
            return None

        values = data[finite]

        # Robust display stretch.
        lo = float(np.percentile(values, 2.0))
        hi = float(np.percentile(values, 98.0))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(values))
            hi = float(np.max(values))

        scaled = (data - lo) / max(hi - lo, 1e-6)
        scaled = np.clip(scaled, 0.0, 1.0)

        # Invalid pixels become black in the preview.
        scaled[~finite] = 0.0

        image = (scaled * 255.0).astype(np.uint8)

        # Keep the web preview lightweight.
        pil = Image.fromarray(image, mode="L")
        pil.thumbnail((1400, 1400))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(output_path, format="PNG", optimize=True)

        return output_path

    except Exception:
        return None


def _progress_from_line(
    line: str,
    progress: gr.Progress,
) -> None:
    """
    Convert existing run.py stage log lines into approximate web progress.
    """
    stages = [
        ("1/7", 0.12, "Scene metadata and routing"),
        ("2/7", 0.25, "Native tiling and GCP mapping"),
        ("3/7", 0.47, "Depth Anything V2 inference"),
        ("4/7", 0.60, "FABDEM acquisition and alignment"),
        ("5/7", 0.72, "Ground estimation"),
        ("6/7", 0.86, "Calibration and DSM composition"),
        ("7/7", 0.96, "DSM export"),
    ]

    lowered = line.lower()

    for token, fraction, label in stages:
        if token.lower() in lowered:
            progress(
                fraction,
                desc=label,
            )
            return

    if "pipeline complete" in lowered or "relative pipeline complete" in lowered:
        progress(1.0, desc="Pipeline complete")


def _discover_outputs(scene_dir: Path) -> dict[str, Path | None]:
    """
    Discover outputs without hard-coding the internal directory layout too
    aggressively. The current run.py writes canonical products under compose/.
    """
    outputs: dict[str, Path | None] = {
        "dsm": _locate_file(scene_dir, "dsm.tif"),
        "ndsm": _locate_file(scene_dir, "ndsm.tif"),
        "dtm": _locate_file(scene_dir, "dtm.tif"),
        "relative_dsm": _locate_file(scene_dir, "relative_dsm.tif"),
        "buildings": _locate_file(scene_dir, "buildings.geojson"),
        "trees": _locate_file(scene_dir, "trees.geojson"),
        "calibration": _locate_file(scene_dir, "calibration.json"),
        "run_meta": _locate_file(scene_dir, "run_meta.json"),
        "glb": _locate_extension(scene_dir, ".glb"),
    }

    return outputs


def _format_summary(
    scene_dir: Path,
    outputs: dict[str, Path | None],
    gcp_used: bool,
) -> str:
    calibration = _read_json(outputs["calibration"])

    mode = calibration.get("mode", "not available")
    scale = calibration.get("a", calibration.get("scale"))
    offset = calibration.get("b", calibration.get("offset"))
    gate = calibration.get("gcp_quality_gate", {})
    gate_status = (
        gate.get("status")
        if isinstance(gate, dict)
        else "not available"
    )

    lines = [
        "## GeoSculpt result",
        "",
        f"**Scene:** `{scene_dir.name}`",
        f"**GCP supplied:** `{gcp_used}`",
        f"**Calibration mode:** `{mode}`",
    ]

    if scale is not None:
        lines.append(f"**Scale (a):** `{float(scale):.6f}`")

    if offset is not None:
        lines.append(f"**Offset (b):** `{float(offset):.6f} m`")

    if gate_status:
        lines.append(f"**GCP quality gate:** `{gate_status}`")

    lines.extend(
        [
            "",
            "### Generated products",
            f"- DSM: `{'yes' if outputs['dsm'] else 'no'}`",
            f"- nDSM: `{'yes' if outputs['ndsm'] else 'no'}`",
            f"- DTM: `{'yes' if outputs['dtm'] else 'no'}`",
            f"- Buildings GeoJSON: `{'yes' if outputs['buildings'] else 'no'}`",
            f"- Trees GeoJSON: `{'yes' if outputs['trees'] else 'no'}`",
            f"- GLB: `{'yes' if outputs['glb'] else 'no'}`",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------

@spaces.GPU(duration=GPU_DURATION_SECONDS)
def run_geosculpt(
    image_file: str | None,
    gcp_file: str | None,
    gcp_crs: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[
    str,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    """
    Run the existing GeoSculpt pipeline from a browser upload.

    The existing run.py remains the source of truth for pipeline execution.
    """
    progress(0.02, desc="Validating inputs")

    if not image_file:
        raise gr.Error("Please upload an RGB aerial image first.")

    image_src = Path(image_file)
    if not image_src.exists():
        raise gr.Error("The uploaded RGB image could not be found.")

    if image_src.suffix.lower() not in IMAGE_EXTENSIONS:
        raise gr.Error(
            "Unsupported image format. Use GeoTIFF, PNG, JPG or JPEG."
        )

    _file_size_ok(image_src)

    gcp_src: Path | None = None
    gcp_used = bool(gcp_file)

    if gcp_file:
        candidate = Path(gcp_file)
        if not candidate.exists():
            raise gr.Error("The uploaded GCP file could not be found.")

        if candidate.suffix.lower() not in GCP_EXTENSIONS:
            raise gr.Error("GCP input must be CSV or JSON.")

        _file_size_ok(candidate)
        gcp_src = candidate

        if not gcp_crs.strip():
            gcp_crs = DEFAULT_GCP_CRS

    run_id = (
        time.strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )

    work_dir = WEB_RUN_ROOT / run_id
    input_dir = work_dir / "input"
    output_root = work_dir / "outputs"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    image_path = input_dir / _safe_name(image_src.name)
    shutil.copy2(image_src, image_path)

    gcp_path: Path | None = None
    if gcp_src is not None:
        gcp_path = input_dir / _safe_name(gcp_src.name)
        shutil.copy2(gcp_src, gcp_path)

    scene_name = f"web_{run_id}"

    # -----------------------------------------------------------------
    # Build the same CLI used by the local project.
    # -----------------------------------------------------------------
    cmd = [
        sys.executable,
        str(ROOT / "run.py"),
        "--input",
        str(image_path),
        "--checkpoint",
        str(CHECKPOINT),
        "--external-model-dir",
        str(EXTERNAL_MODEL_DIR),
        "--output-dir",
        str(output_root),
        "--scene-name",
        scene_name,
        "--device",
        "cuda",
        "--batch-size",
        "1",
        "--tile-size",
        "1024",
        "--overlap",
        "0.50",
        "--train-gsd-m",
        "1.0",
        "--fabdem-cache",
        str(FABDEM_CACHE),
    ]

    if gcp_path is not None:
        cmd.extend(
            [
                "--gcps",
                str(gcp_path),
                "--gcp-crs",
                gcp_crs.strip(),
                "--gcp-z-reference",
                "ellipsoidal",
                "--gcp-geoid-grid",
                str(GEOID_GRID),
            ]
        )

    progress(0.08, desc="Starting GeoSculpt")

    if not CHECKPOINT.exists():
        raise gr.Error(
            f"Checkpoint not found at {CHECKPOINT}. "
            "Upload models/best.pth to the Space repository."
        )

    if not EXTERNAL_MODEL_DIR.exists():
        raise gr.Error(
            f"Depth Anything V2 directory not found at "
            f"{EXTERNAL_MODEL_DIR}."
        )

    if not (ROOT / "run.py").exists():
        raise gr.Error("run.py is missing from the Space repository.")

    log_lines: list[str] = []

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert process.stdout is not None

        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue

            log_lines.append(line)

            # Keep the progress UI useful without changing the pipeline.
            _progress_from_line(line, progress)

            # Expose a small amount of live diagnostic information in logs.
            print(line, flush=True)

        return_code = process.wait()

    except Exception as exc:
        raise gr.Error(f"Could not start GeoSculpt: {exc}") from exc

    if return_code != 0:
        tail = "\n".join(log_lines[-25:])
        raise gr.Error(
            "GeoSculpt failed.\n\n"
            f"Last log lines:\n{tail}"
        )

    scene_dir = output_root / scene_name

    if not scene_dir.exists():
        # Defensive fallback in case run.py changes its output nesting.
        candidates = [
            p for p in output_root.iterdir()
            if p.is_dir()
        ]
        if candidates:
            scene_dir = candidates[0]

    if not scene_dir.exists():
        raise gr.Error(
            "Pipeline reported success, but its scene output directory "
            "was not found."
        )

    progress(0.98, desc="Collecting results")

    outputs = _discover_outputs(scene_dir)

    # Generate a lightweight visual preview.
    preview: Path | None = None
    preview_source = outputs["ndsm"] or outputs["dsm"]
    if preview_source is not None:
        preview = _make_preview(
            preview_source,
            scene_dir / "web_preview.png",
        )

    summary = _format_summary(
        scene_dir=scene_dir,
        outputs=outputs,
        gcp_used=gcp_used,
    )

    progress(1.0, desc="Complete")

    # Return paths in stable UI order:
    # summary, preview, DSM, nDSM, DTM, buildings, trees, GLB
    return (
        summary,
        str(preview) if preview else None,
        str(outputs["dsm"]) if outputs["dsm"] else None,
        str(outputs["ndsm"]) if outputs["ndsm"] else None,
        str(outputs["dtm"]) if outputs["dtm"] else None,
        str(outputs["buildings"]) if outputs["buildings"] else None,
        str(outputs["trees"]) if outputs["trees"] else None,
        str(outputs["glb"]) if outputs["glb"] else None,
    )


# ---------------------------------------------------------------------
# Demo / cleanup helpers
# ---------------------------------------------------------------------

def clear_inputs() -> tuple[None, None, str]:
    return None, None, DEFAULT_GCP_CRS


def cleanup_old_runs(max_age_hours: float = 12.0) -> int:
    """
    Remove stale web-run directories.

    The Space disk is ephemeral, so this is only housekeeping while a
    container stays alive for a long time.
    """
    now = time.time()
    removed = 0

    for child in WEB_RUN_ROOT.iterdir():
        try:
            age_hours = (now - child.stat().st_mtime) / 3600.0
            if age_hours > max_age_hours:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue

    return removed


# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------

with gr.Blocks(
    title="GeoSculpt — Aerial Image to 3D",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
# 🌍 GeoSculpt
### Single RGB Aerial Image → Relative Depth → Metric DSM → 3D Scene

Upload an aerial/orthomosaic image and run the existing GeoSculpt pipeline
directly from your browser.

**Georeferenced input:** FABDEM/DTM alignment and metric calibration are
automatically used by the pipeline.  
**Optional GCP CSV:** provide `id,x,y,z` coordinates plus the GCP CRS.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.File(
                label="RGB Aerial / Orthomosaic",
                file_types=[
                    ".tif",
                    ".tiff",
                    ".png",
                    ".jpg",
                    ".jpeg",
                ],
                type="filepath",
            )

            gcp = gr.File(
                label="Optional GCP CSV / JSON",
                file_types=[".csv", ".json"],
                type="filepath",
            )

            gcp_crs = gr.Textbox(
                label="GCP CRS",
                value=DEFAULT_GCP_CRS,
                placeholder="Example: EPSG:25830",
            )

            with gr.Row():
                run_button = gr.Button(
                    "🚀 Run GeoSculpt",
                    variant="primary",
                )
                clear_button = gr.Button("Clear")

            gr.Markdown(
                """
**GCP CSV format**

```text
id,x,y,z
GCP1,....
GCP2,....
```

The existing `run.py` decides whether the scene is relative-only,
georeferenced/prior, or GCP-calibrated from the input raster and the
presence of GCPs.
"""
            )

        with gr.Column(scale=1):
            summary = gr.Markdown(
                "Upload an image and press **Run GeoSculpt**."
            )

            preview = gr.Image(
                label="Elevation Preview",
                type="filepath",
            )

    with gr.Row():
        dsm_file = gr.File(label="Download DSM")
        ndsm_file = gr.File(label="Download nDSM")
        dtm_file = gr.File(label="Download DTM")

    with gr.Row():
        buildings_file = gr.File(
            label="Download Buildings GeoJSON"
        )
        trees_file = gr.File(
            label="Download Trees GeoJSON"
        )
        glb_file = gr.Model3D(
            label="3D Scene (GLB)",
            clear_color=[0.95, 0.95, 0.95, 1.0],
        )

    gr.Markdown(
        """
### Viewer
The **GLB preview above is shown when the current pipeline has produced a
GLB reconstruction**. The final interactive fly/orbit/walk Three.js viewer
will use the same generated scene once `reconstruction.py` and the viewer
are connected in the repository.
"""
    )

    run_button.click(
        fn=run_geosculpt,
        inputs=[image, gcp, gcp_crs],
        outputs=[
            summary,
            preview,
            dsm_file,
            ndsm_file,
            dtm_file,
            buildings_file,
            trees_file,
            glb_file,
        ],
        show_progress="full",
    )

    clear_button.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[image, gcp, gcp_crs],
    )


# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------

if __name__ == "__main__":
    cleanup_old_runs()
    demo.launch()
