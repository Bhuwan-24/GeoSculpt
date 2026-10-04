from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import rasterio
from rasterio.enums import Resampling
from scipy.spatial import cKDTree


LOGGER = logging.getLogger(
    "depthwizard.reconstruction3d"
)


# =====================================================================
# Reference-method constants
# =====================================================================

DEFAULT_Z_SCALE = 100.0

DEFAULT_DH = 0.0
DEFAULT_DW = 0.0

DEFAULT_NEIGHBOR_DISTANCE = 1.5

DEFAULT_POISSON_DEPTH = 8

DEFAULT_ORTHO_RESOLUTION = 1.0


# =====================================================================
# Basic helpers
# =====================================================================


def _safe_float(
    value: Any,
) -> float | None:

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(value):
        return None

    return value


def _normalize_0_to_zscale(
    values: np.ndarray,
    z_scale: float,
) -> np.ndarray:
    """
    Reference-method normalization.

    Equivalent to the notebook's:

        norm_0_to_zscale(point_arr, z_scale)

    Only the third coordinate / height values are changed.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    finite = np.isfinite(
        values
    )

    if not np.any(finite):

        return np.zeros_like(
            values,
            dtype=np.float32,
        )

    vmin = float(
        np.min(
            values[
                finite
            ]
        )
    )

    vmax = float(
        np.max(
            values[
                finite
            ]
        )
    )

    if vmax <= vmin:

        result = np.zeros_like(
            values,
            dtype=np.float32,
        )

        result[
            finite
        ] = 0.0

        return result

    result = (
        (
            values
            -
            np.float32(vmin)
        )
        /
        np.float32(
            vmax - vmin
        )
    ) * np.float32(
        z_scale
    )

    result[
        ~finite
    ] = 0.0

    return result.astype(
        np.float32,
        copy=False,
    )


# =====================================================================
# Raster loading
# =====================================================================


def _load_heightmap(
    path: Path,
) -> tuple[
    np.ndarray,
    rasterio.Affine,
    Any,
    float,
    float,
]:
    """
    Load GeoSculpt nDSM.

    This is the input equivalent of the reference notebook's
    `depth_nobackground` / heightmap.

    We intentionally DO NOT run the reference depth estimation,
    min-pooling, or direction pipeline because GeoSculpt already
    has a calibrated nDSM.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"nDSM not found: {path}"
        )

    with rasterio.open(
        path
    ) as src:

        heightmap = src.read(
            1
        ).astype(
            np.float32,
            copy=False,
        )

        transform = (
            src.transform
        )

        crs = src.crs

        pixel_x = math.hypot(
            transform.a,
            transform.d,
        )

        pixel_y = math.hypot(
            transform.b,
            transform.e,
        )

        nodata = src.nodata

        valid = np.isfinite(
            heightmap
        )

        if (
            nodata is not None
            and math.isfinite(
                float(nodata)
            )
        ):

            valid &= (
                heightmap
                !=
                float(nodata)
            )

        heightmap[
            ~valid
        ] = np.nan

    if (
        pixel_x <= 0.0
        or pixel_y <= 0.0
    ):

        raise ValueError(
            f"Invalid raster pixel size: "
            f"{transform}"
        )

    return (
        heightmap,
        transform,
        crs,
        pixel_x,
        pixel_y,
    )


def _load_rgb(
    path: Path,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    Load RGB imagery and align it to the heightmap.

    Output:

        H x W x 3
        float32
        [0, 255]
    """

    if not path.exists():

        raise FileNotFoundError(
            f"RGB image not found: {path}"
        )

    with rasterio.open(
        path
    ) as src:

        if src.count >= 3:

            rgb = src.read(
                [1, 2, 3],
                out_shape=(
                    3,
                    target_height,
                    target_width,
                ),
                resampling=(
                    Resampling.bilinear
                ),
            )

        else:

            band = src.read(
                1,
                out_shape=(
                    target_height,
                    target_width,
                ),
                resampling=(
                    Resampling.bilinear
                ),
            )

            rgb = np.stack(
                [
                    band,
                    band,
                    band,
                ],
                axis=0,
            )

    rgb = np.moveaxis(
        rgb,
        0,
        -1,
    ).astype(
        np.float32,
        copy=False,
    )

    return np.clip(
        rgb,
        0.0,
        255.0,
    )


# =====================================================================
# Reference "unscrew" transformation
# =====================================================================


def make_transform(
    kx: float,
    ky: float,
    bx: float = 0.0,
    by: float = 0.0,
):
    """
    Reference-method transformation.

    From the author's notebook:

        return (
            i[0] + kx*i[2] + bx,
            i[1] + ky*i[2] + by,
            i[2]
        )

    The point representation is:

        [row, column, height]
    """

    def transform(
        point: np.ndarray,
    ) -> np.ndarray:

        result = np.asarray(
            point,
            dtype=np.float32,
        ).copy()

        result[0] = (
            point[0]
            +
            np.float32(kx)
            *
            point[2]
            +
            np.float32(bx)
        )

        result[1] = (
            point[1]
            +
            np.float32(ky)
            *
            point[2]
            +
            np.float32(by)
        )

        result[2] = (
            point[2]
        )

        return result

    return transform


def transform_point_array(
    point_array: np.ndarray,
    dh: float,
    dw: float,
    max_z: float,
) -> tuple[
    np.ndarray,
    float,
    float,
]:
    """
    Reproduce the reference notebook's interactive update.

    Reference behavior:

        kx = DH / 100
        ky = DW / 100

        original height is first normalized to 0..100

        transform h and w using normalized height

        third coordinate is then normalized to Max Z

    This gives the exact conceptual behavior of the reference
    sliders without copying its depth-estimation pipeline.
    """

    original = np.asarray(
        point_array,
        dtype=np.float32,
    ).copy()

    if original.ndim != 2:
        raise ValueError(
            "Point array must be Nx3."
        )

    if original.shape[1] != 3:
        raise ValueError(
            "Point array must have exactly 3 columns."
        )

    # -------------------------------------------------------------
    # Reference:
    #
    # norm_0_to_zscale(original_points, DEFAULT_Z_SCALE)
    # -------------------------------------------------------------

    normalized = original.copy()

    normalized[
        :,
        2
    ] = _normalize_0_to_zscale(
        normalized[
            :,
            2
        ],
        DEFAULT_Z_SCALE,
    )

    # -------------------------------------------------------------
    # Reference slider conversion
    #
    # kx = DH / DEFAULT_Z_SCALE
    # ky = DW / DEFAULT_Z_SCALE
    # -------------------------------------------------------------

    kx = (
        float(dh)
        /
        DEFAULT_Z_SCALE
    )

    ky = (
        float(dw)
        /
        DEFAULT_Z_SCALE
    )

    transform_f = make_transform(
        kx=kx,
        ky=ky,
    )

    transformed = np.empty_like(
        normalized,
        dtype=np.float32,
    )

    for index in range(
        normalized.shape[0]
    ):

        transformed[
            index
        ] = transform_f(
            normalized[
                index
            ]
        )

    # -------------------------------------------------------------
    # Reference:
    #
    # norm_0_to_zscale(
    #     transformed,
    #     setpoint_z_scale
    # )
    #
    # Only Z is renormalized.
    # -------------------------------------------------------------

    transformed[
        :,
        2
    ] = _normalize_0_to_zscale(
        transformed[
            :,
            2
        ],
        float(max_z),
    )

    return (
        transformed,
        float(kx),
        float(ky),
    )


# =====================================================================
# Point generation
# =====================================================================


def heightmap_to_point_array(
    heightmap: np.ndarray,
    rgb: np.ndarray,
    valid: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Construct the reference-method point array.

    EXACT representation:

        point = (row, column, height)

    This is deliberate.

    We do NOT convert it to Three.js axes here.
    The browser viewer performs the final visual axis mapping.
    """

    rows, cols = np.nonzero(
        valid
    )

    heights = heightmap[
        rows,
        cols,
    ].astype(
        np.float32,
        copy=False,
    )

    points = np.column_stack(
        [
            rows.astype(
                np.float32
            ),
            cols.astype(
                np.float32
            ),
            heights,
        ]
    )

    colors = rgb[
        rows,
        cols,
        :
    ].astype(
        np.float32,
        copy=False,
    )

    return (
        points,
        colors,
    )


# =====================================================================
# Point-cloud filtering
# =====================================================================


def filter_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    threshold_distance: float | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Reference notebook's optional nearest-neighbor filter.

    The original notebook uses:

        nearest neighbor #2
        keep if spatial distance < 1.5

    The filter is optional because it is primarily used by the
    reference notebook's GUI point-cloud visualization.
    """

    if (
        threshold_distance is None
        or threshold_distance <= 0.0
        or points.shape[0] < 2
    ):

        return (
            points,
            colors,
        )

    LOGGER.info(
        "Filtering point cloud with "
        "nearest-neighbor threshold %.3f...",
        threshold_distance,
    )

    # scipy KD-tree gives the same geometric test much faster
    # than looping through Open3D KDTreeFlann one point at a time.

    tree = cKDTree(
        points.astype(
            np.float64,
            copy=False,
        )
    )

    distances, _ = tree.query(
        points.astype(
            np.float64,
            copy=False,
        ),
        k=2,
    )

    nearest_distance = (
        distances[
            :,
            1
        ]
    )

    keep = (
        np.isfinite(
            nearest_distance
        )
        &
        (
            nearest_distance
            <
            float(
                threshold_distance
            )
        )
    )

    filtered_points = (
        points[
            keep
        ]
    )

    filtered_colors = (
        colors[
            keep
        ]
    )

    LOGGER.info(
        "Point cloud filtering: %d -> %d",
        len(points),
        len(filtered_points),
    )

    if len(filtered_points) == 0:

        LOGGER.warning(
            "Filter removed all points; "
            "using the unfiltered point cloud."
        )

        return (
            points,
            colors,
        )

    return (
        filtered_points,
        filtered_colors,
    )


# =====================================================================
# Open3D helpers
# =====================================================================


def make_open3d_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    Create Open3D cloud in the SAME coordinate convention
    as the reference method:

        X = image row
        Y = image column
        Z = height

    Open3D itself uses Z as the conventional up axis,
    which matches the reference notebook's PLY representation.
    """

    cloud = (
        o3d.geometry.PointCloud()
    )

    cloud.points = (
        o3d.utility.Vector3dVector(
            points.astype(
                np.float64,
                copy=False,
            )
        )
    )

    cloud.colors = (
        o3d.utility.Vector3dVector(
            (
                colors
                /
                255.0
            ).astype(
                np.float64,
                copy=False,
            )
        )
    )

    return cloud


def save_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    path: Path,
) -> None:

    cloud = (
        make_open3d_point_cloud(
            points,
            colors,
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = (
        o3d.io.write_point_cloud(
            str(path),
            cloud,
        )
    )

    if not success:

        raise RuntimeError(
            f"Could not write point cloud: {path}"
        )


# =====================================================================
# Poisson mesh — same construction as reference
# =====================================================================


def build_poisson_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    poisson_depth: int,
) -> o3d.geometry.TriangleMesh:
    """
    Reproduce the reference notebook's mesh construction:

        PointCloud
        -> estimate_normals(radius=1, max_nn=30)
        -> orient_normals_consistent_tangent_plane(16)
        -> Poisson(depth=8)
        -> crop to point-cloud bbox
        -> transfer nearest-point colors

    This is intentionally the reference method rather than the
    continuous raster mesh introduced previously.
    """

    cloud = (
        make_open3d_point_cloud(
            points,
            colors,
        )
    )

    LOGGER.info(
        "Estimating normals "
        "(radius=1.0, max_nn=30)..."
    )

    cloud.estimate_normals(
        search_param=(
            o3d.geometry
            .KDTreeSearchParamHybrid(
                radius=1.0,
                max_nn=30,
            )
        )
    )

    LOGGER.info(
        "Orienting normals "
        "(tangent-plane=16)..."
    )

    cloud.orient_normals_consistent_tangent_plane(
        16
    )

    LOGGER.info(
        "Running Poisson reconstruction "
        "(depth=%d)...",
        poisson_depth,
    )

    mesh, _ = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            cloud,
            depth=int(
                poisson_depth
            ),
        )
    )

    LOGGER.info(
        "Cropping Poisson mesh to point-cloud bounds..."
    )

    bbox = (
        cloud
        .get_axis_aligned_bounding_box()
    )

    mesh = mesh.crop(
        bbox
    )

    LOGGER.info(
        "Transferring RGB colors..."
    )

    mesh_vertices = np.asarray(
        mesh.vertices
    )

    cloud_points = np.asarray(
        cloud.points
    )

    cloud_colors = np.asarray(
        cloud.colors
    )

    if (
        len(mesh_vertices) > 0
        and len(cloud_points) > 0
    ):

        tree = cKDTree(
            cloud_points
        )

        _, nearest = tree.query(
            mesh_vertices,
            k=1,
        )

        mesh.vertex_colors = (
            o3d.utility.Vector3dVector(
                cloud_colors[
                    nearest
                ]
            )
        )

    return mesh


def save_mesh(
    mesh: o3d.geometry.TriangleMesh,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = (
        o3d.io.write_triangle_mesh(
            str(path),
            mesh,
            write_vertex_colors=True,
        )
    )

    if not success:

        raise RuntimeError(
            f"Could not write mesh: {path}"
        )


# =====================================================================
# Reference-style orthographic projection
# =====================================================================


def ortho_from_pointcloud(
    points: np.ndarray,
    height: int,
    width: int,
    resolution: float = 1.0,
    colors: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reproduce the reference notebook's orthographic z-buffer.

    In the reference coordinate convention:

        X = row-like coordinate
        Y = column-like coordinate
        Z = height

    For each X/Y output cell, keep the highest Z.
    """

    x = points[
        :,
        0
    ]

    y = points[
        :,
        1
    ]

    z = points[
        :,
        2
    ]

    x_min = float(
        np.min(x)
    )

    y_min = float(
        np.min(y)
    )

    i = (
        (
            x
            -
            np.float32(
                x_min
            )
        )
        /
        np.float32(
            resolution
        )
    ).astype(
        np.int32
    )

    j = (
        (
            y
            -
            np.float32(
                y_min
            )
        )
        /
        np.float32(
            resolution
        )
    ).astype(
        np.int32
    )

    if colors is None:

        image = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.float32,
        )

    else:

        image = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

    depth = np.full(
        (
            height,
            width,
        ),
        -np.inf,
        dtype=np.float32,
    )

    for index in range(
        len(points)
    ):

        jj = i[index]
        ii = j[index]

        if (
            jj < 0
            or jj >= height
            or ii < 0
            or ii >= width
        ):
            continue

        if (
            z[index]
            >
            depth[
                jj,
                ii
            ]
        ):

            depth[
                jj,
                ii
            ] = z[index]

            if colors is None:

                image[
                    jj,
                    ii
                ] = z[index]

            else:

                image[
                    jj,
                    ii
                ] = np.clip(
                    colors[index],
                    0.0,
                    255.0,
                ).astype(
                    np.uint8
                )

    return image


def save_preview_png(
    path: Path,
    image: np.ndarray,
) -> None:
    """
    Save RGB orthographic preview using Pillow.
    """

    try:

        from PIL import Image

    except ImportError:

        LOGGER.warning(
            "Pillow is not installed; "
            "skipping PNG preview."
        )

        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if image.ndim != 3:

        normalized = (
            _normalize_0_to_zscale(
                image,
                255.0,
            )
            .astype(
                np.uint8
            )
        )

        rgb = np.stack(
            [
                normalized,
                normalized,
                normalized,
            ],
            axis=-1,
        )

    else:

        rgb = image.astype(
            np.uint8,
            copy=False,
        )

    Image.fromarray(
        rgb,
        mode="RGB",
    ).save(
        path
    )


# =====================================================================
# Reconstruction
# =====================================================================


def reconstruct(
    ndsm_path: Path,
    rgb_path: Path,
    output_dir: Path,
    dh: float,
    dw: float,
    max_z: float | None,
    z_normalization_scale: float,
    neighbor_distance: float | None,
    poisson_depth: int,
    build_mesh_flag: bool,
    visualize_cloud: bool,
    visualize_mesh: bool,
    write_ortho: bool,
) -> dict[str, Any]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load nDSM
    # -------------------------------------------------------------

    LOGGER.info(
        "Loading GeoSculpt nDSM: %s",
        ndsm_path,
    )

    (
        heightmap,
        transform,
        crs,
        pixel_x,
        pixel_y,
    ) = _load_heightmap(
        ndsm_path
    )

    height_px, width_px = (
        heightmap.shape
    )

    LOGGER.info(
        "Heightmap grid: %dx%d",
        width_px,
        height_px,
    )

    LOGGER.info(
        "Pixel size: %.6f x %.6f m",
        pixel_x,
        pixel_y,
    )

    # -------------------------------------------------------------
    # RGB
    # -------------------------------------------------------------

    LOGGER.info(
        "Loading RGB: %s",
        rgb_path,
    )

    rgb = _load_rgb(
        rgb_path,
        height_px,
        width_px,
    )

    # -------------------------------------------------------------
    # Valid pixels
    # -------------------------------------------------------------

    valid = np.isfinite(
        heightmap
    )

    valid_count = int(
        np.count_nonzero(
            valid
        )
    )

    if valid_count == 0:

        raise ValueError(
            "No valid heightmap pixels."
        )

    LOGGER.info(
        "Valid height pixels: %d / %d",
        valid_count,
        valid.size,
    )

    # -------------------------------------------------------------
    # Reference point representation
    #
    # EXACTLY:
    #
    # point = (h, w, z)
    #
    # h = raster row
    # w = raster column
    # z = GeoSculpt nDSM height
    # -------------------------------------------------------------

    LOGGER.info(
        "Constructing reference point array..."
    )

    (
        original_points,
        original_colors,
    ) = heightmap_to_point_array(
        heightmap=heightmap,
        rgb=rgb,
        valid=valid,
    )

    LOGGER.info(
        "Original points: %d",
        len(original_points),
    )

    # -------------------------------------------------------------
    # Max-Z behavior
    #
    # The reference notebook uses Max Z as a user-controlled
    # vertical scale.
    #
    # For GeoSculpt we default to the actual nDSM maximum so the
    # model remains approximately metric while retaining exactly
    # the same normalization procedure.
    # -------------------------------------------------------------

    nDSM_values = (
        original_points[
            :,
            2
        ]
    )

    actual_min_z = float(
        np.min(
            nDSM_values
        )
    )

    actual_max_z = float(
        np.max(
            nDSM_values
        )
    )

    if max_z is None:

        max_z_value = max(
            actual_max_z
            -
            actual_min_z,
            1e-6,
        )

        max_z_source = (
            "automatic: nDSM range"
        )

    else:

        max_z_value = float(
            max_z
        )

        if max_z_value <= 0.0:

            raise ValueError(
                "--max-z must be > 0."
            )

        max_z_source = (
            "user supplied"
        )

    # -------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------

    LOGGER.info(
        "Applying reference-style 3D "
        "unscrew transformation..."
    )

    (
        transformed_points,
        kx,
        ky,
    ) = transform_point_array(
        point_array=original_points,
        dh=dh,
        dw=dw,
        max_z=max_z_value,
    )

    LOGGER.info(
        "DH = %.3f",
        dh,
    )

    LOGGER.info(
        "DW = %.3f",
        dw,
    )

    LOGGER.info(
        "kx = %.6f",
        kx,
    )

    LOGGER.info(
        "ky = %.6f",
        ky,
    )

    LOGGER.info(
        "Max Z = %.6f",
        max_z_value,
    )

    # -------------------------------------------------------------
    # Optional nearest-neighbor filter
    #
    # This corresponds to the notebook's visualization helper.
    # The transformed cloud remains the canonical reconstruction.
    # -------------------------------------------------------------

    (
        filtered_points,
        filtered_colors,
    ) = filter_point_cloud(
        points=transformed_points,
        colors=original_colors,
        threshold_distance=neighbor_distance,
    )

    # -------------------------------------------------------------
    # Save arrays
    # -------------------------------------------------------------

    transformed_points_path = (
        output_dir
        /
        "transformed_point_array.npy"
    )

    color_array_path = (
        output_dir
        /
        "color_array.npy"
    )

    np.save(
        transformed_points_path,
        filtered_points,
    )

    np.save(
        color_array_path,
        filtered_colors,
    )

    # -------------------------------------------------------------
    # Save PLY
    # -------------------------------------------------------------

    point_cloud_path = (
        output_dir
        /
        "point_cloud.ply"
    )

    save_point_cloud(
        points=filtered_points,
        colors=filtered_colors,
        path=point_cloud_path,
    )

    # -------------------------------------------------------------
    # Orthographic projection
    # -------------------------------------------------------------

    ortho_height_path = None
    ortho_color_path = None
    ortho_height_npy_path = None
    ortho_color_npy_path = None

    if write_ortho:

        LOGGER.info(
            "Generating reference-style "
            "orthographic projection..."
        )

        ortho_height = (
            ortho_from_pointcloud(
                points=filtered_points,
                height=height_px,
                width=width_px,
                resolution=(
                    DEFAULT_ORTHO_RESOLUTION
                ),
                colors=None,
            )
        )

        ortho_color = (
            ortho_from_pointcloud(
                points=filtered_points,
                height=height_px,
                width=width_px,
                resolution=(
                    DEFAULT_ORTHO_RESOLUTION
                ),
                colors=filtered_colors,
            )
        )

        ortho_height_npy_path = (
            output_dir
            /
            "orthographic_height.npy"
        )

        ortho_color_npy_path = (
            output_dir
            /
            "orthographic_color.npy"
        )

        np.save(
            ortho_height_npy_path,
            ortho_height,
        )

        np.save(
            ortho_color_npy_path,
            ortho_color,
        )

        ortho_height_path = (
            output_dir
            /
            "orthographic_height.png"
        )

        ortho_color_path = (
            output_dir
            /
            "orthographic_color.png"
        )

        save_preview_png(
            ortho_height_path,
            ortho_height,
        )

        save_preview_png(
            ortho_color_path,
            ortho_color,
        )

    # -------------------------------------------------------------
    # Optional Poisson mesh
    # -------------------------------------------------------------

    mesh_path = None

    mesh_vertices = 0
    mesh_triangles = 0

    mesh = None

    if build_mesh_flag:

        LOGGER.info(
            "Starting reference-style "
            "Poisson mesh generation..."
        )

        mesh = build_poisson_mesh(
            points=filtered_points,
            colors=filtered_colors,
            poisson_depth=poisson_depth,
        )

        mesh_path = (
            output_dir
            /
            "surface_mesh.ply"
        )

        save_mesh(
            mesh=mesh,
            path=mesh_path,
        )

        mesh_vertices = len(
            mesh.vertices
        )

        mesh_triangles = len(
            mesh.triangles
        )

        LOGGER.info(
            "Mesh vertices: %d",
            mesh_vertices,
        )

        LOGGER.info(
            "Mesh triangles: %d",
            mesh_triangles,
        )

    # -------------------------------------------------------------
    # Open3D visualization
    # -------------------------------------------------------------

    if visualize_cloud:

        LOGGER.info(
            "Opening Open3D point-cloud viewer..."
        )

        cloud = make_open3d_point_cloud(
            filtered_points,
            filtered_colors,
        )

        o3d.visualization.draw_geometries(
            [cloud],
            window_name=(
                "GeoSculpt - Reference "
                "3D Point Cloud"
            ),
            width=1400,
            height=900,
        )

    if visualize_mesh:

        if mesh is None:

            raise ValueError(
                "--visualize-mesh requires --mesh."
            )

        LOGGER.info(
            "Opening Open3D mesh viewer..."
        )

        o3d.visualization.draw_geometries(
            [mesh],
            window_name=(
                "GeoSculpt - Reference "
                "3D Poisson Mesh"
            ),
            width=1400,
            height=900,
        )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    metadata = {

        "stage":
            "3D_RECONSTRUCTION",

        "reference_method":
            "aliaksandr960/maps_screenshot_to_3d",

        "method": {

            "description":
                (
                    "GeoSculpt nDSM used as the reference "
                    "heightmap, followed by the author's "
                    "point representation, z normalization, "
                    "DH/DW height-dependent horizontal "
                    "unscrew, Open3D point cloud and optional "
                    "Poisson mesh."
                ),

            "point_representation":
                "[row, column, height]",

            "transform":
                (
                    "row' = row + kx * z; "
                    "column' = column + ky * z; "
                    "z' = normalized Max-Z"
                ),

            "kx":
                float(kx),

            "ky":
                float(ky),

            "dh":
                float(dh),

            "dw":
                float(dw),

            "z_normalization_before_transform":
                float(z_normalization_scale),

            "max_z":
                float(max_z_value),

            "max_z_source":
                max_z_source,

            "poisson_depth":
                (
                    int(poisson_depth)
                    if build_mesh_flag
                    else None
                ),

        },

        "inputs": {

            "ndsm":
                str(ndsm_path),

            "rgb":
                str(rgb_path),

        },

        "grid": {

            "width":
                int(width_px),

            "height":
                int(height_px),

            "pixel_size_x_m":
                float(pixel_x),

            "pixel_size_y_m":
                float(pixel_y),

            "crs":
                (
                    str(crs)
                    if crs is not None
                    else None
                ),

        },

        "height_statistics": {

            "input_nDSM_min_m":
                _safe_float(
                    actual_min_z
                ),

            "input_nDSM_max_m":
                _safe_float(
                    actual_max_z
                ),

            "output_Z_min":
                _safe_float(
                    np.min(
                        filtered_points[
                            :,
                            2
                        ]
                    )
                ),

            "output_Z_max":
                _safe_float(
                    np.max(
                        filtered_points[
                            :,
                            2
                        ]
                    )
                ),

        },

        "statistics": {

            "valid_pixels":
                int(valid_count),

            "original_points":
                int(
                    len(
                        original_points
                    )
                ),

            "transformed_points":
                int(
                    len(
                        transformed_points
                    )
                ),

            "filtered_points":
                int(
                    len(
                        filtered_points
                    )
                ),

            "mesh_vertices":
                int(mesh_vertices),

            "mesh_triangles":
                int(mesh_triangles),

        },

        "outputs": {

            "transformed_point_array":
                str(
                    transformed_points_path
                ),

            "color_array":
                str(
                    color_array_path
                ),

            "point_cloud":
                str(
                    point_cloud_path
                ),

            "surface_mesh":
                (
                    str(mesh_path)
                    if mesh_path is not None
                    else None
                ),

            "orthographic_height":
                (
                    str(
                        ortho_height_npy_path
                    )
                    if ortho_height_npy_path
                    is not None
                    else None
                ),

            "orthographic_color":
                (
                    str(
                        ortho_color_npy_path
                    )
                    if ortho_color_npy_path
                    is not None
                    else None
                ),

            "orthographic_height_preview":
                (
                    str(
                        ortho_height_path
                    )
                    if ortho_height_path
                    is not None
                    else None
                ),

            "orthographic_color_preview":
                (
                    str(
                        ortho_color_path
                    )
                    if ortho_color_path
                    is not None
                    else None
                ),

        },

        "viewer_coordinate_mapping": {

            "reference_x":
                "raster row",

            "reference_y":
                "raster column",

            "reference_z":
                "height",

            "threejs_x":
                "reference Y",

            "threejs_y":
                "reference Z",

            "threejs_z":
                "negative reference X",

            "note":
                (
                    "The PLY is intentionally written in the "
                    "reference method's [row, column, height] "
                    "coordinate convention. The browser viewer "
                    "must map it to Three.js X/Y/Z rather than "
                    "rotating the reconstruction arbitrarily."
                ),

        },

    }

    metadata_path = (
        output_dir
        /
        "reconstruction3d_meta.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Final logging
    # -------------------------------------------------------------

    LOGGER.info(
        "=========================================================="
    )

    LOGGER.info(
        "GeoSculpt 3D reconstruction complete"
    )

    LOGGER.info(
        "Point cloud: %s",
        point_cloud_path,
    )

    if mesh_path is not None:

        LOGGER.info(
            "Surface mesh: %s",
            mesh_path,
        )

    LOGGER.info(
        "Metadata: %s",
        metadata_path,
    )

    LOGGER.info(
        "=========================================================="
    )

    return metadata


# =====================================================================
# CLI
# =====================================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(

        description=(
            "GeoSculpt 3D construction using "
            "the Aliaksandr reference method: "
            "heightmap -> [row,column,z] -> "
            "DH/DW unscrew -> Open3D point cloud "
            "-> optional Poisson mesh."
        )

    )

    parser.add_argument(
        "--ndsm",
        required=True,
        help=(
            "GeoSculpt nDSM GeoTIFF."
        ),
    )

    parser.add_argument(
        "--rgb",
        required=True,
        help=(
            "RGB orthomosaic/image."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "3D reconstruction output directory."
        ),
    )

    parser.add_argument(
        "--dh",
        type=float,
        default=DEFAULT_DH,
        help=(
            "Reference-method DH slider value."
        ),
    )

    parser.add_argument(
        "--dw",
        type=float,
        default=DEFAULT_DW,
        help=(
            "Reference-method DW slider value."
        ),
    )

    parser.add_argument(
        "--max-z",
        type=float,
        default=None,
        help=(
            "Reference-method Max Z. "
            "Defaults to the actual nDSM height range."
        ),
    )

    parser.add_argument(
        "--z-normalization-scale",
        type=float,
        default=DEFAULT_Z_SCALE,
        help=(
            "Initial Z normalization used before DH/DW "
            "transform. The reference notebook uses 100."
        ),
    )

    parser.add_argument(
        "--neighbor-distance",
        type=float,
        default=None,
        help=(
            "Optional reference-style nearest-neighbor "
            "filter threshold. Use 1.5 to reproduce the "
            "notebook's default GUI filter. Disabled by default."
        ),
    )

    parser.add_argument(
        "--mesh",
        action="store_true",
        help=(
            "Generate the reference-style Poisson mesh."
        ),
    )

    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=DEFAULT_POISSON_DEPTH,
        help=(
            "Poisson reconstruction depth."
        ),
    )

    parser.add_argument(
        "--visualize-cloud",
        action="store_true",
        help=(
            "Open the transformed point cloud in Open3D."
        ),
    )

    parser.add_argument(
        "--visualize-mesh",
        action="store_true",
        help=(
            "Open the Poisson mesh in Open3D."
        ),
    )

    parser.add_argument(
        "--no-ortho",
        action="store_true",
        help=(
            "Do not create the reference-style "
            "orthographic preview."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser


# =====================================================================
# Main
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

        reconstruct(

            ndsm_path=Path(
                args.ndsm
            ),

            rgb_path=Path(
                args.rgb
            ),

            output_dir=Path(
                args.output_dir
            ),

            dh=float(
                args.dh
            ),

            dw=float(
                args.dw
            ),

            max_z=(
                float(
                    args.max_z
                )
                if args.max_z is not None
                else None
            ),

            z_normalization_scale=float(
                args.z_normalization_scale
            ),

            neighbor_distance=(
                float(
                    args.neighbor_distance
                )
                if args.neighbor_distance
                is not None
                else None
            ),

            poisson_depth=int(
                args.poisson_depth
            ),

            build_mesh_flag=bool(
                args.mesh
            ),

            visualize_cloud=bool(
                args.visualize_cloud
            ),

            visualize_mesh=bool(
                args.visualize_mesh
            ),

            write_ortho=(
                not args.no_ortho
            ),

        )

        return 0

    except KeyboardInterrupt:

        LOGGER.warning(
            "Reconstruction interrupted."
        )

        return 130

    except Exception as exc:

        LOGGER.exception(
            "3D reconstruction failed: %s",
            exc,
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )