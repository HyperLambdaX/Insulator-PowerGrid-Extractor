"""
2D boundary extraction helper backed by an alpha shape implementation.

"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError
from shapely.geometry import LineString, LinearRing, MultiPoint, MultiPolygon, Point, Polygon

import alphashape

# Shapely 2.0 removed iteration over MultiPoint; restore it for alphashape.
if not hasattr(MultiPoint, "__iter__"):
    MultiPoint.__iter__ = lambda self: iter(self.geoms)  # type: ignore[attr-defined]


def boundary_2d(points: np.ndarray,
               shrink_factor: float = 1.0) -> np.ndarray:
    """
    ``boundary`` for a 2D point set using alpha shapes.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 2)`` array of 2D points where ``N >= 3``.
    shrink_factor : float, optional
        Controls how tightly the boundary conforms to the data (0-1 range).
        ``0`` returns the convex hull; ``1`` keeps the tightest admissible
        alpha shape. Intermediate values blend between those extremes.

    Returns
    -------
    np.ndarray
        Indices (into ``points``) describing the boundary in counter-clockwise
        order without repeating the starting vertex.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array")

    n_points = pts.shape[0]
    if n_points == 0:
        return np.array([], dtype=int)
    if n_points == 1:
        return np.array([0], dtype=int)
    if n_points == 2:
        return np.array([0, 1], dtype=int)

    shrink = float(shrink_factor)
    if np.isnan(shrink):
        shrink = 1.0
    shrink = max(min(shrink, 1.0), 0.0)

    if shrink == 0.0:
        return _convex_hull_indices(pts)

    try:
        tri = Delaunay(pts)
    except QhullError:
        return _convex_hull_indices(pts)

    radii = _triangle_circumradii(pts, tri.simplices)
    finite_radii = radii[np.isfinite(radii) & (radii > 0)]

    if finite_radii.size == 0:
        return _convex_hull_indices(pts)

    median_radius = float(np.median(finite_radii))
    if not math.isfinite(median_radius) or median_radius <= 0:
        positive = finite_radii[finite_radii > 0]
        if positive.size == 0:
            return _convex_hull_indices(pts)
        median_radius = float(np.median(positive))
        if not math.isfinite(median_radius) or median_radius <= 0:
            return _convex_hull_indices(pts)

    tight_scale = 0.96
    alpha_max = tight_scale / (median_radius + 1e-12)
    alpha = shrink * alpha_max
    alpha = max(alpha, 1e-6)

    bbox_span = pts.max(axis=0) - pts.min(axis=0)
    bbox_area = float(bbox_span[0] * bbox_span[1]) if np.all(bbox_span > 0) else 0.0

    polygon = None
    boundary_indices = np.array([], dtype=int)
    for _ in range(6):
        geometry = alphashape.alphashape(pts, alpha)
        polygon = _get_primary_polygon(geometry)
        if polygon is None or polygon.is_empty:
            alpha *= 0.7
            continue

        boundary_indices = _collect_boundary_indices(polygon, pts)
        if boundary_indices.size == 0:
            alpha *= 0.7
            continue

        if shrink > 0.5 and bbox_area > 0:
            area_ratio = polygon.area / bbox_area
            if area_ratio < 0.1:
                alpha *= 0.7
                continue
        break
    else:
        return _convex_hull_indices(pts)

    return boundary_indices


def _triangle_circumradii(points: np.ndarray,
                          simplices: np.ndarray) -> np.ndarray:
    """Compute circumradii for each triangle in the Delaunay triangulation."""
    radii: List[float] = []
    for simplex in simplices:
        triangle = points[simplex]
        a = np.linalg.norm(triangle[0] - triangle[1])
        b = np.linalg.norm(triangle[1] - triangle[2])
        c = np.linalg.norm(triangle[2] - triangle[0])
        # Twice the area via cross product magnitude
        area_twice = abs(np.cross(triangle[1] - triangle[0],
                                  triangle[2] - triangle[0]))
        if area_twice <= 0:
            radii.append(np.inf)
            continue
        radius = (a * b * c) / (2.0 * area_twice)
        radii.append(radius)
    return np.asarray(radii, dtype=float)


def _get_primary_polygon(geometry) -> Optional[Polygon]:
    """Return the primary (outermost) polygon from an alpha shape result."""
    if geometry.is_empty:
        return None

    if isinstance(geometry, Polygon):
        return geometry

    if isinstance(geometry, MultiPolygon):
        non_empty = [poly for poly in geometry.geoms if not poly.is_empty]
        if not non_empty:
            return None
        return max(non_empty, key=lambda g: g.area)

    if isinstance(geometry, MultiPoint):
        pts = np.array([point.coords[0] for point in geometry.geoms])
        if pts.shape[0] < 3:
            return None
        hull = ConvexHull(pts)
        return Polygon(pts[hull.vertices])

    return None


def _collect_boundary_indices(polygon: Polygon, points: np.ndarray) -> np.ndarray:
    """Collect boundary point indices by measuring distance to polygon exterior."""
    ring = polygon.exterior
    if isinstance(ring, LinearRing):
        boundary_line = LineString(ring.coords)
    else:
        boundary_line = LineString(ring)

    length = boundary_line.length
    if not math.isfinite(length) or length <= 0:
        return np.array([], dtype=int)

    ptp = np.ptp(points, axis=0)
    scale = float(np.max(ptp)) if np.any(ptp) else 1.0
    scale = max(scale, 1.0)
    tol = scale * 1e-5

    distances = []
    indices = []
    for idx, coord in enumerate(points):
        point = Point(coord)
        dist = boundary_line.distance(point)
        if dist <= tol:
            measure = boundary_line.project(point)
            distances.append(float(measure))
            indices.append(idx)

    if not indices:
        return np.array([], dtype=int)

    order = np.argsort(distances)
    ordered_indices = np.asarray(indices, dtype=int)[order]

    # Remove consecutive duplicates while preserving order
    unique_indices: List[int] = []
    for idx in ordered_indices:
        if not unique_indices or idx != unique_indices[-1]:
            unique_indices.append(int(idx))

    # Ensure the boundary is closed without repeating the start index at the end
    if len(unique_indices) > 1 and unique_indices[0] == unique_indices[-1]:
        unique_indices.pop()

    return np.asarray(unique_indices, dtype=int)


def _convex_hull_indices(points: np.ndarray) -> np.ndarray:
    """Fallback to the convex hull indices if alpha shape fails."""
    hull = ConvexHull(points)
    return hull.vertices
