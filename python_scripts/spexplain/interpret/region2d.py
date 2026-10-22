# 2D regions given by constraints A x <= B within the domains [0,1]^2

import numpy as np

# Feasibility tolerance in geometric units (the constraints are normalized)
TOLERANCE = 1e-7


def vertices(A, B):
    """The vertices of the region {x in [0,1]^2 : A x <= B}, ordered counter-clockwise.

    Unlike a proper polygon, the region may also be degenerate: a segment (2 vertices) or a point (1 vertex);
    an empty region has no vertices.
    The vertices are the feasible intersections of all pairs of the boundary lines (including the domains).
    """
    assert A.shape[1] == 2
    A = np.vstack([A, [[1, 0], [-1, 0], [0, 1], [0, -1]]])
    B = np.concatenate([B, [1, 0, 1, 0]])

    norms = np.linalg.norm(A, axis=1)
    A, B = A / norms[:, None], B / norms

    i, j = np.triu_indices(len(A), k=1)
    det = A[i, 0] * A[j, 1] - A[i, 1] * A[j, 0]
    # Skip parallel lines
    ok = np.abs(det) > 1e-12
    i, j, det = i[ok], j[ok], det[ok]
    pts = np.column_stack([(B[i] * A[j, 1] - B[j] * A[i, 1]) / det, (A[i, 0] * B[j] - A[j, 0] * B[i]) / det])

    pts = pts[np.all(pts @ A.T <= B + TOLERANCE, axis=1)]
    if len(pts) == 0:
        return pts

    # Merge the duplicates, e.g. of more lines intersecting at the same point
    _, unique = np.unique(np.round(pts, 9), axis=0, return_index=True)
    pts = pts[np.sort(unique)]

    centre = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))]


def farthest_pair(pts):
    """The pair of the points with the maximal distance."""
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    return pts[i], pts[j]
