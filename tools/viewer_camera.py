"""Frame displayed geometry and set Viser's orbit target consistently."""

import numpy as np


def focus_camera(client, bounds, *, target=None, reset=False):
    bounds = np.asarray(bounds, dtype=float)
    center = bounds.mean(axis=0) if target is None else np.asarray(target, dtype=float)
    # Enclose all geometry even when orbiting an off-center object.
    extent = np.maximum(np.abs(bounds[0] - center), np.abs(bounds[1] - center))
    radius = max(float(np.linalg.norm(extent)), 0.02)
    camera = client.camera
    direction = np.asarray([1.0, -1.0, 0.75]) if reset else camera.position - camera.look_at
    if np.linalg.norm(direction) < 1e-6:
        direction = np.asarray([1.0, -1.0, 0.75])
    half_fov = min(camera.fov / 2, np.arctan(np.tan(camera.fov / 2) * camera.aspect))
    distance = 1.15 * radius / np.sin(half_fov)
    with client.atomic():
        if reset:
            camera.up_direction = (0.0, 0.0, 1.0)
        # Viser translates look_at along with position. Set the target LAST.
        camera.position = center + direction / np.linalg.norm(direction) * distance
        camera.look_at = center
