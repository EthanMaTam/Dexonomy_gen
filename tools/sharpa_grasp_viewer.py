#!/usr/bin/env python3
"""View successful hand grasps exported by Dexonomy."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import numpy as np
import trimesh
import viser


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "output/sharpa_tripod_demo_sharpa_left/succ_grasp"


class GraspViewer:
    def __init__(self, result_dir: Path, host: str, port: int, title: str = "Sharpa grasps") -> None:
        self.result_dir = result_dir
        self.paths = sorted(result_dir.rglob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No successful grasps in {result_dir}")
        self.server = viser.ViserServer(host=host, port=port, label=title)
        self.root = None
        self.center = np.zeros(3)

        names = [str(path.relative_to(result_dir).with_suffix("")) for path in self.paths]
        self.path_by_name = dict(zip(names, self.paths))
        self.dropdown = self.server.gui.add_dropdown("Successful grasp", names)
        self.info = self.server.gui.add_markdown("")
        self.dropdown.on_update(lambda _: self.load(self.path_by_name[self.dropdown.value]))
        self.load(self.paths[0])

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.look_at = self.center
            client.camera.position = self.center + np.asarray([0.28, -0.28, 0.20])

    def load(self, data_path: Path) -> None:
        if self.root is not None:
            self.root.remove()
        self.root = self.server.scene.add_frame("/result", show_axes=True, axes_length=0.03)

        relative = data_path.relative_to(self.result_dir)
        mesh_dir = self.result_dir.parent / "vis_3d/succ_grasp"
        stem = mesh_dir / relative.with_suffix("")
        meshes = []
        for kind, color in (("hand", (80, 155, 235)), ("obj", (210, 210, 210))):
            mesh_path = Path(f"{stem}_{kind}.obj")
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
            meshes.append(mesh)
            self.server.scene.add_mesh_simple(
                f"/result/{kind}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=color,
                side="double",
            )

        data = np.load(data_path, allow_pickle=True).item()
        contacts = data["ho_c"]
        points = np.asarray(contacts["pos"])
        normals = np.asarray(contacts["normal"])
        if len(points):
            self.server.scene.add_point_cloud(
                "/result/contacts/points",
                points=points,
                colors=(255, 70, 70),
                point_size=0.005,
                point_shape="circle",
            )
            self.server.scene.add_arrows(
                "/result/contacts/normals",
                points=np.stack([points, points + 0.02 * normals], axis=1),
                colors=(255, 70, 70),
                shaft_radius=0.0008,
                head_radius=0.0025,
                head_length=0.005,
            )
        bounds = np.vstack([mesh.bounds for mesh in meshes])
        self.center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2.0
        self.info.content = (
            f"**Result:** `{data_path.name}`  \n"
            f"**Actual contacts:** {len(points)}  \n"
            f"**Contact bodies:** `{', '.join(sorted(set(contacts['bn1'])))}`"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8046)
    parser.add_argument("--title", default="Sharpa grasps")
    args = parser.parse_args()
    GraspViewer(args.result_dir, args.host, args.port, args.title)
    print(f"Open http://{args.host}:{args.port}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
