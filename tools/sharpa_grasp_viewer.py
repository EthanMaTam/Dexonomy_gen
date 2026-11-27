#!/usr/bin/env python3
"""View successful hand grasps exported by Dexonomy."""

from __future__ import annotations

import argparse
import csv
import threading
from pathlib import Path

import numpy as np
import trimesh
import viser

from tools.viewer_camera import focus_camera


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "output/sharpa_tripod_demo_sharpa_left/succ_grasp"


class GraspViewer:
    def __init__(self, result_dir: Path, host: str, port: int, title: str = "Sharpa grasps",
                 validation_csv: Path | None = None) -> None:
        self.result_dir = result_dir
        self._load_lock = threading.RLock()
        self.validation = {}
        if validation_csv is not None:
            with validation_csv.open() as stream:
                self.validation = {(validation_csv.parent/row["path"]).resolve(): row
                                   for row in csv.DictReader(stream)}
        self.paths = sorted(result_dir.rglob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No successful grasps in {result_dir}")
        self.server = viser.ViserServer(host=host, port=port, label=title)
        self.server.scene.world_axes.visible = False
        self.root = None
        self.contact_points = None
        self.contact_normals = None
        self.center = np.zeros(3)
        self.bounds = np.zeros((2, 3))

        names = [str(path.relative_to(result_dir).with_suffix("")) for path in self.paths]
        self.path_by_name = dict(zip(names, self.paths))
        templates = sorted({name.split("/")[0] for name in names})
        self.template_filter = self.server.gui.add_dropdown("Grasp type", ["(all)", *templates])
        self.dropdown = self.server.gui.add_dropdown("Successful grasp", names)
        self.template_filter.on_update(lambda _: self._filter_paths())
        previous = self.server.gui.add_button("Previous grasp")
        following = self.server.gui.add_button("Next grasp")
        previous.on_click(lambda _: self._step(-1))
        following.on_click(lambda _: self._step(1))
        reset = self.server.gui.add_button("Reset view / focus object")
        reset.on_click(lambda event: self.focus(event.client, reset=True))
        with self.server.gui.add_folder("Contact display", expand_by_default=False):
            self.show_contacts = self.server.gui.add_checkbox("Show contact points", True)
            self.show_normals = self.server.gui.add_checkbox("Show contact normals", False)
            self.show_contacts.on_update(lambda _: self._update_contact_visibility())
            self.show_normals.on_update(lambda _: self._update_contact_visibility())
        self.info = self.server.gui.add_markdown("")
        self.dropdown.on_update(lambda _: self.load(self.path_by_name[self.dropdown.value]))
        self.load(self.paths[0])

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self.focus(client, reset=True)

    def focus(self, client, *, reset=False) -> None:
        if client is not None:
            focus_camera(client, self.bounds, target=self.center, reset=reset)

    def _filter_paths(self) -> None:
        selected = self.template_filter.value
        self.dropdown.options = [name for name in self.path_by_name
                                 if selected == "(all)" or name.split("/")[0] == selected]

    def _step(self, delta: int) -> None:
        options = list(self.dropdown.options)
        self.dropdown.value = options[(options.index(self.dropdown.value) + delta) % len(options)]

    def load(self, data_path: Path) -> None:
        with self._load_lock:
            self._load(data_path)

    def _update_contact_visibility(self) -> None:
        with self._load_lock:
            if self.contact_points is not None:
                self.contact_points.visible = self.show_contacts.value
            if self.contact_normals is not None:
                self.contact_normals.visible = self.show_normals.value

    def _load(self, data_path: Path) -> None:
        if self.root is not None:
            self.root.remove()
        # This is only a transform parent. Visible frame axes also create an
        # origin sphere (default radius 5 cm), which can obscure the object.
        self.root = self.server.scene.add_frame("/result", show_axes=False)
        self.contact_points = None
        self.contact_normals = None

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
            self.contact_points = self.server.scene.add_point_cloud(
                "/result/contacts/points",
                points=points,
                colors=(255, 70, 70),
                point_size=0.002,
                point_shape="circle",
                visible=self.show_contacts.value,
            )
            self.contact_normals = self.server.scene.add_arrows(
                "/result/contacts/normals",
                points=np.stack([points, points + 0.02 * normals], axis=1),
                colors=(255, 70, 70),
                shaft_radius=0.0008,
                head_radius=0.0025,
                head_length=0.005,
                visible=self.show_normals.value,
            )
        bounds = np.vstack([mesh.bounds for mesh in meshes])
        self.bounds = np.stack([bounds.min(axis=0), bounds.max(axis=0)])
        self.center = meshes[1].bounds.mean(axis=0)
        for client in self.server.get_clients().values():
            self.focus(client)
        self.info.content = (
            f"**Result:** `{data_path.name}`  \n"
            f"**Actual contacts:** {len(points)}  \n"
            f"**Contact bodies:** `{', '.join(sorted(set(contacts['bn1'])))}`"
        )
        assessment = self.validation.get(data_path.resolve())
        if assessment is not None:
            verdict = "合格" if assessment["closed_loop_accepted"] == "True" else "超出工程限值"
            self.info.content += (f"  \n**闭环检查：** {verdict}  \n"
                                  f"**最大连接误差：** {1000*float(assessment['max_connect_error_m']):.4f} mm")
            if assessment.get("max_sampled_dynamic_connect_error_m"):
                self.info.content += ("  \n**最大动态采样误差：** "
                                      f"{1000*float(assessment['max_sampled_dynamic_connect_error_m']):.4f} mm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8046)
    parser.add_argument("--title", default="Sharpa grasps")
    parser.add_argument("--validation-csv", type=Path, help="Optional per-sample closed-loop acceptance report")
    args = parser.parse_args()
    GraspViewer(args.result_dir, args.host, args.port, args.title, args.validation_csv)
    print(f"Open http://{args.host}:{args.port}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
