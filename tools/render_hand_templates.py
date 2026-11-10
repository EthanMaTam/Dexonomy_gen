"""渲染手型全部标注的接触点和法向总览图，不进行动力学评估。"""

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dexonomy.util.file_util import load_yaml
from tools.sharpa_template_editor import _contact_entries


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="linker_o30_right")
    parser.add_argument("--output", type=Path, default=ROOT / "output/o30_taxonomy_verified/template_atlas.png")
    args = parser.parse_args()
    directory = ROOT / "assets/hand" / args.hand
    config = load_yaml(str(ROOT / "dexonomy/config/hand" / (args.hand+".yaml")))
    model = mujoco.MjModel.from_xml_path(str(ROOT / config["xml_path"].replace("${asset_root}", "assets")))
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = 0.6
    model.vis.headlight.diffuse[:] = 0.8
    paths = sorted((directory / "raw_anno").glob("*.yaml"), key=lambda p:
                   (int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else 1000, p.stem))
    assert paths, "没有可渲染的标注"
    keypoint_path = directory / "keypoint.yaml"
    keypoints = load_yaml(str(keypoint_path)) if keypoint_path.exists() else {}
    for name, value in keypoints.items():
        if isinstance(value, str):
            keypoints[name] = keypoints[value]
    cols = 6
    rows = math.ceil(len(paths)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3*rows), facecolor="#20252b")
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.045, 0.005, 0.105]
    camera.distance = .31
    camera.azimuth = 190
    camera.elevation = -12
    with mujoco.Renderer(model, height=300, width=320) as renderer:
        for axis, path in zip(axes.flat, paths):
            annotation = load_yaml(str(path))
            data.qpos[:] = annotation["qpos"]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            for body, values in annotation["contact"].items():
                for value in _contact_entries(values):
                    if isinstance(value, list):
                        point, normal = np.array(value).reshape(2, 3)
                    else:
                        bid = model.body(body).id
                        local = np.array(keypoints[body][int(value)]).reshape(2, 3)
                        rotation = data.xmat[bid].reshape(3, 3)
                        point = rotation @ local[0] + data.xpos[bid]
                        normal = rotation @ local[1]
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                                       np.array([.001, .001, .001]), point, np.eye(3).ravel(),
                                       np.array([1., .5, .05, 1.], dtype=np.float32))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .001,
                                        point, point + .012*normal)
                    renderer.scene.ngeom += 1
            axis.imshow(renderer.render())
            axis.set_title(path.stem.replace("_", " "), fontsize=8, color="white")
        for axis in axes.flat:
            axis.axis("off")
    fig.suptitle(f"{args.hand}: {len(paths)} annotation templates | orange: contacts and outward normals",
                 color="white", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .975))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
