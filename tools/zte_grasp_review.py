"""Serve a frozen ZTE review batch with category, object and scale filters."""

from __future__ import annotations

import argparse
import csv
import json
import threading
from pathlib import Path

from tools.sharpa_grasp_viewer import GraspViewer, ROOT


class ReviewViewer(GraspViewer):
    def __init__(self, batch: Path, dataset: str, host: str, port: int) -> None:
        self.dataset = dataset
        self.run = batch / dataset
        manifest = json.loads((batch / "review_manifest.json").read_text())
        summary = json.loads((batch / "review_summary.json").read_text())
        job = json.loads((self.run / "job.json").read_text())
        self.object_filter = None
        self.scale_filter = None
        with (self.run / "closed_loop_samples.csv").open() as stream:
            self.records = {
                (self.run / row["path"]).resolve(): row for row in csv.DictReader(stream)
            }
        super().__init__(
            self.run / "shards/0000/succ_grasp", host, port,
            f"ZTE 抓取审阅 | {dataset} | {batch.name}",
            self.run / "closed_loop_samples.csv",
        )
        if self.server.get_port() != port:
            self.server.stop()
            raise RuntimeError(f"Review port {port} is already in use")
        self.path_by_name["(no matching samples)"] = None
        self.template_filter.options = ["(all)", *manifest["templates"]]
        with self.server.gui.add_folder("物体与尺度筛选", expand_by_default=True):
            self.object_filter = self.server.gui.add_dropdown(
                "Object", ["(all)", *[Path(p).name for p in job["objects"]]]
            )
            self.scale_filter = self.server.gui.add_dropdown(
                "Scale", ["(all)", *["scale" + s for s in job["settings"]["scales"]]]
            )
            self.object_filter.on_update(lambda _: self._filter_paths())
            self.scale_filter.on_update(lambda _: self._filter_paths())
            reset = self.server.gui.add_button("Clear filters")
            reset.on_click(lambda _: self.clear_filters())
        values = summary["datasets"][dataset]
        with self.server.gui.add_folder("本轮生成统计", expand_by_default=False):
            self.server.gui.add_markdown(
                f"**{dataset}**：{len(job['objects'])} 个物体 × "
                f"{len(job['settings']['scales'])} 个尺度 × {len(manifest['templates'])} 类。  \n"
                f"初始候选 {values['initializations']}；有效候选 {values['refined']}；"
                f"仿真成功 {values['successful']}。  \n"
                f"有成功结果的类别：{values['categories_with_success']}/{len(manifest['templates'])}。  \n"
                "本轮无成功样本：" + ("、".join(values["missing_categories"]) or "无") + "。  \n"
                "请选择 Grasp type，再按 Object / Scale 筛选；Previous / Next 逐条检查。"
            )

    def clear_filters(self) -> None:
        self.object_filter.value = "(all)"
        self.scale_filter.value = "(all)"
        self.template_filter.value = "(all)"
        self._filter_paths()

    def _filter_paths(self) -> None:
        if self.object_filter is None or self.scale_filter is None:
            return super()._filter_paths()
        names = []
        for name, path in self.path_by_name.items():
            if path is None:
                continue
            parts = Path(name).parts
            if self.template_filter.value != "(all)" and parts[0] != self.template_filter.value:
                continue
            if self.object_filter.value != "(all)" and parts[1] != self.object_filter.value:
                continue
            if self.scale_filter.value != "(all)" and parts[3] != self.scale_filter.value:
                continue
            names.append(name)
        self.dropdown.options = names or ["(no matching samples)"]

    def load(self, path: Path | None) -> None:
        if path is None:
            with self._load_lock:
                if self.root is not None:
                    self.root.visible = False
                self.info.content = "当前类别、物体及尺度组合没有仿真成功样本。可切换筛选或点击 Clear filters。"
            return
        super().load(path)

    def _load(self, path: Path) -> None:
        super()._load(path)
        row = self.records[path.resolve()]
        self.info.content += (
            f"  \n**类别：** {row['template']}  \n**物体：** {row['object']}  \n"
            f"**尺度：** {row['scale']}  \n"
            f"**相对已验收模板的电机角度变化（平均 / 最大）：** "
            f"{float(row['mean_motor_deviation_deg']):.1f}° / "
            f"{float(row['max_motor_deviation_deg']):.1f}°"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=ROOT / "output/zte_review_20261127")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", type=int, nargs=2, default=[8062, 8063])
    args = parser.parse_args()
    batch = args.batch.resolve()
    views = [ReviewViewer(batch, dataset, args.host, port)
             for dataset, port in zip(["DGN_5k", "objaverse_5k"], args.ports)]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            state = {
                view.dataset: dict(
                    center=view.center.tolist(), selection=view.dropdown.value,
                    visible=view.root.visible, options=len(view.dropdown.options),
                    template=view.template_filter.value, object=view.object_filter.value,
                    scale=view.scale_filter.value, result_axes=view.root.show_axes,
                    world_axes=view.server.scene.world_axes.visible,
                    contacts_visible=view.show_contacts.value,
                    normals_visible=view.show_normals.value,
                    point_handle_visible=(view.contact_points.visible
                                          if view.contact_points is not None else None),
                    normal_handle_visible=(view.contact_normals.visible
                                           if view.contact_normals is not None else None),
                    clients=[dict(position=c.camera.position.tolist(), look_at=c.camera.look_at.tolist())
                             for c in view.server.get_clients().values()],
                ) for view in views
            }
            temporary = batch / "view_state.tmp"
            temporary.write_text(json.dumps(state))
            temporary.replace(batch / "view_state.json")
            stop.wait(.5)

    threading.Thread(target=monitor, daemon=True).start()
    print("REVIEW READY: " + " ".join(f"http://{args.host}:{p}" for p in args.ports), flush=True)
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for view in views:
            view.server.stop()


if __name__ == "__main__":
    main()
