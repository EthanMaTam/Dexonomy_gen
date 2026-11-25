"""分类型复跑成功抓取的原始六方向外力评估，记录实际仿真状态的闭环误差。"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import mujoco
import numpy as np

from dexonomy.op import eval_data
from dexonomy.util.file_util import load_yaml
from tools.closed_loop_hand import connect_errors
from tools.prepare_zte_hand import ROOT, DESTINATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"output/zte_hand_validation")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--all-samples", action="store_true", help="Replay individual samples rather than one per template")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    model_hash = hashlib.sha256((DESTINATION/"right.xml").read_bytes()).hexdigest()
    assert json.loads((args.run/"job.json").read_text())["model_sha256"] == model_hash
    chosen = {}
    for p in sorted(args.run.glob("shards/*/succ_grasp/**/*.npy")):
        name = p.relative_to(next(a for a in p.parents if a.name == "succ_grasp")).parts[0]
        chosen.setdefault(str(p) if args.all_samples else name, (name, p.resolve()))
    if not chosen:
        parser.error("No successful grasps available")
    original_class = eval_data.MuJoCo_EvalEnv
    records = []
    telemetry = {}

    class MonitoredEnv(original_class):
        def step_sim(self, substep):
            super().step_sim(substep)
            mujoco.mj_kinematics(self._model, self._data)
            maximum = float(np.linalg.norm(connect_errors(self._model, self._data), axis=1).max())
            telemetry["max_connect_error_m"] = max(telemetry.get("max_connect_error_m", 0.), maximum)
            telemetry["sampled_simulation_states"] = telemetry.get("sampled_simulation_states", 0)+1

        def eval_fc(self, *args, **kwargs):
            result = super().eval_fc(*args, **kwargs)
            telemetry["physics_success"] = bool(result[0])
            return result

    eval_data.MuJoCo_EvalEnv = MonitoredEnv
    try:
        with tempfile.TemporaryDirectory(prefix="zte-replay-", dir=args.run) as temporary:
            for index, (name, path) in enumerate(list(chosen.values())[:args.count]):
                telemetry.clear()
                grasp_dir = next(a for a in path.parents if a.name == "grasp_data")
                saved_cfg = load_yaml(str(grasp_dir.parent/"log/eval_0/.hydra/config.yaml"))
                cfg = SimpleNamespace(
                    skip_traj=True, legacy_api=False, obj_mass=saved_cfg["obj_mass"], debug_render=False, debug_view=False,
                    grasp_dir=str(grasp_dir), succ_grasp_dir=str(Path(temporary)/f"{index}_{name}"/"succ_grasp"),
                    debug_dir=str(Path(temporary)/"debug"), tmpl_upd_mode="disabled",
                    log_dir=str(Path(temporary)/"log"),
                    hand=SimpleNamespace(xml_path=str(DESTINATION/"right.xml"),
                                         connect_eval_limit=saved_cfg["hand"].get("connect_eval_limit")),
                    op=SimpleNamespace(**saved_cfg["op"]),
                )
                eval_data._single_eval.__wrapped__((str(path), cfg))
                assert telemetry.get("physics_success"), (name, telemetry)
                accepted_path = Path(cfg.succ_grasp_dir)/path.relative_to(grasp_dir)
                telemetry["accepted_by_evaluator"] = accepted_path.exists()
                assert telemetry["accepted_by_evaluator"], (name, telemetry)
                records.append({"template": name, "path": str(path.relative_to(ROOT)), **telemetry})
                print(name, telemetry, flush=True)
    finally:
        eval_data.MuJoCo_EvalEnv = original_class
    result = {"model_sha256": model_hash,
              "method": ("按路径排序逐样本复跑，最多 count 个" if args.all_samples else "每类取按路径排序的首个成功抓取，最多 count 类")+"；同一六方向外力评估，在每次 step_sim 后采样连接误差",
              "count": len(records), "all_replayed_physics_successful": True,
              "max_connect_error_m": max(r["max_connect_error_m"] for r in records), "records": records}
    (args.run/"connect_replay.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
