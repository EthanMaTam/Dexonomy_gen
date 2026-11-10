#!/usr/bin/env python3
"""Time-bounded, resumable hand dataset generation using the existing operations."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import mujoco

from dexonomy.util.file_util import load_scene_cfg, load_yaml


ROOT = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.warning("Stop requested; finishing the current shard's grasp/evaluation.")


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def split_for(object_name):
    bucket = int(hashlib.sha256(object_name.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def stage_scene(source, destination):
    data = load_scene_cfg(str(source))

    def resolve_paths(value):
        for key, item in value.items():
            if isinstance(item, dict):
                resolve_paths(item)
            elif key.endswith("_path") and isinstance(item, str):
                value[key] = str(Path(item).resolve())
                if not Path(value[key]).is_file():
                    raise FileNotFoundError(value[key])

    resolve_paths(data["scene"])
    np.save(destination, data)
    return data["scene_id"]


def summarize(output, templates):
    totals = {name: [0, 0, 0] for name in templates}
    completed = sorted((output / "shards").glob("*/summary.json"))
    index_tmp = output / "index.tmp"
    with index_tmp.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "split", "template", "scene_id"])
        for path in completed:
            report = json.loads(path.read_text())
            for name, counts in report["counts"].items():
                totals[name] = [a + b for a, b in zip(totals[name], counts)]
            with (path.parent / "index.csv").open() as shard_index:
                reader = csv.reader(shard_index)
                next(reader)
                writer.writerows(reader)
    index_tmp.replace(output / "index.csv")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["template", "initializations", "grasps", "successful", "success_percent"])
        for name, counts in totals.items():
            writer.writerow([name, *counts, round(100 * counts[2] / max(1, counts[0]), 2)])
    return {"completed_shards": len(completed), "counts": {
        label: sum(value[i] for value in totals.values())
        for i, label in enumerate(["initializations", "grasps", "successful"])
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand", default="sharpa_left", help="Hand configuration name")
    parser.add_argument("--hours", type=float, default=36)
    parser.add_argument("--objects-per-shard", type=int, default=12)
    parser.add_argument("--scales", nargs="+", default=["005", "008", "011"])
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--object-limit", type=int, default=0)
    parser.add_argument("--object-offset", type=int, default=0,
                        help="Skip objects after deterministic shuffling, for disjoint validation")
    parser.add_argument("--annotation-dir", help="Use alternative annotations without changing the hand library")
    parser.add_argument("--templates", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.hours <= 0 or min(args.objects_per_shard, args.points, args.workers) <= 0:
        parser.error("Time, shard size, sampling points and workers must be positive")
    if args.object_offset < 0 or args.object_limit < 0:
        parser.error("Object offset and limit must be nonnegative")
    os.chdir(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(output / "run.log"), logging.StreamHandler()])
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    job_path = output / "job.json"
    if job_path.exists():
        if not args.resume:
            parser.error("Existing job: use --resume to preserve its original deadline and settings")
        job = json.loads(job_path.read_text())
    else:
        raw = output / "raw_anno_snapshot"
        raw.mkdir()
        hand_cfg = load_yaml(str(ROOT / "dexonomy/config/hand" / f"{args.hand}.yaml"))
        xml_path = hand_cfg["xml_path"].replace("${asset_root}", str(ROOT / "assets"))
        model = mujoco.MjModel.from_xml_path(str(ROOT / xml_path))
        annotation_dir = (Path(args.annotation_dir).resolve() if args.annotation_dir
                          else ROOT / "assets/hand" / args.hand / "raw_anno")
        paths = sorted(annotation_dir.glob("*.yaml"))
        if args.templates:
            paths = [p for p in paths if p.stem in args.templates]
            assert {p.stem for p in paths} == set(args.templates)
        assert paths
        for path in paths:
            data = load_yaml(str(path))
            assert len(data["qpos"]) == model.nq and data.get("contact"), path
            shutil.copy2(path, raw / path.name)
        objects = sorted(p for p in (ROOT / "assets/object/DGN_5k/scene_cfg").iterdir()
                         if p.is_dir() and all((p / "floating" / f"scale{s}.npy").is_file()
                                              for s in args.scales))
        random.Random(20261107).shuffle(objects)
        objects = objects[args.object_offset:]
        if args.object_limit:
            objects = objects[:args.object_limit]
        assert objects
        job = {"started_at": time.time(), "deadline": time.time() + args.hours * 3600,
               "templates": [p.stem for p in paths], "objects": [str(p) for p in objects],
               "model_sha256": hashlib.sha256(Path(xml_path).read_bytes()).hexdigest(),
               "settings": {k: v for k, v in vars(args).items() if k not in ["output", "resume"]}}
        write_json(job_path, job)
    settings = job["settings"]
    if "model_sha256" in job:
        hand_cfg = load_yaml(str(ROOT / "dexonomy/config/hand" / f"{settings['hand']}.yaml"))
        model_path = Path(hand_cfg["xml_path"].replace("${asset_root}", str(ROOT / "assets")))
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == job["model_sha256"], "Hand model changed since job creation"
    templates = job["templates"]
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "MPLCONFIGDIR": "/tmp/dexonomy-matplotlib", "PYTHONUNBUFFERED": "1"}
    common = [f"hand={settings.get('hand', 'sharpa_left')}", f"raw_anno_dir={output}/raw_anno_snapshot",
              f"init_tmpl_dir={output}/init_tmpl", "tmpl_upd_mode=disabled", "skip_traj=True"]
    progress = summarize(output, templates)

    def status(state, **extra):
        write_json(output / "status.json", {"state": state, "updated_at": time.time(),
                   "pid": os.getpid(), "deadline": job["deadline"], **progress, **extra})

    def run(label, overrides, directory, expected):
        log_path = directory / f"{label}.console.log"
        logging.info("START %s (%s)", label, directory.name)
        status("running", stage=label, shard=str(directory.relative_to(output)))
        started = time.monotonic()
        with log_path.open("a") as stream:
            stream.write("\n--- invocation ---\n"); stream.flush()
            proc = subprocess.Popen([sys.executable, "-m", "dexonomy.main", *common, *overrides],
                                    stdout=stream, stderr=subprocess.STDOUT, env=environment,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise RuntimeError(f"{label} exceeded 30 minutes; see {log_path}")
        text = log_path.read_text()
        if code or expected not in text:
            raise RuntimeError(f"{label} failed (exit {code}); see {log_path}")
        problems = text.count("Traceback (most recent call last)") + text.count("[ERROR]")
        logging.info("END %s: %.1fs, diagnostics=%d", label, time.monotonic()-started, problems)
        return problems

    def stopping():
        reserve = min(900, settings["hours"] * 360)
        return (STOP_REQUESTED or (output / "STOP").exists()
                or time.time() >= job["deadline"] - reserve
                or shutil.disk_usage(output).free < 20 * 1024**3)

    try:
        logging.info("JOB: %d templates, %d objects, scales=%s, deadline=%s",
                     len(templates), len(job["objects"]), settings["scales"],
                     time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(job["deadline"])))
        if not (output / "converted.json").exists() and not stopping():
            warnings = run("convert", ["op=tmpl", "n_worker=1", f"save_dir={output}"],
                           output, "Finish template conversion")
            assert all((output / "init_tmpl" / f"{name}.npy").exists() for name in templates)
            write_json(output / "converted.json", {"diagnostics": warnings})
        size = settings["objects_per_shard"]
        for offset in range(0, len(job["objects"]), size):
            shard = output / "shards" / f"{offset // size:04d}"
            if (shard / "summary.json").exists():
                continue
            if stopping():
                break
            shard.mkdir(parents=True, exist_ok=True)
            scenes = shard / "scene_cfg"; scenes.mkdir(exist_ok=True)
            manifest = []
            for object_path in job["objects"][offset:offset+size]:
                obj = Path(object_path)
                for scale in settings["scales"]:
                    source = obj / "floating" / f"scale{scale}.npy"
                    target = scenes / f"{obj.name}_scale{scale}.npy"
                    try:
                        scene_id = stage_scene(source, target)
                        manifest.append({"scene_id": scene_id, "source": str(source),
                                         "object": obj.name, "split": split_for(obj.name)})
                    except (FileNotFoundError, ValueError, KeyError) as error:
                        logging.warning("Skipping scene %s: %s", source, error)
            write_json(shard / "manifest.json", manifest)
            if not manifest:
                raise RuntimeError(f"No usable scenes in {shard}")
            diagnostics = 0
            for i, name in enumerate(templates):
                marker = shard / f"init_{name}.done.json"
                if marker.exists():
                    continue
                if stopping():
                    break
                diagnostics += run(f"init_{name}", ["op=init", f"tmpl_name={name}",
                    f"save_dir={shard}", f"log_id={i}", "n_worker=1", "skip_done=True",
                    f"init_gpu=[{settings['gpu']}]", "op.device=cuda", "op.epoch=1",
                    "op.object.batch_size=3", f"op.object.n_init_point={settings['points']}",
                    "op.object.n_init_inplane=8", "op.object.n_cfg=0",
                    f"op.object.cfg_path={scenes}/*.npy", "op.filter.general.n_final=8"],
                    shard, "Finish initialization")
                write_json(marker, {"finished_at": time.time()})
            for op, expected in [("grasp", "Finish grasp generation"), ("eval", "Finish evaluation")]:
                marker = shard / f"{op}.done.json"
                input_dir = shard / ("init_data" if op == "grasp" else "grasp_data")
                if not marker.exists() and any(input_dir.rglob("*.npy")):
                    diagnostics += run(op, [f"op={op}", f"save_dir={shard}", "skip_done=True",
                                           f"n_worker={settings['workers']}"], shard, expected)
                    write_json(marker, {"finished_at": time.time()})
            counts = {name: [len(list((shard / stage / name).rglob("*.npy")))
                            for stage in ["init_data", "grasp_data", "succ_grasp"]]
                      for name in templates}
            scene_by_id = {item["scene_id"]: item for item in manifest}
            with (shard / "index.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["path", "split", "template", "scene_id"])
                for path in sorted((shard / "succ_grasp").rglob("*.npy")):
                    assert path.exists(), path
                    relative = path.relative_to(shard / "succ_grasp")
                    scene_id = str(Path(*relative.parts[1:-1]))
                    writer.writerow([str(path.relative_to(output)), scene_by_id[scene_id]["split"],
                                     relative.parts[0], scene_id])
            write_json(shard / "summary.json", {"counts": counts, "diagnostics": diagnostics})
            progress = summarize(output, templates)
            status("running", stage="shard_complete", shard=shard.name)
            logging.info("SHARD %s COMPLETE: %s", shard.name, progress)
        reason = "budget_or_stop_or_disk" if stopping() else "all_objects_processed"
        status("finished", reason=reason)
        logging.info("FINISHED: %s; %s", reason, progress)
    except BaseException as error:
        status("failed", error=str(error))
        logging.exception("Generation stopped; fix the issue and resume with the same output directory")
        raise


if __name__ == "__main__":
    main()
