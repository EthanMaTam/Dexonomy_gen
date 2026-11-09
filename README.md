# Dexonomy_gen

Sharpa Wave left-hand grasp generation and annotation tools built on
[Dexonomy](https://github.com/JYChen18/Dexonomy). This repository retains the
upstream history, research code, attribution, and license.

Local additions include the `sharpa_left` hand configuration and grasp annotations,
a browser-based Shadow/Sharpa template editor, a successful-grasp viewer, and a
time-bounded dataset runner with checkpoint/resume support and object-level
train/validation/test splits.

Datasets, generated results, Python environments, and the external Sharpa model
are excluded from Git. Prepare them separately using the instructions below.

[Project Page](https://pku-epic.github.io/Dexonomy) | [Paper](https://arxiv.org/abs/2504.18829) | [Dataset](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy/tree/main) | [Learning](https://github.com/JYChen18/DexLearn)

## Overview

<div style="text-align: center;">
    <img src="img/teaser.png" width=100% >
</div>

Our algorithm synthesizes **contact-rich, penetration-free, and physically plausible** dexterous grasps for:
- Any grasp type
- Any object
- Any articulated hand

All starting from just **one** human-annotated template *per hand and grasp type*.

## Supported Features
- Hands: Shadow, Allegro, Leap, MANO, Unitree_G1
- Object Scenes: Single (floating & tabletop), Clustered (light clutter)
- Object Assets: Rigid objects (ShapeNet & Objaverse), Articulated objects (PartNet)
- Physics Simulator: MuJoCo
- Grasping Trajectory Synthesis: cuRobo integration

TODO: add scene_cfg examples for articulated objects, clutter scenes, and other tasks.


## Installation
The upstream code was tested on **Ubuntu 20.04** with **NVIDIA RTX 3090** GPUs.
The local Pixi environment targets Linux x86-64, Python 3.10, and PyTorch 2.8.0
with CUDA 12.8; GPU generation requires a compatible NVIDIA driver.

Run all commands from the repository root. With Pixi installed:

```bash
git clone https://github.com/EthanMaTam/Dexonomy_gen.git
cd Dexonomy_gen
git submodule update --init --recursive --progress
pixi install --locked
```

The upstream submodules use GitHub SSH URLs and require SSH access. For HTTPS
access instead, configure the URLs locally before initializing them:

```bash
git config submodule.third_party/mujoco_menagerie.url https://github.com/google-deepmind/mujoco_menagerie.git
git config submodule.third_party/BODex.url https://github.com/JYChen18/BODex.git
git submodule update --init --recursive --progress
```

Alternatively, use a Conda environment and install the project in editable mode:

```bash
conda create -n dexonomy python=3.10 
conda activate dexonomy
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

Use `python` in place of `pixi run python` in the examples when using Conda.

### Sharpa model assets

Download the external model separately. The configuration and annotations use
the following model revision:

```bash
git clone https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git third_party/sharpa-urdf-usd-xml
git -C third_party/sharpa-urdf-usd-xml checkout d323b371d02f4fc04a0873c19c5234cae6d803dd
pixi run python -m tools.sharpa_template_editor --check
```

The expected model is
`third_party/sharpa-urdf-usd-xml/wave_01/left_sharpa_wave/left_sharpa_wave.xml`.
Keep the model's own license and notices with its assets.

(Optional) BODex is only used for collision-free grasping trajectory synthesis. Currently, we only provide support for UR10e + shadow hand.

```bash
cd third_party/BODex
git lfs pull
# install BODex
pip install -e . --no-build-isolation
# link to dexonomy's object folder
ln -s ../../../../../../assets/object src/curobo/content/assets/object  
cd ../..
```

## Quick Start
### 1. Prepare Object Assets 

Download our pre-processed object assets `DGN_5k_processed.zip` from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy), and organize the unzipped folders as below. 
```
assets/object/DGN_5k
|- processed_data
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- scene_cfg
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- valid_split
|  |- all.json
|  |_ ...
```
Alternatively, you can pre-process your own object assets using [MeshProcess](https://github.com/JYChen18/MeshProcess).

If the extracted dataset is stored at `dataset/DGN_5k_processed/DGN_5k`, link it
into the expected location instead of copying it:

```bash
mkdir -p assets/object
ln -s ../../dataset/DGN_5k_processed/DGN_5k assets/object/DGN_5k
```

Both `dataset/` and `assets/object/` are ignored by Git.

### 2. Sharpa template editing and dataset generation

Launch the template editor and open `http://127.0.0.1:8045`:

```bash
pixi run sharpa-template-editor
```

The editor displays the Shadow reference alongside Sharpa and saves Sharpa
annotations under `assets/hand/sharpa_left/raw_anno/`. These YAML files are
included in this repository; converted `.npy` templates are generated locally.

For a small generation run after preparing the model and object assets:

```bash
pixi run python -m tools.run_sharpa_dataset \
  --output output/sharpa_smoke --hours 1 --object-limit 1 \
  --templates 14_Tripod --points 128 --workers 1 --gpu 0
```

For a longer run using all included templates:

```bash
pixi run python -m tools.run_sharpa_dataset \
  --output output/sharpa_dataset --hours 36 --workers 4 --gpu 0
```

To resume, pass the same output directory with `--resume`. The runner preserves
the original settings and deadline; resuming does not extend the time budget.
Create a `STOP` file in the output directory or send Ctrl+C to request a stop
after the current stage. The runner stops scheduling new work near its deadline
or when free disk space drops below 20 GiB.

Each run writes `job.json`, `status.json`, `summary.csv`, `index.csv`, logs, and
per-shard results. Successful grasps live under `shards/<id>/succ_grasp/`.
Splits are assigned deterministically by object name (80/10/10 hash buckets).
Trajectory synthesis is disabled for this Sharpa workflow.

To inspect successful grasps, first export their meshes, then start the viewer:

```bash
pixi run python -m dexonomy.main op=v3d hand=sharpa_left \
  save_dir=output/sharpa_smoke/shards/0000
pixi run sharpa-grasp-viewer \
  --result-dir output/sharpa_smoke/shards/0000/succ_grasp
```

Open `http://127.0.0.1:8046`. The viewer needs both the `.npy` results and their
corresponding meshes under the shard's `vis_3d/succ_grasp/` directory.

### 3. Upstream grasp synthesis
Run the following commands to synthesize grasps for a specific grasp type:
```bash
dexrun op=tmpl  # Convert human annotations to valid templates
dexsyn 'tmpl_name=[1_Large_Diameter]' # Run the script for complete synthesis pipeline
```
where `dexrun` is the alias for `python -m dexonomy.main` and `dexsyn` is the alias for `python -m dexonomy.script`.

## Tutorial
For a detailed walkthrough of **template annotation** and **code usage**, please refer to [getting_started](https://github.com/JYChen18/Dexonomy/tree/main/getting_started).

## Upstream updates

For a fresh clone, configure the original repository as `upstream`:

```bash
git remote add upstream https://github.com/JYChen18/Dexonomy.git
git fetch upstream
```

Commit local changes before integrating updates with `git merge upstream/main`.
`origin` should point to `EthanMaTam/Dexonomy_gen`.

## License

This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

## Citation

If you find this work useful for your research, please consider citing:
```
@article{chen2025dexonomy,
        title={Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy},
        author={Chen, Jiayi and Ke, Yubin and Peng, Lin and Wang, He},
        journal={Robotics: Science and Systems},
        year={2025}
      }
```
