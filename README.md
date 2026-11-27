# Dexonomy_gen

基于 [Dexonomy](https://github.com/JYChen18/Dexonomy) 的灵巧手抓取生成与模板编辑工具。
给定手部模型、抓取模板和物体，生成手掌位姿及关节姿态，并在 MuJoCo 中评估抓取。
本仓库保留上游研究代码、历史、署名和许可证。

按下面三个步骤使用：

1. [配置环境与手部模型](#1-配置环境与手部模型)
2. [编辑抓取模板](#2-编辑抓取模板)（也可直接使用仓库已有模板）
3. [生成抓取手势并查看结果](#3-生成抓取手势并查看结果)

[项目主页](https://pku-epic.github.io/Dexonomy) · [论文](https://arxiv.org/abs/2504.18829) · [物体数据](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy) · [DexLearn](https://github.com/JYChen18/DexLearn)

## 1. 配置环境与手部模型

### 1.1 安装环境

支持 Linux x86-64；环境固定为 Python 3.10、PyTorch 2.8.0 + CUDA 12.8。
**生成抓取需要 NVIDIA GPU 和兼容驱动**，模板编辑可在 CPU 上运行。
先安装 [Pixi](https://pixi.sh/)，确认 `pixi --version` 可用。
下面所有命令均在仓库根目录执行。

```bash
git clone https://github.com/EthanMaTam/Dexonomy_gen.git
cd Dexonomy_gen

# 编辑器使用 Shadow 参考手，先通过 HTTPS 下载网格，无需 GitHub SSH 密钥。
git config submodule.third_party/mujoco_menagerie.url https://github.com/google-deepmind/mujoco_menagerie.git
git submodule update --init --recursive third_party/mujoco_menagerie

pixi install --locked
```

检查依赖和 GPU：

```bash
pixi run python -c "import torch, mujoco, viser; print('torch:', torch.__version__); print('mujoco:', mujoco.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

生成抓取前应看到 `CUDA available: True` 且 GPU 数量至少为 1。
若为 `False`，先检查 `nvidia-smi` 能否识别显卡及驱动。
本流程使用 Python 模块入口，无需配置 `dexrun` / `dexsyn` 别名。

<details>
<summary>已有 Conda：替代 Pixi 的安装方式</summary>

先完成上面的仓库克隆和 Shadow 子模块下载，再执行：

```bash
conda create -n dexonomy python=3.10
conda activate dexonomy
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

后续命令中的 `pixi run python` 均替换为 `python`，始终保持该 Conda 环境激活。

</details>

### 1.2 选择并准备手部模型

**只需准备你要使用的手型。** 后续编辑、生成和导出时使用同一个手型名。

| 手型 | 配置名 | 模型准备 |
| --- | --- | --- |
| Linker Hand O30 右手 | `linker_o30_right` | 按下方 O30 命令下载子模块并转换 |
| Sharpa Wave 左手 | `sharpa_left` | 按下方 Sharpa 命令下载固定版本模型 |
| ZTE 右手 | `zte_hand` | 需自行提供指定版本的 `right_hand_V1.0.zip`，见 [ZTE 专用说明](https://github.com/EthanMaTam/Dexonomy_gen/blob/main/assets/hand/zte_hand/README.md)；模型包无公开下载入口 |

**O30 右手：**

```bash
git submodule update --init third_party/linkerhand-urdf
pixi run python -m tools.prepare_linker_o30
pixi run python -m tools.sharpa_template_editor --hand linker_o30_right --check
```

**Sharpa 左手：**

```bash
git clone https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git third_party/sharpa-urdf-usd-xml
git -C third_party/sharpa-urdf-usd-xml checkout d323b371d02f4fc04a0873c19c5234cae6d803dd
pixi run python -m tools.sharpa_template_editor --hand sharpa_left --check
```

保留外部模型自带的许可证和声明。检查成功时终端会输出关节数和模板数量后退出。
已有模型目录时跳过重复 `git clone`，确认其版本后运行检查即可。
**编辑模板不需要先下载物体数据。**

## 2. 编辑抓取模板

仓库已提供初始 YAML 模板；直接生成抓取时可以跳到第 3 步。
下面以 O30 为例，Sharpa 用户将 `linker_o30_right` 替换为 `sharpa_left`。

```bash
pixi run python -m tools.sharpa_template_editor --hand linker_o30_right --port 8045
```

浏览器打开 **http://127.0.0.1:8045**。左侧是 Shadow 参考手，右侧是待编辑的手。
启动时会加载同名目标模板（若存在）。在 **View** 中用 **Focus target hand** 聚焦右侧手，
用 **Focus Shadow reference** 聚焦参考手，或用 **Reset view / show both hands** 恢复双手总览。
聚焦后拖动视图会围绕所选手部中心旋转。
O30、ZTE 默认通过 **Align palm with Shadow** 对齐原生坐标中相差 90° 的掌面；
该操作只改变显示。**Load matching target template** 默认开启，切换 Shadow 类别会同步加载同名目标标注。

1. 在 **Shadow template** 或 **Load linker_o30_right template** 中选择抓取类型，默认同步另一侧的同名模板；关闭 **Load matching target template** 后可独立选择两侧模板。
2. 用各手指的关节滑块调整右侧姿态，已有接触点会随指节移动。
3. 勾选 **Contact annotation mode**，点击右侧手表面添加接触点。用 **Undo last contact** 撤销最后一点，或 **Clear contacts** 清空；法向箭头应指向表面外侧。Sharpa 默认的 **Elastomer only** 会将选点限制在弹性接触面。
4. 设置 **Output template name**，例如 `my_tripod`（仅字母、数字、下划线和连字符，不带 `.yaml`）。修改已有同名模板时，需勾选 **Allow overwrite**。
5. 点击 **Save raw annotation**，确认状态栏显示 `Saved`。保存至少两个有效接触点，并检查各指接触位置。

默认保存位置：

```text
assets/hand/linker_o30_right/raw_anno/my_tripod.yaml
assets/hand/sharpa_left/raw_anno/my_tripod.yaml
```

保存后退出编辑器或另开终端，检查模板：

```bash
pixi run python -m tools.sharpa_template_editor --hand linker_o30_right --check
pixi run python -m tools.check_hand_assets --hand linker_o30_right
```

第二条命令检查关节限位、接触点与碰撞表面的距离及法向；成功时输出 JSON 报告。
检查失败时，按报错中的 YAML 文件和指节名返回编辑器修正接触点；
Sharpa 已有模板的检查问题见 [检查说明](docs/advanced_usage.md#模板检查说明)。
ZTE 的闭环模型和专用检查按其适配说明执行。
模板保存的是关节角、接触位置与法向；**生成器会自动转换 YAML，无需手动生成 `.npy` 模板。**

若程序运行在远程服务器，用 SSH 转发端口后，在自己电脑上打开同一网址：

```bash
# 将 username@server 替换为实际用户名和服务器地址
ssh -L 8045:127.0.0.1:8045 username@server
```

运行生成器时用 `--templates my_tripod` 选择刚保存的模板，文件名不带 `.yaml`。
不传 `--templates` 会使用该手型 `raw_anno/` 下的全部模板。
每个生成任务会保存一份模板快照；编辑后要使用**新的输出目录**启动任务。

## 3. 生成抓取手势并查看结果

### 3.1 物体数据集

使用原版 Dexonomy 提供的预处理物体数据集：
从 [官方 Hugging Face 数据集](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy/tree/main)
下载 `DGN_5k_processed.tar.gz`，将解压后的 `DGN_5k` 文件夹放到
`assets/object/DGN_5k/`，使其包含 `processed_data/`、`scene_cfg/` 和 `valid_split/`。
生成器默认从 `assets/object/DGN_5k/scene_cfg/` 读取物体场景。

同一数据页还提供 `objaverse_5k_processed.tar.gz`，解压后的 `objaverse_5k` 放到
`assets/object/objaverse_5k/`；运行生成器时加 `--object-dataset objaverse_5k`。
可保留在 `dataset/` 并用符号链接接入 `assets/object/`。两套 `*_vision.tar.gz` 用于网络训练，生成抓取无需下载。

2026-11-27 起，O30、ZTE 的 33 类编号标注已直接从 Shadow 整手重建，包含非接触手指姿态。
原库及其验证报告保存在各手型的 `annotation_history/2026-11-27_before_shadow_retarget/`；
历史成功率不适用于新标注，详见 [O30 重建报告](assets/hand/linker_o30_right/retarget_report.md) 和
[ZTE 重建报告](assets/hand/zte_hand/retarget_report.md)。

物体数据不随 Git 下载；编辑模板无需准备物体数据。

### 3.2 先跑一个小任务

以 O30、已有的 `14_Tripod` 模板、1 个物体、GPU 0 为例：

```bash
pixi run python -m tools.run_sharpa_dataset --hand linker_o30_right \
  --output output/o30_smoke --hours 1 --object-limit 1 \
  --templates 14_Tripod --points 128 --workers 1 --gpu 0
```

Sharpa 用户将手型改为 `sharpa_left`，输出目录改为 `output/sharpa_smoke`。
使用自己编辑的模板时，将 `14_Tripod` 改为保存的名字，如 `my_tripod`。

生成器依次执行 **模板转换 → 物体相对手的初始化 → 抓取姿态优化 → 仿真评估**。
`--hours 1` 是调度时间预算，单个物体完成后可提前结束；小采样用于验证流程，可能没有成功抓取。
生成阶段无需打开浏览器，也无需安装 BODex。

### 3.3 批量生成

小任务跑通后，使用新的输出目录扩大采样和物体数量：

```bash
pixi run python -m tools.run_sharpa_dataset --hand linker_o30_right \
  --output output/o30_dataset --hours 36 \
  --points 1024 --workers 2 --gpu 0
```

| 参数 | 用途 |
| --- | --- |
| `--hand` | 选择手型，与编辑模板时一致 |
| `--templates 14_Tripod 24_Tip_Pinch` | 只使用这些模板；省略则使用全部模板 |
| `--object-limit 1` | 只处理一个物体；省略则处理所有符合尺度要求的物体 |
| `--points 1024` | 初始化采样点数；增加会提高开销 |
| `--workers 2` | 工作进程数；先从 1–2 开始 |
| `--gpu 0` | 初始化所用 GPU 编号 |
| `--hours 36` | 任务调度时间预算，单位小时 |
| `--output` | 任务输出目录；更换模板或配置时使用新目录 |

恢复已有任务：

```bash
pixi run python -m tools.run_sharpa_dataset --output output/o30_dataset --resume
```

恢复会读取原任务的手型、配置、模板快照和截止时间，**不会重新计时**。
按 Ctrl+C 或在输出目录创建 `STOP` 文件可请求完成当前阶段后停止。
移除 `STOP` 文件后才能继续调度；截止时间已过时需新建任务。
可用磁盘空间低于 20 GiB 时也会停止调度，请在下载数据和安装环境之外预留结果空间。

### 3.4 查看生成结果

```text
output/o30_smoke/
├── job.json                 # 固定配置、模板和物体清单
├── status.json              # 当前阶段与进度
├── summary.csv              # 按模板汇总初始化、优化及仿真成功数量
├── index.csv                # 成功抓取路径与数据划分
├── run.log
└── shards/0000/
    ├── log/                 # 各阶段日志
    ├── grasp_data/          # 优化后的候选
    └── succ_grasp/          # 通过仿真评估的抓取 .npy
```

先看 `summary.csv` 的成功数量。若有成功样本，导出网格并启动查看器：

```bash
pixi run python -m dexonomy.main op=v3d hand=linker_o30_right \
  save_dir=output/o30_smoke/shards/0000 n_worker=1 op.hand.mode=collision
pixi run python -m tools.sharpa_grasp_viewer \
  --result-dir output/o30_smoke/shards/0000/succ_grasp --port 8046
```

浏览器打开 **http://127.0.0.1:8046**，选择或切换抓取样本。
用 **Grasp type** 筛选类别，**Previous grasp / Next grasp** 逐条查看。
切换样本时自动以当前物体中心为旋转中心，并调整视距以容纳手和物体；
用 **Reset view / focus object** 恢复默认观察方向。
默认隐藏坐标轴和原点球；**Contact display** 中可开关接触点与法向箭头。
接触点使用较小红点显示，法向箭头默认关闭，便于检查手与物体的接触位置。
查看器需要 `.npy` 抓取结果及导出的 `vis_3d/succ_grasp/` 网格。
查看其他任务时同时替换手型、输出目录和分片编号。
远程查看时按编辑器的方法转发 8046 端口。

已有 ZTE 整批审阅数据时，可同时打开 DGN 与 Objaverse 的分类、物体和尺度筛选视图：

```bash
pixi run python -m tools.zte_grasp_review --batch output/zte_review_20261127
```

默认地址为 `http://127.0.0.1:8062` 和 `http://127.0.0.1:8063`。

若没有成功样本，先检查 `run.log` 和分片 `log/` 是否报错；流程完成但成功数为零时，
再检查模板接触点，或增加 `--points`、更换物体后在新目录重跑。

补充资料：[适配记录、上游操作与可选轨迹合成](docs/advanced_usage.md) ·
[O30 适配与验证](assets/hand/linker_o30_right/README.md) ·
[上游标注教程](getting_started/1_annotation_tutorial.md)。

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
