# Linker Hand O30 右手适配说明

官方模型文件名为 **O30i right / 0803**，来源为
[厂商模型仓库](https://github.com/linker-bot/linkerhand-urdf)，固定提交：
`49f26021b0a91ff5c33150df22970c07bb548672`。
原始 URDF 和网格保存在 `third_party/linkerhand-urdf/`，以 Git 子模块锁定以上版本；
主仓库提交子模块引用、转换代码和标注，网格内容仍保留在厂商仓库。
厂商模型采用 Apache-2.0，使用时保留原有许可证和声明。

在项目根目录获取模型并检查：

```bash
git submodule update --init third_party/linkerhand-urdf
git -C third_party/linkerhand-urdf rev-parse HEAD
pixi run python -m tools.linker_o30_preview --check
```

转换器检查固定提交、源仓库无修改、URDF 右手名称和 20 个活动关节；
子模块未初始化时会给出获取命令。更新模型版本时须同步调整转换器的
`EXPECTED_REVISION`，重新转换并完成运动学、接触和抓取验证。
不应使用 `git submodule update --remote` 自动跟随厂商最新提交。

依赖管理对比：Shadow 的外部网格来自 `mujoco_menagerie` 子模块；
BODex 也由子模块管理；Sharpa 目前仍采用被 Git 忽略的独立克隆和手动固定版本。
O30 厂商仓库含其他型号，模型工作区约 467 MB（另有 Git 历史），其中本次右手目录约 18 MB；
子模块能锁定来源和版本，但首次初始化仍会获取整个厂商仓库。

## 模型转换与仿真假设

`right.xml` 是生成的 MJCF，包含 20 个独立位置执行器，保留源模型的几何、
运动学变换、关节限位及 CAD 惯量。当前参数**尚未经过实物标定**：位置增益 5，
力矩限制 ±1 Nm，阻尼 0.05，电枢惯量 0.0002，关节摩擦损耗 0.01。
源模型的 `ring_mcp_roll` 力矩上限为零，转换时按其他关节设置为 ±1 Nm 的仿真假设。
完整转换记录见 `provenance.json`。

源模型使用完整 CAD 网格作为碰撞几何，其凸包在部分装配关节处重叠。
参照已有 Shadow 模型，排除了记录中列出的 8 对装配邻接部件接触；这会改变这些
连接处的碰撞建模。手指节间自碰撞及手与物体的碰撞检查仍然启用，
没有放宽全局抓取或评估阈值。

## 标注文件在哪里

原始抓取标注位于本目录的 `raw_anno/`，现已覆盖 Shadow 库的 33 类编号抓取，
另保留 `fingertip_large`，共 34 份。最初保留的 3 份如下：

| 文件 | 初始抓取类型 | 标注的手指 |
| --- | --- | --- |
| `raw_anno/24_Tip_Pinch.yaml` | 指尖捏取 | 拇指、食指 |
| `raw_anno/14_Tripod.yaml` | 三指抓取 | 拇指、食指、中指 |
| `raw_anno/fingertip_large.yaml` | 较大三指抓取 | 拇指、食指、中指 |

上表 3 份是最初生成并验证的种子模板，在首次全库验证时没有修改；其中 14、24 类随后完成了独立验证优化，见文末。其余 31 类由 Shadow 标注
自动迁移而来。全部属于程序辅助生成的初始标注，分类名称沿用源库；
几何和物理验证不能替代人工对分类语义的确认。

全部类别的接触数量与逐类评估结果见 [全分类验证表](taxonomy_validation.md)，
可下载的机器可读表为 `taxonomy_validation.csv`。迁移来源、哈希与拟合误差记录在
`taxonomy_provenance.json`。

## 一份标注包含什么

每份 YAML 主要包含两项：

```yaml
qpos: [...]  # 20 个关节角，单位为弧度
contact:
  thumb_distal: [x, y, z, nx, ny, nz]
  index_distal: [x, y, z, nx, ny, nz]
```

- `qpos`：初始手势。顺序与 `right.xml` 中关节顺序一致：拇指 4 个、食指 4 个、
  中指 4 个、无名指 4 个、小指 4 个，共 20 个，不包含手掌的空间位姿。
- `contact` 的键：接触点所在的手部连杆名，例如 `thumb_distal` 是拇指末节。
- 前三项 `[x, y, z]`：在该 `qpos` 姿态下，接触点在手基座坐标系中的位置，单位为米。
  不是物体坐标，也不是该指节的局部坐标。
- 后三项 `[nx, ny, nz]`：同一坐标系下的单位外法向，表示手部表面朝外的方向。
  生成时用它与物体表面的相反法向匹配。

当前 YAML 直接保存六维坐标。格式也支持以整数引用 `keypoint.yaml` 中的关键点，
但当前 O30 标注没有使用整数引用。`keypoint.yaml` 的关键点位于各自连杆的局部
坐标系，与 `raw_anno/` 中直接保存的坐标系不同。

**不要仅修改 YAML 的 `qpos` 而保留原接触坐标。** 改变手势后，原接触点一般就不再
位于手指表面。编辑器在内部将接触点绑定到连杆局部坐标，调整关节后会让接触点随
连杆移动，保存时再转换为上述手基座坐标，因此推荐通过编辑器调整标注。

## 最初三份模板如何生成

1. 在 O30 自身碰撞网格的指腹上，通过射线求交选取接触点和表面外法向。
2. 为捏取、三指抓取设置不同半径的虚拟球。
3. 通过带关节限位和自碰撞惩罚的逆运动学调整姿态，让指腹接触点靠近虚拟球表面，
   法向朝向球心。
4. 将求出的关节角、接触点和法向写入 `raw_anno/*.yaml`。
5. 用 `op=tmpl` 将 YAML 转成运行用的 `init_tmpl/*.npy`。
6. 对实际物体运行 `init → grasp → eval`，优化并筛选成功抓取。

虚拟球只是初始化工具，不是生成结果中的实际物体，也不保证模板可直接抓住一个球。
拟合误差保存在 `template_provenance.json`；最终有效性由抓取细化和物理仿真判断。
`body_group.yaml` 定义按手指分组的接触要求；`skeleton.yaml` 用网格内部线段近似手部，
供初始化阶段粗略碰撞筛选。

## 全部 33 类标注如何补齐

以 `assets/hand/shadow/raw_anno/` 中的 33 类编号模板作为参考，保留抓取类型、
接触指节和严格接触分组；通过解剖连杆对应关系，将接触点投影到 O30 的实际碰撞凸包。
再用有界姿态拟合及自碰撞惩罚调整关节角，最后重新计算手基座坐标下的接触位置和外法向。
O30 与 Shadow 的自由度不同，尤其小指掌骨、拇指结构不能一一对应；相关近似已记录。

此过程默认不覆盖任何已有 YAML，重复执行也保留原有迁移记录：

```bash
pixi run python -m tools.annotate_o30_taxonomy
pixi run python -m tools.check_hand_assets --hand linker_o30_right
MUJOCO_GL=egl pixi run python -m tools.render_hand_templates
```

总览图位于 `output/o30_taxonomy_verified/template_atlas.png`。该图显示初始姿态及橙色
接触法向，不能代替动力学验证。全分类小规模生成命令如下：

```bash
pixi run python -m tools.run_sharpa_dataset --hand linker_o30_right \
  --output output/o30_taxonomy_verified --hours 1 --object-limit 6 \
  --points 512 --workers 4 --gpu 0

pixi run python -m tools.report_o30_taxonomy --run output/o30_taxonomy_verified
```

复现新运行时请换用新的输出目录，且汇总命令的 `--run` 指向相同目录。
在运行汇总之前，先保存该运行对应的几何检查结果：

```bash
pixi run python -m tools.check_hand_assets --hand linker_o30_right \
  --report output/o30_taxonomy_verified/asset_checks.json
```
报告会区分“有成功样本”与“本次无成功样本，需继续改进”，不把文件存在当作评估通过。

## 检查与编辑

在项目根目录运行：

```bash
pixi run python -m tools.prepare_linker_o30
pixi run python -m tools.check_hand_assets --hand linker_o30_right
pixi run python -m tools.sharpa_template_editor --hand linker_o30_right --port 8048
```

浏览器打开 `http://127.0.0.1:8048`，在目标手模板下拉框中选择现有模板。
编辑器复用 Shadow 参考手，支持 20 个关节滑块、表面接触点拾取、法向调整和标注保存。
O30 没有 Sharpa 的弹性体命名，因此默认关闭仅拾取弹性体的过滤选项。

只有需要重新生成初始模板时才运行以下命令；它会覆盖已有标注，包括人工修改：

```bash
pixi run python -m tools.bootstrap_o30_templates --overwrite
```

## 生成与评估

```bash
pixi run python -m tools.run_sharpa_dataset --hand linker_o30_right \
  --output output/o30_validation --hours 0.5 --object-limit 6 \
  --points 1024 --workers 2 --gpu 0

pixi run python -m dexonomy.main op=v3d hand=linker_o30_right \
  save_dir=output/o30_validation/shards/0000 n_worker=2 op.n_max=0

pixi run python -m tools.sharpa_grasp_viewer \
  --result-dir output/o30_validation/shards/0000/succ_grasp \
  --title 'O30 右手成功抓取' --port 8049
```

新任务应使用新的输出目录；只有恢复原任务时才对原目录传入 `--resume`，
它会保留原设置和截止时间。省略 `--hand` 时仍默认使用 Sharpa。
当前使用普通 `grasp`，关闭模板演化，不包含机械臂和轨迹规划器。

## 本地验证结果（2026-11-09）

确定性选择 6 个 DGN 物体，每个使用 3 个尺度，沿用默认接触、碰撞、QP 和评估阈值。
力闭合评估沿 ±X、±Y、±Z 施加外力，要求物体位移小于 0.05 米、转角小于 15 度。

| 模板 | 初始候选 | 有效抓取 | 仿真成功 |
| --- | ---: | ---: | ---: |
| 14_Tripod | 144 | 12 | 9 |
| 24_Tip_Pinch | 144 | 63 | 18 |
| fingertip_large | 144 | 84 | 70 |
| 合计 | 432 | 159 | 97 |

这是小规模集成验证，不代表整个数据集或实物机器人的成功率。
成功样本均已导出网格；运行结果、日志位于 `output/o30_validation/`，不纳入 Git。
资产检查覆盖执行器、关节限位、接触点到表面的距离和外法向；
8 组非零姿态保持了源模型运动学。Sharpa 回归结果为 24 个候选、10 个有效抓取、
10 个仿真成功抓取。

## 全分类验证结果（2026-11-10）

全部 34 份标注的关节限位、接触点、外法向检查通过；新增 31 份均投影到
MuJoCo 碰撞凸包表面。首次全库验证时，原有三份标注与前次提交逐字节一致；后续优化变化另见下节。

6 个物体、每个 3 个尺度的完整运行得到 4,896 个候选、774 个有效抓取、
509 个仿真成功样本。33 类编号抓取全部至少有一个成功样本，详细数据见
[逐类验证报告](taxonomy_validation.md)。少数类别只有 1–2 个成功样本，
这些结果不能作为大规模成功率结论。

本轮运行目录为 `output/o30_taxonomy_verified/`，其中包含标注快照、
几何检查报告、总览图、索引、日志和成功样本。
全集标注编辑器地址为 `http://127.0.0.1:8050`，可通过以下命令启动：

```bash
pixi run python -m tools.sharpa_template_editor --hand linker_o30_right --port 8050

pixi run python -m dexonomy.main op=v3d hand=linker_o30_right \
  save_dir=output/o30_taxonomy_verified/shards/0000 n_worker=4 op.n_max=0

pixi run python -m tools.sharpa_grasp_viewer \
  --result-dir output/o30_taxonomy_verified/shards/0000/succ_grasp \
  --title 'O30 全分类成功抓取' --port 8051
```

## Shadow 交叉验证与弱项优化（2026-11-10）

以相同 26 类、6 个物体、3 个尺度和 512 点采样重新验证 Shadow，整体成功率为
19.76%，Sharpa 为 15.41%，O30 优化前为 10.26%。这三组是整套配置的基线比较：
Shadow 有 22 关节、18 个耦合执行器并设置 ±3 Nm 上限，Sharpa 未设力矩上限，
O30 使用 ±1 Nm 仿真假设，因此不能只归因于标注。完整报告见
[三手交叉验证](cross_validation.md)，逐类数据在 `cross_validation.csv`。

针对 14、24、27、30 类评估了 17 份候选，仅由原六物体调参集选择候选并冻结，
再用另外六个不重叠物体验证。四类合计从 51/576（8.85%）提高到
183/576（31.77%）。这不是全 26 类的分数，不能与上面的整体分数直接比较。

四类候选均写入当前 `raw_anno/`，接触指节集合及数量、力矩限制和评估阈值不变；
独立验证前后逐类结果见 [优化报告](optimization_report.md)。Palmar 只增加了
3 个成功样本，提升较小，仍需扩大样本验证。原标注可在
`annotation_history/2026-11-10_before_optimization/` 找到，发布记录为
`optimization_provenance.json`。

生成器新增 `--annotation-dir`，可使用独立候选目录；新增 `--object-offset`，
可跳过确定性排序后的物体，实现不重叠验证。正常运行默认值仍与原流程一致。

## 优化实验如何复现

完整复现应先使用新的实验目录，并用备份的四类原标注生成候选，避免把优化后
标注当成旧基线。下例中的 `output/o30_taxonomy_verified` 指优化前的完整运行，
其中包含原标注快照和统计；这些大文件不纳入 Git，需保留本地结果或重新运行基线。

```bash
pixi run python -m tools.optimize_o30_annotations \
  --annotation-dir assets/hand/linker_o30_right/annotation_history/2026-11-10_before_optimization \
  --output output/o30_reproduce/candidates
pixi run python -m tools.run_sharpa_dataset --hand linker_o30_right \
  --annotation-dir output/o30_reproduce/candidates --output output/o30_reproduce/train \
  --hours 1 --object-limit 6 --points 512 --workers 4 --gpu 0
pixi run python -m tools.select_o30_candidates --directory output/o30_reproduce \
  --baseline-run output/o30_taxonomy_verified
pixi run python -m tools.run_sharpa_dataset --hand linker_o30_right \
  --annotation-dir assets/hand/linker_o30_right/annotation_history/2026-11-10_before_optimization \
  --output output/o30_reproduce/heldout_baseline --hours 1 \
  --object-offset 6 --object-limit 6 --points 512 --workers 4 --gpu 0
pixi run python -m tools.run_sharpa_dataset --hand linker_o30_right \
  --annotation-dir output/o30_reproduce/selected_annotations \
  --output output/o30_reproduce/heldout_optimized --hours 1 \
  --object-offset 6 --object-limit 6 --points 512 --workers 4 --gpu 0
pixi run python -m tools.report_o30_optimization --directory output/o30_reproduce \
  --baseline-run output/o30_taxonomy_verified
```

最后一步默认只生成报告；确认新实验结果后才使用 `--apply` 发布。当前报告脚本针对
本次四类、六个物体的实验设计，改变实验规模时须同时调整报告断言和汇总说明。

## 提交前回归（2026-11-10）

模型改为子模块后，重新转换的 MJCF 和转换来源记录与原文件逐字节一致；
34 份标注通过几何检查，8 组非零姿态通过源模型运动学对比。
从评测快照重新选择候选，所得四份标注哈希与发布记录一致。
另对一个独立验证物体、005 尺度的 Tip Pinch 跑通转换、初始化、细化和评估：
8 个初始候选、7 个有效抓取、7 个仿真成功，各阶段无错误诊断。
这是提交回归，不替代上面的完整交叉验证与独立验证。
