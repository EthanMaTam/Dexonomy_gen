# Linker Hand O30 右手适配说明

官方模型文件名为 **O30i right / 0803**，来源为
[厂商模型仓库](https://github.com/linker-bot/linkerhand-urdf)，固定提交：
`49f26021b0a91ff5c33150df22970c07bb548672`。
原始 URDF 和网格保存在 `third_party/linkerhand-urdf/`，不纳入本项目 Git；
使用这些资产时应保留其原有许可证和声明。

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

原始抓取标注位于本目录的 `raw_anno/`：

| 文件 | 初始抓取类型 | 标注的手指 |
| --- | --- | --- |
| `raw_anno/24_Tip_Pinch.yaml` | 指尖捏取 | 拇指、食指 |
| `raw_anno/14_Tripod.yaml` | 三指抓取 | 拇指、食指、中指 |
| `raw_anno/fingertip_large.yaml` | 较大三指抓取 | 拇指、食指、中指 |

这 3 份是**程序自动生成的种子模板**，并非人工逐点标注，也不代表完整抓取分类库。

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
但当前 3 份原始标注没有使用整数引用。`keypoint.yaml` 的关键点位于各自连杆的局部
坐标系，与 `raw_anno/` 中直接保存的坐标系不同。

**不要仅修改 YAML 的 `qpos` 而保留原接触坐标。** 改变手势后，原接触点一般就不再
位于手指表面。编辑器在内部将接触点绑定到连杆局部坐标，调整关节后会让接触点随
连杆移动，保存时再转换为上述手基座坐标，因此推荐通过编辑器调整标注。

## 当前模板如何生成

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
