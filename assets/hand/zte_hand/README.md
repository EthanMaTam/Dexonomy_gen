# ZTE 右手 V1.0

当前库包含 33 类编号抓取及 `fingertip_large`，共 34 份程序辅助标注。
2026-11-27 已将 33 类编号标注直接从 Shadow 整手重建，并修正编辑器的 90° 掌面显示偏差。
`fingertip_large` 保留为独立实验种子。[当前验收索引](current_validation.md) 报告本版的形态偏差和小规模回归。
旧版 **29/33 类**证据及完整标注备份于 `annotation_history/2026-11-27_before_shadow_retarget/`，
不能作为本版的全库成功率。分类语义及实物表现仍待人工确认。

## 模型来源与转换

输入为用户提供的 `right_hand_V1.0.zip`，模型名 `zte-hand_right`，SHA-256：
`86453deb218dd653c7c1572cc1a19fd8f67f7d138a88c10e01dc1610289b25a4`。
压缩包未提供公开下载地址或许可证。源包及解压网格保留本地，Git 仅记录转换代码、
MJCF、标注和小型验收报告。源文件及网格指纹见 [provenance.json](provenance.json)。

包内 URDF 有 22 个转动关节，其中 2 个属于腕部，未描述传动闭环；MJCF 有
42 个关节坐标、17 个执行器、10 组 connect 约束。本次以 MJCF 为来源，保留其
驱动与被动关系，去掉腕部两轴，将手掌设为独立手模型的基座。

生成的 `right.xml` 保留 **40 个关节坐标、15 个执行器、25 个被动坐标及 10 组连接约束**。
40 个坐标不表示 40 个独立电机。每指两个无可见几何的末端枢轴转角不改变连接点位置，
拟合时固定为零；其原始惯量和关节仍保留。

当前仿真假设如下，尚未经过实物标定：

- 保留手指 CAD 惯量、关节限位和连接参数，统一连杆及关节名称。
- 位置控制采用 `kp=5`、`kv=0.1`、力矩限制 ±1 Nm；阻尼 0.1、电枢惯量 0.01
  沿用原 MJCF。源 URDF 的 effort/velocity 全为零，源 MJCF 的力矩范围为 ±100，
  因此不能将这些控制参数视为硬件指标。
- 移除源模型的 253 对碰撞排除，仅保留掌部安装壳体与各指前两节的 10 对装配排除。
  不同手指及手与物体的碰撞保持启用，外部碰撞面采用 CAD 网格凸包。
- 内部传动连杆保留可视几何和连接动力学，关闭其碰撞。固定手掌惯量由保留几何编译，
  不沿用包含已删除腕部的聚合惯量。

在项目根目录放置源包，运行：

```bash
pixi run python -m tools.prepare_zte_hand
pixi run python -m tools.check_zte_hand
pixi run python -m tools.check_hand_assets --hand zte_hand
```

源文件解压到 `third_party/zte_hand/right_hand_V1.0/`；输入指纹不符会停止转换。
当前没有 ZTE 上游 Git 地址，因此未创建子模块。转换检查覆盖 8 组非零姿态的源模型
运动学、解析连接雅可比、执行器限幅和自碰撞设置，结果见
[model_validation.json](model_validation.json)。随机姿态的运动学对比与标注闭环检查分别进行。

## 标注位置和编辑

`raw_anno/*.yaml` 使用以下数据：

- `qpos`：按 `right.xml` 关节顺序保存 40 个弧度角，包含被动坐标，不含手的空间位姿。
- `contact`：按指节分组的 `[x,y,z,nx,ny,nz]`；使用手基座坐标系，位置单位为米，
  后三项是单位外法向。
- `required_cbody`：如存在，则保存必须满足的接触分组。

编号标注直接读取 Shadow，拟合全部手指的指节方向、指腹朝向和接触相对布局，
包括没有接触标注的手指；在 ZTE 碰撞凸包上重定位接触，并求解闭环及自碰撞约束。
不再经过 O30 级联迁移。模型与 YAML 均保留原生坐标，只有显示根节点旋转至 Shadow 掌面。
来源和哈希见 `taxonomy_provenance.json`、`taxonomy_metadata.json`；逐类形态偏差见 [重建报告](retarget_report.md)。

需要补齐缺失标注时运行以下命令；默认保留已有文件：

```bash
pixi run python -m tools.annotate_zte_hand
MUJOCO_GL=egl pixi run python -m tools.render_hand_templates --hand zte_hand \
  --output output/shadow_retarget/zte_hand/template_atlas.png
pixi run python -m tools.sharpa_template_editor --hand zte_hand --port 8054
```

直接迁移只需要 Shadow 与 ZTE 资产。浏览器打开 `http://127.0.0.1:8054`，左侧为 Shadow
参考，右侧为 ZTE。编辑器提供 15 个电机滑块并求解被动关节，拒绝无法闭合的目标；
保存时保留必要接触分组。接触点绑定到指节随关节移动，避免直接修改 YAML 角度而
保留旧接触坐标。标注检查要求连接误差小于 **0.02 mm**。
**Align palm with Shadow** 默认开启；**Load matching target template** 同步两侧类别。

`skeleton.yaml`、`body_group.yaml` 和空的 `keypoint.yaml` 由转换器生成。本库直接保存
六维接触坐标，不使用整数关键点引用。编辑器可通过 `--check` 检查资产后退出。

## 生成和验收

使用新的输出目录运行当前流程：

```bash
pixi run python -m tools.run_sharpa_dataset --hand zte_hand \
  --output output/zte_hand_run --hours 1 --object-limit 6 \
  --points 512 --workers 2 --gpu 0
pixi run python -m tools.check_zte_hand --report output/zte_hand_run/model_checks.json
pixi run python -m tools.check_hand_assets --hand zte_hand \
  --report output/zte_hand_run/asset_checks.json
pixi run python -m tools.replay_zte_validation --run output/zte_hand_run --count 8
pixi run python -m tools.report_zte_validation --run output/zte_hand_run
```

抓取、碰撞、接触、QP 及六方向外力评估阈值沿用原流程。ZTE 另启用两项闭环保护：

- `connect_grasp_limit: 0.0002`：最终姿态超限时，保持基座和电机抓取角不变，求解被动
  坐标，再寻找满足限值且留有 10% 余量的最小必要修正。修正后重新运行原有检查，
  再生成预抓取和评估；无法修复或检查失败则拒绝。
- `connect_eval_limit: 0.0002`：每次 `step_sim` 返回后更新运动学，累计动态采样误差。
  超限时即使外力测试通过也拒绝，诊断写入 `log/eval_0/connect_checks/**/*.json`。

**0.2 mm 是运行姿态的工程检查限值，未经硬件公差标定。** 动态检查在 `step_sim`
返回后采样，不代表逐积分步的全轨迹上界；标注的 0.02 mm 限值用于精确初始化。

建议使用 1–2 个工作进程。评估原先会启动 `3*n_worker` 个进程，现已改为遵守配置。
恢复同一任务使用 `--resume`，保留原设置和截止时间，并核对模型及手部配置哈希。
配置改变后使用新目录。设置阶段中断时，可用相同参数加 `--resume` 恢复，先验证已有快照。

查看成功抓取前导出碰撞凸包，避免为每个姿态重复保存完整 CAD 网格：

```bash
pixi run python -m dexonomy.main op=v3d hand=zte_hand \
  save_dir=output/zte_hand_run/shards/0000 n_worker=1 op.n_max=0 op.hand.mode=collision
pixi run python -m tools.sharpa_grasp_viewer \
  --result-dir output/zte_hand_run/shards/0000/succ_grasp \
  --validation-csv output/zte_hand_run/closed_loop_samples.csv --port 8055
```

需要外观细节时，可对少量样本使用 `op.hand.mode=visual op.n_max=16`。
运行日志、标注快照、生成数据及网格保存在 `output/`，不纳入 Git。

## 已验收结果和边界

以下为重建前标注的历史实验；本版状态以 [当前验收索引](current_validation.md) 为准。

| 阶段 | 结果 | 证据 |
| --- | --- | --- |
| 初始种子回归 | 72 个候选、19 个有效抓取、17 个仿真成功 | 本地 `output/zte_hand_smoke/` |
| 首次完整库基线 | 4,896 个候选、458 个有效抓取、344 个标准成功；342 个最终姿态连接误差合格，26/33 类有样本 | [历史全库表](taxonomy_validation.md) |
| Lateral 优化 | 独立六物体从 0/144 提高至 3/144，已替换标注 | [接触优化](optimization_report.md) |
| 原标注补充验证 | Writing Tripod、Inferior Pincer 分别获得 3、2 个合格样本，保留原标注 | [收拢实验](rest_optimization_report.md)、[接触优化](optimization_report.md) |
| 原闭环超限修复 | 两例最终误差约 0.18 mm，生成与外力回归通过；动态复跑最大约 0.1765 mm | `closure_repair_provenance.json` |
| 动态拒绝回归 | 五个真实夹具中四个通过，一个约 0.206 mm 的超限样本被正确拒绝 | `dynamic_guard_regression.json` |

原 342 个合格姿态经新投影函数检查后逐坐标未变。重建前分阶段证据覆盖 29/33 类；
Distal、Tripod Variation、Stick、Ventral 本轮仍无可靠独立证据。收拢候选未通过独立
复验，未发布。原 26 类尚未按新增动态规则逐例全量重算，也未做实物验收。
首次基线与后续实验不能混算成功率；[交叉验证](cross_validation.md) 同样保留为历史基线。

原标注和报告备份在 `annotation_history/2026-11-21_before_optimization/`。
独立实验保存选择时间、模型及标注哈希、物体清单和接触拓扑检查。发布条件为调参集
有合格样本，且独立六物体至少增加三个合格样本；这是单随机种子的工程筛选规则，
不代表统计显著性。

## 实验工具和审阅样本

| 工具 | 用途 |
| --- | --- |
| `tools.optimize_zte_annotations` | 指腹重定位及球/柱候选；`--annotation-dir` 可读取原标注备份 |
| `tools.optimize_zte_rest_pose` | 未参与接触手指的收拢、侧摆候选 |
| `tools.select_zte_candidates` | 仅按调参结果冻结选择；`--rest` 用于收拢实验 |
| `tools.repair_zte_grasps` | 在独立输出中修复超限结果，并重做原检查及外力评估 |
| `tools.report_zte_optimization` | 独立 A/B 汇总；满足条件后才用 `--apply` 发布 |
| `tools.report_zte_current` | 汇总当前整手重建报告及本版生成回归，不沿用旧标注成功证据 |

完整选择器需要对应实验缓存：接触实验使用 `screen`、`cylinder/screen`、
`additional_distal/screen`，收拢实验使用 `train`、`spread/train`。冻结候选再与原标注在
不重叠物体上分别验证；第二轮收拢实验使用另一组六个新物体。历史数据未纳入 Git，
新工作区需要源模型及基线运行结果才能完整复现。各工具提供 `--help`。

旧版 10 个逐例动态审计合格的改进、补充验证和修复样本可汇集历史审阅，不代表当前库：

```bash
pixi run python -m tools.report_zte_current
pixi run python -m tools.build_zte_review --historical
pixi run python -m dexonomy.main op=v3d hand=zte_hand \
  save_dir=output/zte_improvement_review/shards/0000 n_worker=1 op.n_max=0 op.hand.mode=collision
pixi run python -m tools.sharpa_grasp_viewer \
  --result-dir output/zte_improvement_review/shards/0000/succ_grasp \
  --validation-csv output/zte_improvement_review/closed_loop_samples.csv \
  --title 'ZTE 改进与修复验证' --port 8058
```
