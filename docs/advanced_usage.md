# 补充工作流与实验记录

首次使用先阅读 [README](../README.md)，完成环境配置、模板编辑和抓取生成。
本文命令均从仓库根目录运行。

## O30 适配与实验记录

2026-11-27 起当前编号标注已按 Shadow 整手重建；下述成功率属于重建前历史实验。
新版的逐类形态对比与生成回归见 [O30 重建报告](../assets/hand/linker_o30_right/retarget_report.md)
及 [ZTE 当前验收](../assets/hand/zte_hand/current_validation.md)。


O30 右手适配复用现有模板编辑器、数据集生成器和抓取查看器。
官方模型文件名为 `O30i right / 0803`，具有 20 个关节。
官方 URDF 和网格通过 Git 子模块管理，主仓库锁定厂商提交
`49f26021b0a91ff5c33150df22970c07bb548672`。首次安装已初始化子模块时无需重复下载；已有工作区运行：

```bash
git submodule update --init third_party/linkerhand-urdf
pixi run python -m tools.prepare_linker_o30
pixi run python -m tools.check_hand_assets --hand linker_o30_right
pixi run python -m tools.sharpa_template_editor --hand linker_o30_right --port 8048
```

当前已覆盖 33 类编号抓取，另保留较大三指模板，共 34 份初始标注，位于
`assets/hand/linker_o30_right/raw_anno/`。初次补齐分类时保留原有 3 份，随后优化了
14、24、27、30 类；优化前标注已备份。
逐类几何及仿真结果见 [全分类验证表](../assets/hand/linker_o30_right/taxonomy_validation.md)。
执行抓取生成和仿真评估：

```bash
pixi run python -m tools.run_sharpa_dataset --hand linker_o30_right \
  --output output/o30_run --hours 1 --object-limit 6 \
  --points 1024 --workers 2 --gpu 0
```

最初三份模板的集成验证得到 97 个仿真成功抓取。全分类验证覆盖 6 个物体、
3 个尺度，从 4,896 个候选中得到 509 个仿真成功抓取，33 类编号模板均有成功样本。
当前使用假设的执行器参数，尚未实物标定。
模型转换、装配碰撞排除、标注格式与编辑逻辑、复现和查看方法详见
[O30 右手适配说明](../assets/hand/linker_o30_right/README.md)。

已增加 Shadow 同条件对照，并对 O30 的 Tripod、Tip Pinch、Quadpod、Palmar
完成小范围优化及独立物体验证。模型与全局验收阈值不变，原标注已备份。
四类在另外六个物体上的成功率从 8.85% 提升到 31.77%；此分数不是全分类成功率。
详见 [O30 优化报告](../assets/hand/linker_o30_right/optimization_report.md)。


## 上游逐步操作与完整流水线

详见 [上游操作教程](../getting_started/2_running_tutorial.md)。教程中的 `dexrun`、
`dexsyn` 需要先在激活的 Conda 环境中按 README 安装项目，才能作为终端命令使用。
单卡运行时覆盖 `init_gpu=[0]`，并按机器资源调整 `n_worker`。
主 README 中的 dataset runner 直接调用 Python 模块，并显式指定 GPU 和进程数。

## 可选轨迹合成（BODex）

仅生成抓取手势无需安装 BODex。需要 UR10e + Shadow 轨迹合成时，先准备子模块：

```bash
git config submodule.third_party/BODex.url https://github.com/JYChen18/BODex.git
git submodule update --init --recursive third_party/BODex
```

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

## Upstream updates

For a fresh clone, configure the original repository as `upstream`:

```bash
git remote add upstream https://github.com/JYChen18/Dexonomy.git
git fetch upstream
```

Commit local changes before integrating updates with `git merge upstream/main`.
`origin` should point to `EthanMaTam/Dexonomy_gen`.


## 模板检查说明

`tools.sharpa_template_editor --check` 检查模型能否加载、标注格式及选点所需资产；
`tools.check_hand_assets` 进一步检查关节限位、接触点距离及表面外法向。
前者通过不代表后者一定通过，也不代表生成的抓取能通过仿真。

本次文档核对中，O30 本地开发标注通过严格检查（本次未发布这些标注改动）；Sharpa 的
`18_Extensior_Type.yaml` 在 `left_hand_C_MC` 上有接触点距碰撞表面约 2.28 mm，
超过检查器的 2 mm 阈值。原标注保留，尚未修正。
使用 Sharpa 全部模板前应在编辑器检查该指节的接触点并重新保存；
选择 `--templates 14_Tripod` 的小任务不会使用这一模板。
