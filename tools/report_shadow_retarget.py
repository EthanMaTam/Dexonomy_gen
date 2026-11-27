"""Report current template similarity without inheriting stale physics evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from tools.sharpa_template_editor import ROOT


def write_report(hand, smoke_run=None):
    directory=ROOT/'assets/hand'/hand
    report=json.loads((directory/'retarget_report.json').read_text())
    assert hashlib.sha256((directory/'right.xml').read_bytes()).hexdigest()==report['target_model_sha256']
    physics={}
    if smoke_run is not None and (smoke_run/'summary.csv').exists():
        job=json.loads((smoke_run/'job.json').read_text())
        if job['model_sha256'] != report['target_model_sha256']: raise ValueError('Smoke model differs')
        for row in csv.DictReader((smoke_run/'summary.csv').open()):
            path=smoke_run/'raw_anno_snapshot'/(row['template']+'.yaml')
            if path.read_bytes()!=(directory/'raw_anno'/path.name).read_bytes():
                raise ValueError('Smoke annotation differs from current library')
            physics[row['template']]=dict(initializations=int(row['initializations']),
                                         successful=int(row['successful']))
    rows=sorted(report['records'],key=lambda r:int(r['template'].split('_')[0]))
    text=f'# {hand}：Shadow 整手标注重建\n\n'
    text+='2026-11-27 起，33 类编号模板直接从 Shadow 重建；`fingertip_large` 是保留的独立实验种子。\n\n'
    text+='O30、ZTE 的原生手掌坐标相对 Shadow 绕 Z 轴偏转 90°。编辑器只旋转显示根节点，模型、关节轴、接触坐标仍使用原生坐标。\n\n'
    text+='旧流程只拟合接触指节，遗漏非接触手指姿态，且 ZTE 继承了 O30 的偏差。新流程直接从 Shadow 拟合全部指节方向、指腹朝向与接触相对布局，同时保留碰撞、关节限位和 ZTE 闭环检查。\n\n'
    text+='这些指标反映与参考形态的接近程度，不是分类语义人工验收或实物成功率。历史全库成功率及 29/33、33/33 覆盖率只属于旧标注，不能用于本版。\n\n'
    for field,label,scale,unit in [
        ('mean_bone_direction_error_deg','整手指节方向平均偏差',1,'°'),
        ('contact_layout_rms_error_m','接触相对布局 RMS 偏差',1000,' mm')]:
        before=sum(r['before'][field] for r in rows)/len(rows)*scale
        after=sum(r['after'][field] for r in rows)/len(rows)*scale
        text+=f'{label}：{before:.2f}{unit} → {after:.2f}{unit}。\n\n'
    text+='| 类别 | 指节方向偏差：旧/新（°） | 接触布局 RMS：新（mm） | 本版小规模仿真成功/候选 |\n| --- | ---: | ---: | --- |\n'
    records=[]
    for row in rows:
        name=row['template']
        digest=hashlib.sha256((directory/'raw_anno'/(name+'.yaml')).read_bytes()).hexdigest()
        assert digest==row['candidate_sha256'],f'Annotation changed: {name}; regenerate the report'
        source=ROOT/'assets/hand/shadow/raw_anno'/(name+'.yaml')
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256'],f'Shadow reference changed: {name}'
        run=physics.get(name)
        status=f"{run['successful']}/{run['initializations']}" if run else '待验证'
        text+=f"| {name} | {row['before']['mean_bone_direction_error_deg']:.1f}/{row['after']['mean_bone_direction_error_deg']:.1f} | {1000*row['after']['contact_layout_rms_error_m']:.1f} | {status} |\n"
        records.append(dict(template=name,sha256=digest,geometry_passed=True,
                            human_classification_review_complete=False,
                            has_qualified_success=None,qualified_samples_in_supporting_run=None,
                            smoke_physics=run,supporting_experiment='retarget_report.md',
                            annotation_changed=True))
    text+='\n小规模仿真仅检查流程；每个结果仍需与 Shadow 对照审阅。ZTE 成功结果沿用 0.2 mm 最终姿态和动态采样闭环保护。\n'
    (directory/'retarget_report.md').write_text(text)
    if hand=='zte_hand':
        (directory/'current_validation.md').write_text(text.replace('# zte_hand：Shadow 整手标注重建','# ZTE 当前标注库验收索引',1))
        (directory/'current_validation.json').write_text(json.dumps(dict(
            model_sha256=report['target_model_sha256'],numbered_categories=33,templates=34,
            annotation_version='2026-11-27_shadow_whole_hand',
            numbered_categories_with_qualified_samples=None,
            historical_coverage_not_applicable=True,records=records),ensure_ascii=False,indent=2)+'\n')
    print(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hand',required=True,choices=['zte_hand','linker_o30_right'])
    parser.add_argument('--smoke-run',type=Path)
    args=parser.parse_args()
    write_report(args.hand,args.smoke_run)


if __name__=='__main__': main()
