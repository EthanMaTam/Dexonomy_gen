"""Regression checks for palm alignment and whole-hand template transfer."""

import hashlib
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from dexonomy.util.file_util import load_yaml
from tools.hand_frames import display_rotation
from tools.retarget_shadow_annotations import ShadowRetargeter

ROOT=Path(__file__).resolve().parents[1]


class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retargeters={}
        for hand in ('linker_o30_right','zte_hand'):
            try: cls.retargeters[hand]=ShadowRetargeter(hand)
            except (ValueError,FileNotFoundError): pass

    def test_neutral_index_side_and_flexion_align_with_shadow(self):
        if not self.retargeters: self.skipTest('Hand assets unavailable')
        for hand,retarget in self.retargeters.items():
            with self.subTest(hand=hand):
                s,t=retarget.source,retarget.target
                sd,td=mujoco.MjData(s),mujoco.MjData(t)
                mujoco.mj_forward(s,sd);mujoco.mj_forward(t,td)
                sv=sd.xpos[s.body('rh_ffproximal').id]-sd.xpos[s.body('rh_lfproximal').id]
                tv=td.xpos[t.body('index_proximal').id]-td.xpos[t.body('pinky_proximal').id]
                tv=display_rotation(hand)@tv
                self.assertGreater(float(sv@tv/np.linalg.norm(sv)/np.linalg.norm(tv)),.99)
                sd.qpos[s.joint('rh_FFJ3').qposadr[0]]=.4
                name='index_proximal_joint' if hand=='zte_hand' else 'index_mcp_pitch'
                td.qpos[t.joint(name).qposadr[0]]=.4
                mujoco.mj_forward(s,sd);mujoco.mj_forward(t,td)
                sv=sd.xmat[s.body('rh_ffproximal').id].reshape(3,3)[:,2]
                tv=display_rotation(hand)@td.xmat[t.body('index_proximal').id].reshape(3,3)[:,2]
                np.testing.assert_allclose(sv,tv,atol=1e-5)

    def test_tripod_keeps_noncontact_fingers_curled(self):
        if not self.retargeters: self.skipTest('Hand assets unavailable')
        for hand,r in self.retargeters.items():
            with self.subTest(hand=hand):
                ref=load_yaml(str(r.source_dir/'raw_anno/14_Tripod.yaml'))
                candidate=load_yaml(str(r.target_dir/'raw_anno/14_Tripod.yaml'))
                assessment=r.assess(ref,candidate)
                for finger in ('ring','pinky'):
                    self.assertNotIn(finger+'_distal',candidate['contact'])
                    for segment in ('proximal','middle','distal'):
                        self.assertLess(assessment['bone_direction_errors_deg'][finger+'_'+segment],15.)

    def test_current_taxonomy_preserves_contact_topology_and_hashes(self):
        if not self.retargeters: self.skipTest('Hand assets unavailable')
        for hand,r in self.retargeters.items():
            report=json.loads((r.target_dir/'retarget_report.json').read_text())
            self.assertEqual(len(report['records']),33)
            for row in report['records']:
                with self.subTest(hand=hand,template=row['template']):
                    name=row['template']+'.yaml'
                    source=load_yaml(str(r.source_dir/'raw_anno'/name))
                    target=load_yaml(str(r.target_dir/'raw_anno'/name))
                    self.assertEqual({r.mapping[n] for n in source['contact']},set(target['contact']))
                    for name,contacts in source['contact'].items():
                        from tools.sharpa_template_editor import _contact_entries
                        self.assertEqual(len(_contact_entries(contacts)),len(_contact_entries(target['contact'][r.mapping[name]])))
                    if 'required_cbody' in source:
                        self.assertEqual([[r.mapping[n] for n in group] for group in source['required_cbody']],target['required_cbody'])
                    path=r.target_dir/'raw_anno'/(row['template']+'.yaml')
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['candidate_sha256'])
                    path=r.source_dir/'raw_anno'/(row['template']+'.yaml')
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['source_sha256'])


if __name__=='__main__': unittest.main()
