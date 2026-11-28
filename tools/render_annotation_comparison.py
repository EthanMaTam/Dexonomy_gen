"""Render Shadow, previous and retargeted templates in the same palm frame."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from dexonomy.util.file_util import load_yaml
from tools.hand_frames import display_rotation
from tools.sharpa_template_editor import ROOT, _contact_entries


def render(hand, directory, name, azimuth, elevation):
    model = mujoco.MjModel.from_xml_path(str(ROOT/'assets/hand'/hand/'right.xml'))
    data = mujoco.MjData(model)
    annotation=load_yaml(str(directory/(name+'.yaml')))
    data.qpos[:]=annotation['qpos']
    mujoco.mj_forward(model,data)
    model.vis.headlight.ambient[:]=.55
    model.vis.headlight.diffuse[:]=.8
    camera=mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[0.,-.03,.12]
    camera.distance=.36
    camera.azimuth=azimuth
    camera.elevation=elevation
    rotation=display_rotation(hand)
    keypoints=load_yaml(str(ROOT/'assets/hand'/hand/'keypoint.yaml')) or {}
    for key,value in keypoints.items():
        if isinstance(value,str): keypoints[key]=keypoints[value]
    with mujoco.Renderer(model,height=360,width=400) as renderer:
        renderer.update_scene(data,camera=camera)
        for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
            geom.pos[:]=rotation @ geom.pos
            geom.mat[:]=rotation @ geom.mat.reshape(3,3)
        for body,values in annotation['contact'].items():
            for value in _contact_entries(values):
                if isinstance(value,list): point,normal=np.asarray(value).reshape(2,3)
                else:
                    local=np.asarray(keypoints[body][int(value)]).reshape(2,3)
                    bid=model.body(body).id
                    r=data.xmat[bid].reshape(3,3)
                    point,normal=r@local[0]+data.xpos[bid],r@local[1]
                point,normal=rotation@point,rotation@normal
                geom=renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.ones(3)*.001,
                                  point,np.eye(3).ravel(),np.array([1.,.5,.05,1.],dtype=np.float32))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.001,point,point+.012*normal)
                renderer.scene.ngeom+=1
        return renderer.render().copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hand',required=True,choices=['zte_hand','linker_o30_right'])
    parser.add_argument('--before',type=Path)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--templates',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--azimuth',type=float,default=240)
    parser.add_argument('--elevation',type=float,default=-15)
    args=parser.parse_args()
    before=args.before or ROOT/'assets/hand'/args.hand/'raw_anno'
    fig,axes=plt.subplots(len(args.templates),3,figsize=(12,3.6*len(args.templates)),squeeze=False,facecolor='#20252b')
    for row,name in enumerate(args.templates):
        for col,(hand,directory,title) in enumerate([
            ('shadow',ROOT/'assets/hand/shadow/raw_anno','Shadow reference'),
            (args.hand,before,'Previous (palm aligned)'),
            (args.hand,args.after,'Whole-hand retarget')]):
            axes[row,col].imshow(render(hand,directory,name,args.azimuth,args.elevation))
            axes[row,col].set_title(name.replace('_',' ')+'\n'+title,color='white',fontsize=9)
            axes[row,col].axis('off')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=120,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)


if __name__=='__main__': main()
