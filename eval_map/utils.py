import os
import numpy as np
import yaml
import torch
from munch import munchify
from scipy.spatial.transform import Rotation as R

from src.utils.camera_utils import Camera
from src.utils.datasets import get_dataset
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov

def line_to_T(line):
    # Vectorized parsing of the space-separated string
    data = np.fromstring(line, sep=' ')
    T = np.eye(4)
    T[:3, :3] = R.from_quat(data[4:8]).as_matrix()
    T[:3, 3] = data[1:4]
    return T

def get_render_pipline_params(exp_folder):
    with open(os.path.join(exp_folder, 'cfg.yaml'), 'r') as file:
        cfg = yaml.safe_load(file)
    return munchify(cfg["mapping"]["pipeline_params"])

def check_quality_of_transformation(Trans, Rot, T_we_wg_final):
    num_check = 10
    interval = len(Rot) // num_check
    R_wg_we = np.linalg.inv(T_we_wg_final[:3, :3])
    T_we_wg_ref = T_we_wg_final[:3, 3]

    for i in range(num_check):
        start, end = i * interval, (i + 1) * interval
        chunk_R_we_wg = R.from_matrix(np.array(Rot[start:end])).mean().as_matrix()
        relative_rot_ang = np.degrees(R.from_matrix(R_wg_we @ chunk_R_we_wg).magnitude())
        
        if relative_rot_ang > 2.0:
            print(f"Warning: rotation diff {relative_rot_ang:.3f} degree at chunk {i}")
        
        chunk_trans = np.mean(Trans[start:end], axis=0)
        dist = np.linalg.norm(chunk_trans - T_we_wg_ref)
        if dist > 0.05:
            print(f"Warning: translation diff {dist*100:.3f} cm at chunk {i}")

def get_temp_viewpoint(static_cam_cfg, full_resol=False, exp_cfg=None):
    if full_resol:
        fx, fy = static_cam_cfg['fx'], static_cam_cfg['fy']
        cx, cy = static_cam_cfg['ppx'], static_cam_cfg['ppy']
        W, H = static_cam_cfg['width'], static_cam_cfg['height']
    else:
        H_out, W_out = exp_cfg['cam']['H_out'], exp_cfg['cam']['W_out']
        W_orig, H_orig = static_cam_cfg['width'], static_cam_cfg['height']
        r_w, r_h = W_out / W_orig, H_out / H_orig
        fx, fy = static_cam_cfg['fx'] * r_w, static_cam_cfg['fy'] * r_h
        cx, cy = static_cam_cfg['ppx'] * r_w, static_cam_cfg['ppy'] * r_h
        W, H = W_out, H_out

    projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
    ).transpose(0, 1).to(device='cuda')

    identity = torch.eye(4, device='cuda')
    fovx = focal2fov(fx, W)
    fovy = focal2fov(fy, H)
    
    viewpoint = Camera(
        0,              # idx
        None,           # gt_color
        None,           # est_depth
        identity,       # est_pose
        projection_matrix,
        fx, fy, cx, cy,
        fovx, fovy,
        H, W,
        features=None,
        device='cuda',
    )
    return viewpoint