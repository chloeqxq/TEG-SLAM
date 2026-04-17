import os
import cv2
import json
import yaml
import torch
import numpy as np

from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from eval_map.utils import *

def get_resized_color(frame_reader, img):
    return cv2.resize(img, (frame_reader.W_out_with_edge, frame_reader.H_out_with_edge))

exp_scenes = ["ANYmal1","ANYmal2","basketball","crowd",
              "person_tracking","racket",
              "stones","table_tracking1","table_tracking2","umbrella"]

dataset_name = 'Wild_SLAM_Mocap'
full_resol = False

# Global metrics initialization to save time and VRAM
cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
background = torch.tensor([0, 0.0, 0], dtype=torch.float32, device="cuda")

all_psnr, all_ssim, all_lpips = [], [], []

for exp_scene in exp_scenes:
    exp_folder = f"./output/{dataset_name}/{exp_scene}"
    save_folder = f"{exp_folder}/nvs"
    os.makedirs(os.path.join(save_folder, 'vis'), exist_ok=True)

    with open(os.path.join(exp_folder, 'cfg.yaml'), 'r') as file:
        cfg = yaml.safe_load(file)
    
    data_folder = cfg['data']['input_folder'].replace('ROOT_FOLDER_PLACEHOLDER', cfg['data']['root_folder'])
    cfg['data']['input_folder'] = data_folder
    
    frame_reader = get_dataset(cfg, device='cuda')
    gt_poses = frame_reader.poses
    
    with open(os.path.join(exp_folder, 'traj/est_poses_full.txt'), 'r') as f:
        est_lines = f.readlines()

    with open(os.path.join(exp_folder, 'traj/metrics_full_traj.txt'), 'r') as f:
        # Extract scale using search to avoid assert failure on empty lines
        traj_scale = float([l for l in f.readlines() if 'scale' in l][0].replace('scale:', ''))

    Trans, Rot = [], []
    w2c_first_inv = np.linalg.inv(frame_reader.w2c_first_pose)
    
    for idx, line in enumerate(est_lines):
        T_we_c = line_to_T(line)
        T_we_c[:3, 3] *= traj_scale
        T_wg_c = w2c_first_inv @ gt_poses[idx]
        T_we_wg_curr = T_we_c @ np.linalg.inv(T_wg_c)
        Rot.append(T_we_wg_curr[:3, :3])
        Trans.append(T_we_wg_curr[:3, 3])

    T_we_wg = np.eye(4)
    T_we_wg[:3, :3] = R.from_matrix(np.array(Rot)).mean().as_matrix()
    T_we_wg[:3, 3] = np.mean(Trans, axis=0)
    check_quality_of_transformation(Trans, Rot, T_we_wg)

    with open(os.path.join(data_folder, 'nvs/groundtruth.txt'), 'r') as f:
        static_poses_gt = [l for l in f.readlines() if "#" not in l]
    
    with open(os.path.join(data_folder, 'nvs/per_frame_intrinsics.json'), 'r') as f:
        intrins_per_frame = json.load(f)

    gaussians = GaussianModel(0, config=None)
    gaussians.load_ply(os.path.join(exp_folder, 'final_gs.ply'))
    pipe = get_render_pipline_params(exp_folder)
    
    psnr_array, ssim_array, lpips_array = [], [], []

    for i in range(len(intrins_per_frame)):
        T_wg_c = line_to_T(static_poses_gt[i])
        T_we_c = T_we_wg @ T_wg_c
        T_we_c[:3, 3] /= traj_scale
        T_c_we = torch.tensor(np.linalg.inv(T_we_c), device='cuda', dtype=torch.float32)

        this_intrinsic = intrins_per_frame[str(i)]
        viewpoint = get_temp_viewpoint(this_intrinsic, full_resol=full_resol, exp_cfg=cfg)
        viewpoint.update_RT(T_c_we[:3, :3], T_c_we[:3, 3])
        
        with torch.no_grad():
            rendering_pkg = render(viewpoint, gaussians, pipe, background)
            image = torch.clamp(rendering_pkg["render"], 0.0, 1.0)
        
        # Ground Truth Processing
        input_rgb = cv2.imread(os.path.join(data_folder, f'nvs/rgb/nvs_{i:05d}.png'))
        K = np.eye(3)
        K[0, 0], K[0, 2], K[1, 1], K[1, 2] = this_intrinsic['fx'], this_intrinsic['ppx'], this_intrinsic['fy'], this_intrinsic['ppy']
        input_rgb = cv2.undistort(input_rgb, K, np.array(this_intrinsic['coeffs']))
        input_rgb = get_resized_color(frame_reader, input_rgb)

        # Metrics preparation
        gt_image = torch.from_numpy(input_rgb).float().permute(2, 0, 1).to('cuda') / 255.0
        mask = gt_image > 0 
        
        # Metric calculation
        psnr_score = psnr(image[mask].unsqueeze(0), gt_image[mask].unsqueeze(0)).item()
        ssim_score = ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
        lpips_score = cal_lpips(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

        psnr_array.append(psnr_score)
        ssim_array.append(ssim_score)
        lpips_array.append(lpips_score)

        # Visualization
        rendered_rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rendered_rgb_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        
        output_img = np.zeros((input_rgb.shape[0] * 2 + 20, input_rgb.shape[1], 3), dtype=np.uint8)
        output_img[:input_rgb.shape[0], :, :] = input_rgb
        output_img[input_rgb.shape[0] + 20:, :, :] = rendered_rgb_bgr
        cv2.imwrite(os.path.join(save_folder, f'vis/nvs_{i:04d}.png'), output_img)

    output = {
        "mean_psnr": float(np.mean(psnr_array)),
        "mean_ssim": float(np.mean(ssim_array)),
        "mean_lpips": float(np.mean(lpips_array))
    }
    
    all_psnr.append(output["mean_psnr"])
    all_ssim.append(output["mean_ssim"])
    all_lpips.append(output["mean_lpips"])

    print(f'Scene {exp_scene} - Mean PSNR: {output["mean_psnr"]:.4f}, SSIM: {output["mean_ssim"]:.4f}, LPIPS: {output["mean_lpips"]:.4f}')
    json.dump(output, open(os.path.join(save_folder, "final_result.json"), "w"), indent=4)

print(f'\nOverall: mean psnr: {np.mean(all_psnr):.4f}, ssim: {np.mean(all_ssim):.4f}, lpips: {np.mean(all_lpips):.4f}')