import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
from typing import Optional, Tuple, Union

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from munch import munchify
from colorama import Fore, Style
from torch.multiprocessing import Lock
from scipy.ndimage import binary_erosion
from multiprocessing.connection import Connection
import torch.multiprocessing as mp

from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.system_utils import mkdir_p
from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.utils.general_utils import (
    rotation_matrix_to_quaternion,
    quaternion_multiply,
)
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import (
    getProjectionMatrix2,
    getWorld2View2,
)
from src.depth_video import DepthVideo
from src.utils.datasets import get_dataset, load_metric_depth, load_img_feature
from src.utils.common import as_intrinsics_matrix, setup_seed
from src.utils.Printer import Printer, FontColor
from src.utils.pose_utils import update_pose
from src.utils.slam_utils import (
    get_loss_mapping,
    get_loss_mapping_uncertainty,
    get_loss_tracking,
)
from src.utils.camera_utils import Camera
from src.utils.dyn_uncertainty import mapping_utils as map_utils
from src.utils.dyn_uncertainty import temporal_fusion as temporal_utils
from src.utils.dyn_uncertainty.median_filter import MedianPool2d
from src.utils.plot_utils import create_gif_from_directory, create_video_from_directory
from src.gui import gui_utils

class Mapper(object):
    """
    Mapper thread.

    """

    def __init__(
        self, slam, pipe: Connection, uncer_network: Optional[nn.Module] = None, 
        q_main2vis: Optional[mp.Queue] = None, q_vis2main: Optional[mp.Queue] = None
    ):
        # setup seed
        setup_seed(slam.cfg["setup_seed"])
        torch.autograd.set_detect_anomaly(True)

        self.config = slam.cfg
        self.printer: Printer = slam.printer
        self.pipe = pipe
        self.verbose = slam.verbose
        self.device = torch.device(self.config["device"])
        self.video: DepthVideo = slam.video

        # Set gaussian model
        self.model_params = munchify(self.config["mapping"]["model_params"])
        self.opt_params = munchify(self.config["mapping"]["opt_params"])
        self.pipeline_params = munchify(self.config["mapping"]["pipeline_params"])
        use_spherical_harmonics = self.config["mapping"]["Training"][
            "spherical_harmonics"
        ]
        self.model_params.sh_degree = 3 if use_spherical_harmonics else 0
        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)

        # Set background color
        bg_color = [0, 0, 0]
        self.background = torch.tensor(
            bg_color, dtype=torch.float32, device=self.device
        )

        # Set hyperparams
        self._set_hyperparams()

        # Set frame reader (where we get the input dataset)
        self.frame_reader = get_dataset(self.config, device=self.device)
        self.intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(
            self.device
        )

        # Prepare projection matrix
        if self.config["mapping"]["full_resolution"]:
            intrinsic_full = self.frame_reader.get_intrinsic_full_resol()
            fx, fy, cx, cy = [intrinsic_full[i].item() for i in range(4)]
            W_out, H_out = self.frame_reader.W_out_full, self.frame_reader.H_out_full
        else:
            fx, fy, cx, cy = (
                self.frame_reader.fx,
                self.frame_reader.fy,
                self.frame_reader.cx,
                self.frame_reader.cy,
            )
            W_out, H_out = self.frame_reader.W_out, self.frame_reader.H_out

        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            W=W_out,
            H=H_out,
        ).transpose(0, 1)
        self.projection_matrix = projection_matrix.to(device=self.device)

        # Setup for uncertainty-aware mapping
        self.vis_uncertainty_online = False
        self.uncer_params = munchify(self.config["mapping"]["uncertainty_params"])
        self.uncertainty_aware = self.uncer_params["activate"]
        if self.uncertainty_aware:
            self.uncer_network = uncer_network
            self.uncer_optimizer = torch.optim.Adam(
                self.uncer_network.parameters(),
                lr=self.uncer_params["lr"],
                weight_decay=self.uncer_params["weight_decay"],
            )

            self.vis_uncertainty_online = self.uncer_params["vis_uncertainty_online"]

        # Setup queue object for gui communication
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.pause = False

        # Lazy-loaded person segmentation model reused by insertion gating
        # and the demo-only visual eraser.
        self.demo_eraser_model = None
        self.demo_eraser_preprocess = None
        self.gaussian_insertion_stats = {
            "full": 0,
            "masked": 0,
            "skipped": 0,
            "forced_seed": 0,
        }
        self.gaussian_insertion_events = []
        self.post_cleanup_stats = {}

    def run(self):
        """
        Trigger mapping process, get estimated pose and depth from tracking process,
        send continue signal to tracking process when the mapping of the current frame finishes.
        """
        # Initialize list to keep track of Keyframes
        # In short, for any idx "i",
        # self.video.timestamp[video_idx[i]] = self.frame_idxs[i]
        self.frame_idxs = []  # the indices of keyframes in the original frame sequence
        self.video_idxs = []  # keyframe numbering (I sometimes call it kf_idx)

        while True:
            if self.config['gui']:
                if self.q_vis2main.empty():
                    if self.pause:
                        continue
                else:
                    data_vis2main = self.q_vis2main.get()
                    self.pause = data_vis2main.flag_pause
                    if self.pause:
                        self.printer.print("You have paused the process", FontColor.MAPPER)
                        continue
                    else:
                        self.printer.print("You have resume the process", FontColor.MAPPER)

            frame_info = self.pipe.recv()
            frame_idx, video_idx = frame_info["timestamp"], frame_info["video_idx"]
            is_init, is_finished = frame_info["just_initialized"], frame_info["end"]

            if is_finished:
                self.printer.print("Done with Mapping and Tracking", FontColor.MAPPER)
                break

            if self.verbose:
                self.printer.print(f"\nMapping Frame {frame_idx} ...", FontColor.MAPPER)

            if is_init:
                self.printer.print("Initializing the mapping", FontColor.MAPPER)
                self.initialize_mapper(video_idx)
                self.pipe.send("continue")
                continue

            viewpoint, invalid = self._get_viewpoint(video_idx, frame_idx)

            if invalid:
                # Only happens when not using metric depth for tracking regularization
                self.printer.print("WARNING: Too few valid pixels from droid depth", FontColor.MAPPER)
                self.is_kf[video_idx] = False
                self.pipe.send("continue")
                continue  # too few valid pixels from droid depth
            
            # Update the map if depth/pose of any keyframe has been updated
            self._update_keyframes_from_frontend()
            self.frame_idxs.append(frame_idx)
            self.video_idxs.append(video_idx)

            # We need to render from the current pose to obtain the "n_touched" variable
            # which is used later on
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            curr_visibility = (render_pkg["n_touched"] > 0).long()

            # Always create kf
            self.cameras[video_idx] = viewpoint
            self.current_window, _ = self._add_to_window(
                video_idx,
                curr_visibility,
                self.occ_aware_visibility,
                self.current_window,
            )
            self.is_kf[video_idx] = True
            self.depth_dict[video_idx] = torch.tensor(viewpoint.depth).to(self.device)
            self.frame_count_log[video_idx] = 0

            self._insert_gaussians_for_keyframe(
                viewpoint,
                video_idx=video_idx,
                init=False,
            )

            opt_params = []
            for cam_idx in range(len(self.current_window)):
                if self.current_window[cam_idx] == 0:
                    # Do not add first frame for exposure optimization
                    continue
                viewpoint = self.cameras[self.current_window[cam_idx]]
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        "lr": 0.01,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.01,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
            self.keyframe_optimizers = torch.optim.Adam(opt_params)

            with Lock():
                if self.config['fast_mode']:
                    # We are in fast mode,
                    # update map and uncertainty MLP every 4 key frames
                    if video_idx % 4 == 0:
                        gaussian_split = self.map_opt_online(
                            self.current_window, iters=self.mapping_itr_num
                        )
                    else:
                        self._update_occ_aware_visibility(self.current_window)
                else:
                    gaussian_split = self.map_opt_online(
                        self.current_window, iters=self.mapping_itr_num
                    )

                if gaussian_split:
                    # do one more iteration after densify and prune
                    self.map_opt_online(self.current_window, iters=1)
            torch.cuda.empty_cache()

            if self.config['gui']:
                self._send_to_gui(video_idx)

            self.pipe.send("continue")

    """
    Utility functions
    """

    def _set_hyperparams(self):
        self.cameras_extent = 6.0
        mapping_config = self.config["mapping"]

        self.init_itr_num = mapping_config["Training"]["init_itr_num"]
        self.init_gaussian_update = mapping_config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = mapping_config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = mapping_config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = mapping_config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = mapping_config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = mapping_config["Training"][
            "gaussian_update_offset"
        ]
        self.gaussian_th = mapping_config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = mapping_config["Training"]["gaussian_reset"]
        self.size_threshold = mapping_config["Training"]["size_threshold"]
        self.window_size = mapping_config["Training"]["window_size"]

        self.save_dir = self.config["data"]["output"] + "/" + self.config["scene"]

        self.deform_gaussians = self.config["mapping"]["deform_gaussians"]
        self.online_plotting = self.config["mapping"]["online_plotting"]

    def _get_viewpoint(self, video_idx: int, frame_idx: int) -> Tuple[Camera, bool]:
        """
        Create and initialize a Camera object for a given frame.

        Args:
            video_idx (int): Index in the video class (keyframe idx).
            frame_idx (int): Index in the original frame sequences.

        Returns:
            Tuple[Camera, bool]: Initialized Camera object and an invalid flag.
        """
        # Load color image based on resolution configuration
        if self.config["mapping"]["full_resolution"]:
            color = (
                self.frame_reader.get_color_full_resol(frame_idx)
                .to(self.device)
                .squeeze()
            )
            load_feature_suffix = "full"
        else:
            color = self.frame_reader.get_color(frame_idx).to(self.device).squeeze()
            load_feature_suffix = ""

        # Load metric depth
        metric_depth = load_metric_depth(frame_idx, self.save_dir).to(self.device)

        # Load features if uncertainty-aware
        if self.uncertainty_aware:
            features = load_img_feature(
                frame_idx, self.save_dir, suffix=load_feature_suffix
            ).to(self.device)
        else:
            features = None

        # Get estimated depth and camera pose
        est_depth, est_w2c, invalid = self.get_w2c_and_depth(
            video_idx, frame_idx, metric_depth
        )

        # Prepare data dictionary for Camera initialization
        camera_data = {
            "idx": video_idx,
            "gt_color": color,
            "est_depth": est_depth.cpu().numpy(),
            "est_pose": est_w2c,
            "features": features,
        }

        # Initialize Camera object
        viewpoint = Camera.init_from_dataset(
            self.frame_reader,
            camera_data,
            self.projection_matrix,
            full_resol=self.config["mapping"]["full_resolution"],
        )

        # Update camera pose and compute gradient mask
        # The Camera class is based on MonoGS and
        # init_from_dataset function only updates the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        viewpoint.compute_grad_mask(self.config)

        return viewpoint, invalid

    def _update_keyframes_from_frontend(self):
        """
        Update keyframe information based on the latest frontend data.
        This includes updating camera poses, depths, and mapping points (deform gaussians)
        for all keyframes.
        """
        for keyframe_idx, frame_idx in zip(self.video_idxs, self.frame_idxs):
            # Get updated pose and depth
            if self.video.metric_depth_reg:
                c2w_updated = self.video.get_pose(keyframe_idx, self.device)
                w2c_updated = torch.linalg.inv(c2w_updated)
                depth_updated = None
                invalid = False
            else:
                metric_depth = load_metric_depth(frame_idx, self.save_dir).to(
                    self.device
                )
                depth_updated, w2c_updated, invalid = self.get_w2c_and_depth(
                    keyframe_idx, frame_idx, metric_depth
                )

            # Get old pose
            w2c_old = torch.eye(4, device=self.device)
            w2c_old[:3, :3] = self.cameras[keyframe_idx].R
            w2c_old[:3, 3] = self.cameras[keyframe_idx].T

            pose_unchanged = torch.allclose(w2c_old, w2c_updated, atol=1e-6)
            if pose_unchanged and depth_updated is None:
                continue

            # Update camera
            self.cameras[keyframe_idx].update_RT(
                w2c_updated[:3, :3], w2c_updated[:3, 3]
            )
            if depth_updated is not None:
                self.cameras[keyframe_idx].depth = depth_updated.cpu().numpy()
                self.depth_dict[keyframe_idx] = depth_updated

            # Update viewpoint if it exists
            if self.is_kf[keyframe_idx]:
                self.cameras[keyframe_idx].update_RT(
                    w2c_updated[:3, :3], w2c_updated[:3, 3]
                )
                if depth_updated is not None:
                    self.cameras[keyframe_idx].depth = depth_updated.cpu().numpy()

            # Update mapping parameters
            if self.deform_gaussians and self.is_kf[keyframe_idx]:
                if invalid or depth_updated is None:
                    self._update_mapping_points(
                        keyframe_idx,
                        w2c_updated,
                        w2c_old,
                        depth_updated,
                        self.depth_dict[keyframe_idx],
                        method="rigid",
                    )
                else:
                    self._update_mapping_points(
                        keyframe_idx,
                        w2c_updated,
                        w2c_old,
                        depth_updated,
                        self.depth_dict[keyframe_idx],
                    )

    def _update_mapping_points(
        self, frame_idx, w2c, w2c_old, depth, depth_old, method=None
    ):
        """Refer to splat-slam"""
        if method == "rigid":
            # just move the points according to their SE(3) transformation without updating depth
            frame_idxs = (
                self.gaussians.unique_kfIDs
            )  # idx which anchored the set of points
            frame_mask = frame_idxs == frame_idx  # global variable
            if frame_mask.sum() == 0:
                return
            # Retrieve current set of points to be deformed
            # But first we need to retrieve all mean locations and clone them
            means = self.gaussians.get_xyz.detach()
            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means[frame_mask], pix_ones), dim=1)
            means[frame_mask] = (transformation @ pts4.T).T[:, :3]
            # put the new means back to the optimizer
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(
                means, "xyz"
            )["xyz"]
            # transform the corresponding rotation matrices
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(
                transformation.expand_as(rots[frame_mask]), rots[frame_mask]
            )

            with torch.no_grad():
                self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(
                    rots, "rotation"
                )["rotation"]
        else:
            # Update pose and depth by projecting points into the pixel space to find updated correspondences.
            # This strategy also adjusts the scale of the gaussians to account for the distance change from the camera
            depth = depth.to(self.device)
            frame_idxs = (
                self.gaussians.unique_kfIDs
            )  # idx which anchored the set of points
            frame_mask = frame_idxs == frame_idx  # global variable
            if frame_mask.sum() == 0:
                return

            # Retrieve current set of points to be deformed
            means = self.gaussians.get_xyz.detach()[frame_mask]

            # Project the current means into the old camera to get the pixel locations
            pix_ones = torch.ones(means.shape[0], 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            pixel_locations = (self.intrinsics @ (w2c_old @ pts4.T)[:3, :]).T
            pixel_locations[:, 0] /= pixel_locations[:, 2]
            pixel_locations[:, 1] /= pixel_locations[:, 2]
            pixel_locations = pixel_locations[:, :2].long()
            height, width = depth.shape
            # Some pixels may project outside the viewing frustum.
            # Assign these pixels the depth of the closest border pixel
            pixel_locations[:, 0] = torch.clamp(
                pixel_locations[:, 0], min=0, max=width - 1
            )
            pixel_locations[:, 1] = torch.clamp(
                pixel_locations[:, 1], min=0, max=height - 1
            )

            # Extract the depth at those pixel locations from the new depth
            depth = depth[pixel_locations[:, 1], pixel_locations[:, 0]]
            depth_old = depth_old[pixel_locations[:, 1], pixel_locations[:, 0]]
            # Next, we can either move the points to the new pose and then adjust the
            # depth or the other way around.
            # Lets adjust the depth per point first
            # First we need to transform the global means into the old camera frame
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            means_cam = (w2c_old @ pts4.T).T[:, :3]

            rescale_scale = (1 + 1 / (means_cam[:, 2]) * (depth - depth_old)).unsqueeze(
                -1
            )  # shift
            # account for 0 depth values - then just do rigid deformation
            rigid_mask = torch.logical_or(depth == 0, depth_old == 0)
            rescale_scale[rigid_mask] = 1
            if (rescale_scale <= 0.0).sum() > 0:
                rescale_scale[rescale_scale <= 0.0] = 1

            rescale_mean = rescale_scale.repeat(1, 3)
            means_cam = rescale_mean * means_cam

            # Transform back means_cam to the world space
            pts4 = torch.cat((means_cam, pix_ones), dim=1)
            means = (torch.linalg.inv(w2c_old) @ pts4.T).T[:, :3]

            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pts4 = torch.cat((means, pix_ones), dim=1)
            means = (transformation @ pts4.T).T[:, :3]

            # reassign the new means of the frame mask to the self.gaussian object
            global_means = self.gaussians.get_xyz.detach()
            global_means[frame_mask] = means
            # print("mean nans: ", global_means.isnan().sum()/global_means.numel())
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(
                global_means, "xyz"
            )["xyz"]

            # update the rotation of the gaussians
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(
                transformation.expand_as(rots[frame_mask]), rots[frame_mask]
            )
            self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(
                rots, "rotation"
            )["rotation"]

            # Update the scale of the Gaussians
            scales = self.gaussians._scaling.detach()
            scales[frame_mask] = scales[frame_mask] + torch.log(rescale_scale)
            self.gaussians._scaling = self.gaussians.replace_tensor_to_optimizer(
                scales, "scaling"
            )["scaling"]

    def _update_occ_aware_visibility(self, current_window):
        self.occ_aware_visibility = {}
        for kf_idx in current_window:
            viewpoint = self.cameras[kf_idx]
            render_pkg = render(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
            )
            self.occ_aware_visibility[kf_idx] = (
                render_pkg["n_touched"] > 0
            ).long()

    def get_w2c_and_depth(self, video_idx, idx, mono_depth, print_info=False):
        est_frontend_depth, valid_depth_mask, c2w = self.video.get_depth_and_pose(
            video_idx, self.device
        )
        c2w = c2w.to(self.device)
        w2c = torch.linalg.inv(c2w)

        if self.video.metric_depth_reg:
            return est_frontend_depth, w2c, False

        # The following is only useful when no metric depth used for tracking regularization
        # Code is from Splat-SLAM
        if print_info:
            self.printer.print(
                f"valid depth number: {valid_depth_mask.sum().item()}, "
                f"valid depth ratio: {(valid_depth_mask.sum()/(valid_depth_mask.shape[0]*valid_depth_mask.shape[1])).item()}",
                FontColor.MAPPER
            )

        if valid_depth_mask.sum() < 100:
            invalid = True
            self.printer.print(
                f"Skip mapping frame {idx} at video idx {video_idx} because of not enough valid depth ({valid_depth_mask.sum()}).", FontColor.MAPPER
            )
        else:
            invalid = False

        est_frontend_depth[~valid_depth_mask] = 0
        if not invalid:
            mono_depth[mono_depth > 4 * mono_depth.mean()] = 0
            mono_depth = mono_depth.cpu().numpy()
            binary_image = (mono_depth > 0).astype(int)
            # Add padding around the binary_image to protect the borders
            iterations = 5
            padded_binary_image = np.pad(
                binary_image, pad_width=iterations, mode="constant", constant_values=1
            )
            structure = np.ones((3, 3), dtype=int)
            # Apply binary erosion with padding
            eroded_padded_image = binary_erosion(
                padded_binary_image, structure=structure, iterations=iterations
            )
            # Remove padding after erosion
            eroded_image = eroded_padded_image[
                iterations:-iterations, iterations:-iterations
            ]
            # set mono depth to zero at mask
            mono_depth[eroded_image == 0] = 0

            if (mono_depth == 0).sum() > 0:
                mono_depth = torch.from_numpy(
                    cv2.inpaint(
                        mono_depth,
                        (mono_depth == 0).astype(np.uint8),
                        inpaintRadius=3,
                        flags=cv2.INPAINT_NS,
                    )
                ).to(self.device)
            else:
                mono_depth = torch.from_numpy(mono_depth).to(self.device)

            valid_mask = (
                torch.from_numpy(eroded_image).to(self.device) * valid_depth_mask
            )  # new

            cur_wq = self.video.get_depth_scale_and_shift(
                video_idx, mono_depth, est_frontend_depth, valid_mask
            )
            mono_depth_wq = mono_depth * cur_wq[0] + cur_wq[1]

            est_frontend_depth[~valid_depth_mask] = mono_depth_wq[~valid_depth_mask]

        return est_frontend_depth, w2c, invalid

    def _add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        """Refer to MonoGS"""
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["mapping"]["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["mapping"]["Training"]
                else 0.4
            )
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.window_size:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame
    
    def _send_to_gui(self, video_idx):
        """Send data to the GUI for visualization.
        """
        viewpoint = self.cameras[video_idx]
        keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
        uncertainty_map = self.get_viewpoint_uncertainty_no_grad(viewpoint)
        uncertainty_map = uncertainty_map.cpu().squeeze(0).numpy()
        
        current_window_dict = {}
        current_window_dict[self.current_window[0]] = self.current_window[1:]
        keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                current_frame=viewpoint,
                gaussians=self.gaussians,
                gtcolor=viewpoint.original_image.squeeze(),
                gtdepth=viewpoint.depth,
                keyframes=keyframes,
                kf_window=current_window_dict,
                uncertainty=uncertainty_map,
            )
        )


    def initialize_mapper(self, cur_video_idx):
        self.printer.print("Resetting the mapper", FontColor.MAPPER)

        self.iteration_count = 0
        self.iterations_after_densify_or_reset = 0
        self.occ_aware_visibility = {}
        self.frame_count_log = {}
        self.current_window = []
        self.keyframe_optimizers = None

        # Keys are video_idx and value is boolean.
        # This is only useful in ablation study of no depth-regularization
        self.is_kf = {}

        # Create dictionary which stores the depth maps from the previous iteration
        # This depth is used during map deformation if we have missing pixels
        self.depth_dict = {}

        # Dictionary of Camera objects at the frame index
        # self.cameras contains all cameras.
        self.cameras = {}

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

        opt_params = []

        for video_idx in range(cur_video_idx + 1):
            frame_idx = int(self.video.timestamp[video_idx])
            self.frame_idxs.append(frame_idx)
            self.video_idxs.append(video_idx)

            viewpoint, invalid = self._get_viewpoint(video_idx, frame_idx)
            # Dictionary of Camera objects at the frame index
            self.cameras[video_idx] = viewpoint

            if invalid:
                # Only happens when not using metric depth for tracking regularization
                self.printer.print("WARNING: Too few valid pixels from droid depth", FontColor.MAPPER)
                self.is_kf[video_idx] = False
                continue  # too few valid pixels from droid depth

            # update the dictionaries
            self.depth_dict[video_idx] = torch.tensor(viewpoint.depth).to(self.device)
            self.is_kf[video_idx] = True
            self.frame_count_log[video_idx] = 0

            self._insert_gaussians_for_keyframe(
                viewpoint,
                video_idx=video_idx,
                init=True,
            )

            self.current_window.append(video_idx)

            # Do not add first frame for exposure optimization
            if video_idx != 0:
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        "lr": 0.01,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.01,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
        self.keyframe_optimizers = torch.optim.Adam(opt_params)

        self.initialize_map_opt()
        # Only keep the recent <self.window_size> number of keyframes in the window
        self.current_window = self.current_window[-self.window_size :]

        if self.config['gui']:
            self._send_to_gui(cur_video_idx)

    def refine_pose_non_key_frame(
        self, frame_idx: int, w2c_init: torch.Tensor, features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Optimize the pose for a non-key frame.

        Args:
            frame_idx (int): Camera index.
            w2c_init (torch.Tensor): Initial world-to-camera transformation.
            features (torch.Tensor, optional): Image features for uncertainty estimation.

        Returns:
            torch.Tensor: Refined world-to-camera transformation.
        """
        # We always use the downsampled image for tracking.
        # Full resolution only for the mapping even if it's activated.
        color = self.frame_reader.get_color(frame_idx).to(self.device).squeeze()

        data = {
            "idx": frame_idx,
            "gt_color": color.squeeze(),
            "est_depth": None,
            "est_pose": w2c_init,
            "features": features,
        }

        # Using uncertainty only when uncertainty-aware tracking is activated
        if self.video.uncertainty_aware:
            with torch.no_grad():
                uncer = self.uncer_network(features.to(color.device))
                uncer = torch.clip(uncer, min=0.1) + 1e-3
                uncer_resized = F.interpolate(
                    uncer.unsqueeze(0).unsqueeze(0),
                    size=color.shape[-2:],
                    mode="bilinear",
                ).squeeze()
                data_rate = 1 + 1 * map_utils.compute_bias_factor(
                    self.config["mapping"]["uncertainty_params"]["train_frac_fix"], 0.8
                )
                uncer_resized = (uncer_resized - 0.1) * data_rate + 0.1
        else:
            uncer_resized = None

        viewpoint = Camera.init_from_dataset(
            self.frame_reader,
            data,
            self.projection_matrix,
        )
        viewpoint.compute_grad_mask(self.config)
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        opt_params = [
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["mapping"]["Training"]["lr"]["cam_rot_delta"],
                "name": f"rot_{viewpoint.uid}",
            },
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["mapping"]["Training"]["lr"]["cam_trans_delta"],
                "name": f"trans_{viewpoint.uid}",
            },
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": f"exposure_a_{viewpoint.uid}",
            },
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": f"exposure_b_{viewpoint.uid}",
            },
        ]

        pose_optimizer = torch.optim.Adam(opt_params)
        tracking_itr_num = 100  # This number is taken from monoGS
        for _ in range(tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config["mapping"],
                image,
                depth,
                opacity,
                viewpoint,
                uncertainty=uncer_resized,
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if converged:
                break

        w2c_refined = torch.eye(4, device=viewpoint.R.device)
        w2c_refined[:3, :3] = viewpoint.R
        w2c_refined[:3, 3] = viewpoint.T

        return w2c_refined

    """
    Map Optimization functions (init, online, final_refine)
    """
    def initialize_map_opt(self):
        viewpoint_stack = []
        viewpoint_id_stack = []
        for kf_idx in self.current_window:
            viewpoint = self.cameras[kf_idx]
            viewpoint_stack.append(viewpoint)
            viewpoint_id_stack.append(kf_idx)

        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            self.iterations_after_densify_or_reset += 1
            # randomly select a viewpoint from the first K keyframes
            cam_idx = np.random.choice(range(len(viewpoint_stack)))
            viewpoint = viewpoint_stack[cam_idx]
            kf_idx = viewpoint_id_stack[cam_idx]

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            if not self.uncertainty_aware:
                loss_init = get_loss_mapping(
                    self.config["mapping"],
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    initialization=True,
                )
            else:
                train_frac = self.uncer_params["train_frac_fix"]
                ssim_frac = self.uncer_params["train_frac_fix"]
                if self.config["mapping"]["full_resolution"]:
                    depth = F.interpolate(
                        depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
                    ).squeeze(0)
                (
                    temporal_prior,
                    temporal_static_evidence,
                    temporal_params,
                ) = self._get_temporal_mapping_state(viewpoint, image.device)
                current_uncertainty, loss_init = get_loss_mapping_uncertainty(
                    self.config["mapping"],
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    self.uncer_network,
                    train_frac,
                    ssim_frac,
                    initialization=True,
                    temporal_prior=temporal_prior,
                    temporal_static_evidence=temporal_static_evidence,
                    temporal_params=temporal_params,
                )

                stride = self.config["mapping"]["uncertainty_params"]["reg_stride"]
                feature_buffer = [
                    viewpoint.features[::stride, ::stride].to(device=image.device),
                ]
                uncer_buffer = [
                    current_uncertainty[::stride, ::stride].unsqueeze(-1),
                ]
                loss_init += self.config["mapping"]["uncertainty_params"][
                    "reg_mult"
                ] * map_utils.compute_dino_regularization_loss(
                    uncer_buffer, feature_buffer
                )

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_init += 10 * isotropic_loss.mean()

            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                    self.iterations_after_densify_or_reset = 0

                if self.iteration_count == self.init_gaussian_reset:
                    self.gaussians.reset_opacity()
                    self.iterations_after_densify_or_reset = 0

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                if self.uncertainty_aware:
                    self.uncer_optimizer.step()
                    self.uncer_optimizer.zero_grad()

                self.frame_count_log[kf_idx] += 1

            self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
        self.printer.print("Initialized map", FontColor.MAPPER)

        # online plotting
        if self.online_plotting:
            plot_dir = self.save_dir + "/online_plots"
            suffix = "_init"
            for cur_idx in self.current_window:
                self.save_fig_everything(cur_idx, plot_dir, suffix)

        return render_pkg

    def map_opt_online(self, current_window, iters=1):
        if len(current_window) == 0:
            raise ValueError("No keyframes in the current window")

        # Online plot before optimization
        cur_idx = current_window[np.array(current_window).argmax()]
        if self.online_plotting:
            plot_dir = self.save_dir + "/online_plots"
            suffix = "_before_opt"
            self.save_fig_everything(cur_idx, plot_dir, suffix)

        viewpoint_stack, viewpoint_kf_idx_stack = [], []
        for kf_idx, viewpoint in self.cameras.items():
            if self.is_kf[kf_idx]:
                viewpoint_stack.append(viewpoint)
                viewpoint_kf_idx_stack.append(kf_idx)

        # We set the current frame to be chosen by prob at least 50%
        # and the rest frame evenly distribute the remaining prob
        cur_window_prob = 0.5
        prob = np.full(
            len(viewpoint_stack),
            (1 - cur_window_prob)
            * iters
            / (len(viewpoint_stack) - len(current_window)),
        )
        assert viewpoint_kf_idx_stack[-1] == cur_idx
        if len(current_window) <= len(viewpoint_stack) / 2.0:
            for view_idx in range(len(viewpoint_kf_idx_stack)):
                kf_idx = viewpoint_kf_idx_stack[view_idx]
                if kf_idx in current_window:
                    prob[view_idx] = cur_window_prob * iters / (len(current_window))
        prob /= prob.sum()

        for cur_iter in range(iters):
            self.iteration_count += 1
            self.iterations_after_densify_or_reset += 1

            loss_mapping = 0

            cam_idx = np.random.choice(np.arange(len(viewpoint_stack)), p=prob)
            viewpoint = viewpoint_stack[cam_idx]
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            if self.config["mapping"]["full_resolution"]:
                depth = F.interpolate(
                    depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
                ).squeeze(0)
            if not self.uncertainty_aware:
                loss_mapping += get_loss_mapping(
                    self.config["mapping"], image, depth, viewpoint, opacity
                )
            else:
                train_frac = self.uncer_params["train_frac_fix"]
                ssim_frac = self.uncer_params["train_frac_fix"]
                (
                    temporal_prior,
                    temporal_static_evidence,
                    temporal_params,
                ) = self._get_temporal_mapping_state(viewpoint, image.device)

                (
                    current_uncertainty,
                    current_loss_mapping,
                ) = get_loss_mapping_uncertainty(
                    self.config["mapping"],
                    (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b,
                    depth,
                    viewpoint,
                    opacity,
                    self.uncer_network,
                    train_frac,
                    ssim_frac,
                    freeze_uncertainty_loss=self.iterations_after_densify_or_reset < 20,
                    temporal_prior=temporal_prior,
                    temporal_static_evidence=temporal_static_evidence,
                    temporal_params=temporal_params,
                )
                loss_mapping += current_loss_mapping

                # Dino_regularization loss
                if self.iterations_after_densify_or_reset >= 20:
                    stride = self.config["mapping"]["uncertainty_params"]["reg_stride"]
                    reg_multi = self.config["mapping"]["uncertainty_params"]["reg_mult"]

                    viewpoint = viewpoint_stack[cam_idx]
                    feature_buffer = [
                        viewpoint_stack[reg_cam_idx].features.to(device=image.device)
                        for reg_cam_idx in range(
                            max(0, cam_idx - 2), min(len(viewpoint_stack), cam_idx + 3)
                        )
                    ]
                    feat_dim = feature_buffer[0].shape[-1]
                    feature_buffer = torch.stack(feature_buffer).view(-1, feat_dim)
                    num_samples = feature_buffer.shape[0] // (stride ** 4)
                    sampled_feature = feature_buffer[
                        torch.randperm(feature_buffer.shape[0])[:num_samples]
                    ].unsqueeze(0)
                    sampled_uncer = self.uncer_network(sampled_feature)
                    loss_mapping += (
                        reg_multi
                        * map_utils.compute_dino_regularization_loss(
                            sampled_uncer, sampled_feature
                        )
                    )

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()
            gaussian_split = False
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                if cur_iter == iters - 1:
                    self._update_occ_aware_visibility(current_window)

                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True
                    self.iterations_after_densify_or_reset = 0
                    self.printer.print("Densify and prune the Gaussians", FontColor.MAPPER)

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    self.printer.print(
                        "Resetting the opacity of non-visible Gaussians",
                        FontColor.MAPPER,
                    )
                    self.gaussians.reset_opacity_nonvisible([visibility_filter])
                    gaussian_split = True
                    self.iterations_after_densify_or_reset = 0

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                if self.uncertainty_aware:
                    self.uncer_optimizer.step()
                    self.uncer_optimizer.zero_grad()

            self.frame_count_log[viewpoint_kf_idx_stack[cam_idx]] += 1

        # Online plotting
        if self.online_plotting:
            plot_dir = self.save_dir + "/online_plots"
            suffix = "_after_opt"
            self.save_fig_everything(cur_idx, plot_dir, suffix)

        # online plot the uncertainty mask
        if self.vis_uncertainty_online:
            self._vis_uncertainty_mask_all()
        return gaussian_split

    def final_refine(self, iters=26000):
        self.printer.print("Starting final refinement", FontColor.MAPPER)

        # Do final update of depths and poses
        self._update_keyframes_from_frontend()

        random_viewpoint_stack, random_viewpoint_kf_idx_stack = [], []
        for kf_idx, viewpoint in self.cameras.items():
            if self.is_kf[kf_idx]:
                random_viewpoint_stack.append(viewpoint)
                random_viewpoint_kf_idx_stack.append(kf_idx)

        for _ in tqdm(range(iters)):
            self.iteration_count += 1
            self.iterations_after_densify_or_reset += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            uncer_buffer, feature_buffer = [], []

            rand_idx = np.random.choice(range(len(random_viewpoint_stack)))
            random_viewpoint_kf_idxs = []

            viewpoint = random_viewpoint_stack[rand_idx]
            random_viewpoint_kf_idxs.append(random_viewpoint_kf_idx_stack[rand_idx])
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            if self.config["mapping"]["full_resolution"]:
                depth = F.interpolate(
                    depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
                ).squeeze(0)
            if not self.uncertainty_aware:
                loss_mapping += get_loss_mapping(
                    self.config["mapping"], image, depth, viewpoint, opacity
                )
            else:
                train_frac = self.uncer_params["train_frac_fix"]
                ssim_frac = self.uncer_params["train_frac_fix"]
                (
                    temporal_prior,
                    temporal_static_evidence,
                    temporal_params,
                ) = self._get_temporal_mapping_state(viewpoint, image.device)

                (
                    current_uncertainty,
                    loss_mapping_this_frame,
                ) = get_loss_mapping_uncertainty(
                    self.config["mapping"],
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    self.uncer_network,
                    train_frac,
                    ssim_frac,
                    freeze_uncertainty_loss=self.iterations_after_densify_or_reset
                    < 200,
                    temporal_prior=temporal_prior,
                    temporal_static_evidence=temporal_static_evidence,
                    temporal_params=temporal_params,
                )
                loss_mapping += loss_mapping_this_frame

                stride = self.config["mapping"]["uncertainty_params"]["reg_stride"]
                uncer_buffer.append(
                    current_uncertainty[::stride, ::stride].unsqueeze(-1)
                )
                feature_buffer.append(
                    viewpoint.features[::stride, ::stride].to(device=image.device)
                )

            if self.uncertainty_aware and self.iterations_after_densify_or_reset >= 200:
                stride = self.config["mapping"]["uncertainty_params"]["reg_stride"]
                reg_multi = self.config["mapping"]["uncertainty_params"]["reg_mult"]
                viewpoint = random_viewpoint_stack[rand_idx]
                feature_buffer = [
                    random_viewpoint_stack[reg_cam_idx].features.to(device=image.device)
                    for reg_cam_idx in range(
                        max(0, rand_idx - 2),
                        min(len(random_viewpoint_stack), rand_idx + 3),
                    )
                ]
                feat_dim = feature_buffer[0].shape[-1]
                feature_buffer = torch.stack(feature_buffer).view(-1, feat_dim)
                num_samples = feature_buffer.shape[0] // (stride ** 4)
                sampled_feature = feature_buffer[
                    torch.randperm(feature_buffer.shape[0])[:num_samples]
                ].unsqueeze(0)
                sampled_uncer = self.uncer_network(sampled_feature)
                loss_mapping += reg_multi * map_utils.compute_dino_regularization_loss(
                    sampled_uncer, sampled_feature
                )

            viewspace_point_tensor_acm.append(viewspace_point_tensor)
            visibility_filter_acm.append(visibility_filter)
            radii_acm.append(radii)
            n_touched_acm.append(n_touched)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                # Optimize the exposure compensation
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                if self.uncertainty_aware:
                    self.uncer_optimizer.step()
                    self.uncer_optimizer.zero_grad()

            for kf_idx in random_viewpoint_kf_idxs:
                self.frame_count_log[kf_idx] += 1

        self._run_post_refine_dynamic_gaussian_cleanup()
        self._run_post_refine_dynamic_point_prune()
        self._run_post_refine_dynamic_point_opacity_decay()
        self._persist_post_cleanup_stats()
        self._report_gaussian_insertion_stats()

        if self.vis_uncertainty_online:
            self._vis_uncertainty_mask_all(is_final=True)
        
        if self.config['gui']:
            self._send_to_gui(self.current_window[np.array(self.current_window).argmax()])

        self.printer.print("Final refinement done", FontColor.MAPPER)

    """
    Viusalization functions
    """

    @torch.no_grad()
    def _get_uncertainty_ssim_loss_vis(
        self, gt_image: torch.Tensor, rendered_img: torch.Tensor, opacity: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates and visualizes the SSIM-based uncertainty loss.
        Note that this is the Nerf-on-the-go version of SSIM. Only use this for visualization.

        Parameters:
        - gt_image (torch.Tensor): Ground truth image.
        - rendered_img (torch.Tensor): Rendered image from the model.
        - opacity (torch.Tensor): Opacity values for the rendered image.

        Returns:
        - torch.Tensor: Nerf-on-the-go version of SSIM-based uncertainty loss map (scaled).
        """
        ssim_frac = self.uncer_params["train_frac_fix"]
        l, c, s = map_utils.compute_ssim_components(
            gt_image,
            rendered_img,
            window_size=self.config["mapping"]["uncertainty_params"][
                "ssim_window_size"
            ],
        )
        ssim_loss = torch.clip(
            (100 + 900 * map_utils.compute_bias_factor(ssim_frac, 0.8))
            * (1 - l)
            * (1 - s)
            * (1 - c),
            max=5.0,
        )
        median_filter = MedianPool2d(
            kernel_size=self.config["mapping"]["uncertainty_params"][
                "ssim_median_filter_size"
            ],
            stride=1,
            padding=0,
            same=True,
        )
        ssim_loss = (
            median_filter(ssim_loss.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        )
        opacity_th = self.config["mapping"]["uncertainty_params"][
            "opacity_th_for_uncer_loss"
        ]
        ssim_loss[opacity < opacity_th] = 0

        return ssim_loss

    @torch.no_grad()
    def _get_temporal_mapping_state(self, viewpoint: Camera, device):
        if not getattr(self.video, "temporal_uncertainty_aware", False):
            return None, None, None

        temporal_prior, temporal_static_evidence = self.video.get_temporal_mapping_state(
            viewpoint.uid, viewpoint.features.shape[:2], device
        )
        return temporal_prior, temporal_static_evidence, self.video.temporal_params

    @torch.no_grad()
    def _get_dynamic_cleanup_mask(
        self, viewpoint: Camera, target_shape: Tuple[int, int], device
    ) -> Optional[torch.Tensor]:
        temporal_params = getattr(self.video, "temporal_params", {})
        if not (
            temporal_params.get("post_cleanup_activate", False)
            or temporal_params.get("post_cleanup_prune_activate", False)
            or temporal_params.get("post_cleanup_opacity_decay_activate", False)
        ):
            return None

        dynamic_score = torch.zeros(target_shape, device=device, dtype=torch.float32)
        dynamic_prior = self._get_temporal_dynamic_score(viewpoint)
        temporal_seed_mask = None
        if dynamic_prior is not None:
            if tuple(dynamic_prior.shape[-2:]) != tuple(target_shape):
                dynamic_prior = map_utils.resample_tensor_to_shape(
                    dynamic_prior, target_shape
                )
            dynamic_score = torch.maximum(dynamic_score, dynamic_prior)
            support_threshold = float(
                temporal_params.get("post_cleanup_temporal_support_threshold", 0.2)
            )
            temporal_seed_mask = dynamic_prior > support_threshold

        uncertainty_threshold = float(
            temporal_params.get("post_cleanup_uncertainty_threshold", -1.0)
        )
        if uncertainty_threshold > 0:
            uncertainty_map = self.get_viewpoint_uncertainty_no_grad(viewpoint).squeeze(0)
            if tuple(uncertainty_map.shape[-2:]) != tuple(target_shape):
                uncertainty_map = map_utils.resample_tensor_to_shape(
                    uncertainty_map, target_shape
                )
            uncertainty_scale = float(
                temporal_params.get("post_cleanup_uncertainty_scale", 1.0)
            )
            uncertainty_score = torch.clamp(
                (uncertainty_map - uncertainty_threshold)
                / max(uncertainty_scale, 1e-6),
                min=0.0,
                max=1.0,
            )
            support_radius = int(
                temporal_params.get("post_cleanup_temporal_support_radius", 0)
            )
            if support_radius > 0 and temporal_seed_mask is not None:
                temporal_support_mask = (
                    temporal_utils.max_pool_spatial_map(
                        temporal_seed_mask.float(), support_radius
                    )
                    > 0.5
                )
                uncertainty_score = uncertainty_score * temporal_support_mask.float()
            dynamic_score = torch.maximum(dynamic_score, uncertainty_score)

        dilation_radius = int(temporal_params.get("post_cleanup_dilation_radius", 0))
        if dilation_radius > 0:
            dynamic_score = temporal_utils.max_pool_spatial_map(
                dynamic_score, dilation_radius
            )

        dynamic_threshold = float(
            temporal_params.get(
                "post_cleanup_dynamic_threshold",
                temporal_params.get("insertion_dynamic_threshold", 0.45),
            )
        )
        dynamic_mask = dynamic_score > dynamic_threshold

        valid_depth = torch.as_tensor(
            viewpoint.depth, device=device, dtype=torch.float32
        ) > 0
        if tuple(valid_depth.shape[-2:]) != tuple(target_shape):
            valid_depth = (
                F.interpolate(
                    valid_depth.float().view(1, 1, *valid_depth.shape[-2:]),
                    size=target_shape,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                > 0
            )
        dynamic_mask = dynamic_mask & valid_depth

        valid_depth_count = valid_depth.float().sum().item()
        if valid_depth_count < 1.0:
            return None

        max_mask_ratio = float(temporal_params.get("post_cleanup_max_mask_ratio", 0.5))
        if max_mask_ratio > 0:
            dynamic_ratio_valid = (dynamic_mask & valid_depth).float().sum().item()
            dynamic_ratio_valid /= valid_depth_count
            if dynamic_ratio_valid > max_mask_ratio:
                valid_scores = dynamic_score[valid_depth]
                if valid_scores.numel() > 0:
                    quantile_target = max(0.0, min(1.0, 1.0 - max_mask_ratio))
                    adaptive_threshold = float(
                        torch.quantile(valid_scores, quantile_target).item()
                    )
                    dynamic_threshold = max(dynamic_threshold, adaptive_threshold)
                    dynamic_mask = (dynamic_score > dynamic_threshold) & valid_depth

        min_mask_ratio = float(temporal_params.get("post_cleanup_min_mask_ratio", 0.01))
        if dynamic_mask.float().mean().item() < min_mask_ratio:
            return None

        return dynamic_mask

    @torch.no_grad()
    def _erode_binary_mask(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        inverse_mask = (~mask).float()
        dilated_inverse = temporal_utils.max_pool_spatial_map(inverse_mask, radius)
        return dilated_inverse < 0.5

    @torch.no_grad()
    def _get_temporal_dynamic_score(self, viewpoint: Camera) -> Optional[torch.Tensor]:
        if not getattr(self.video, "temporal_uncertainty_aware", False):
            return None

        dynamic_prior, _ = self.video.get_temporal_mapping_state(
            viewpoint.uid,
            (viewpoint.image_height, viewpoint.image_width),
            self.device,
        )
        return dynamic_prior

    @torch.no_grad()
    def _get_temporal_prior_score(
        self, viewpoint: Camera, target_shape: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        if (
            not getattr(self.video, "temporal_uncertainty_aware", False)
            or getattr(self.video, "temporal_priors", None) is None
        ):
            return None

        idx = int(viewpoint.uid)
        if idx < 0 or idx >= self.video.counter.value:
            return None

        temporal_prior = self.video.temporal_priors[idx].to(self.device)
        if tuple(temporal_prior.shape[-2:]) != tuple(target_shape):
            temporal_prior = map_utils.resample_tensor_to_shape(
                temporal_prior, target_shape
            )
        return temporal_prior

    @torch.no_grad()
    def _insert_gaussians_for_keyframe(
        self, viewpoint: Camera, video_idx: int, init: bool
    ) -> None:
        insertion_keep_mask, skip_insertion = self._get_gaussian_insertion_keep_mask(
            viewpoint
        )
        has_existing_gaussians = self.gaussians.get_xyz.shape[0] > 0
        action = "masked"

        if skip_insertion and has_existing_gaussians:
            action = "skipped"
            self.gaussian_insertion_stats["skipped"] += 1
            self._record_gaussian_insertion_event(
                video_idx=video_idx,
                action=action,
                insertion_keep_mask=insertion_keep_mask,
                skip_insertion=skip_insertion,
                had_existing_gaussians=has_existing_gaussians,
            )
            self.printer.print(
                f"Skip Gaussian insertion for keyframe {video_idx} due to fail-closed dynamic gating.",
                FontColor.MAPPER,
            )
            return

        if skip_insertion and not has_existing_gaussians:
            action = "forced_seed"
            self.gaussian_insertion_stats["forced_seed"] += 1
            self.printer.print(
                f"Force masked seed insertion for keyframe {video_idx} to avoid an empty map.",
                FontColor.MAPPER,
            )
        elif insertion_keep_mask is None:
            action = "full"
            self.gaussian_insertion_stats["full"] += 1
        else:
            action = "masked"
            self.gaussian_insertion_stats["masked"] += 1

        self._record_gaussian_insertion_event(
            video_idx=video_idx,
            action=action,
            insertion_keep_mask=insertion_keep_mask,
            skip_insertion=skip_insertion,
            had_existing_gaussians=has_existing_gaussians,
        )
        self.gaussians.extend_from_pcd_seq(
            viewpoint,
            kf_id=video_idx,
            init=init,
            depthmap=viewpoint.depth,
            pixel_mask=insertion_keep_mask,
        )

    @torch.no_grad()
    def _record_gaussian_insertion_event(
        self,
        video_idx: int,
        action: str,
        insertion_keep_mask: Optional[torch.Tensor],
        skip_insertion: bool,
        had_existing_gaussians: bool,
    ) -> None:
        keep_ratio = None
        mask_ratio = None
        if insertion_keep_mask is not None:
            keep_ratio = float(insertion_keep_mask.float().mean().item())
            mask_ratio = 1.0 - keep_ratio

        self.gaussian_insertion_events.append(
            {
                "video_idx": int(video_idx),
                "action": action,
                "skip_insertion": bool(skip_insertion),
                "had_existing_gaussians": bool(had_existing_gaussians),
                "keep_ratio": keep_ratio,
                "mask_ratio": mask_ratio,
            }
        )

    @torch.no_grad()
    def _persist_gaussian_insertion_stats(self) -> None:
        total_actions = sum(self.gaussian_insertion_stats.values())
        if total_actions <= 0:
            return

        payload = {
            "summary": {
                **self.gaussian_insertion_stats,
                "total": total_actions,
            },
            "events": self.gaussian_insertion_events,
        }
        output_path = os.path.join(self.save_dir, "gaussian_insertion_stats.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            self.printer.print(
                f"Failed to save Gaussian insertion stats to {output_path}: {exc}",
                FontColor.ERROR,
            )

    @torch.no_grad()
    def _persist_post_cleanup_stats(self) -> None:
        if not self.post_cleanup_stats:
            return

        output_path = os.path.join(self.save_dir, "post_cleanup_stats.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.post_cleanup_stats, f, indent=2)
        except OSError as exc:
            self.printer.print(
                f"Failed to save post-cleanup stats to {output_path}: {exc}",
                FontColor.ERROR,
            )

    @torch.no_grad()
    def _report_gaussian_insertion_stats(self) -> None:
        total_actions = sum(self.gaussian_insertion_stats.values())
        if total_actions <= 0:
            return

        self._persist_gaussian_insertion_stats()
        self.printer.print(
            "Gaussian insertion summary: "
            f"full={self.gaussian_insertion_stats['full']}, "
            f"masked={self.gaussian_insertion_stats['masked']}, "
            f"skipped={self.gaussian_insertion_stats['skipped']}, "
            f"forced_seed={self.gaussian_insertion_stats['forced_seed']}",
            FontColor.MAPPER,
        )

    @torch.no_grad()
    def _get_gaussian_insertion_keep_mask(
        self, viewpoint: Camera
    ) -> Tuple[Optional[torch.Tensor], bool]:
        temporal_params = getattr(self.video, "temporal_params", {})
        if not temporal_params.get("insertion_mask_activate", False):
            return None, False

        dynamic_prior = self._get_temporal_dynamic_score(viewpoint)
        if dynamic_prior is None:
            return None, False

        dynamic_threshold = float(
            temporal_params.get("insertion_dynamic_threshold", 0.4)
        )
        dynamic_score = dynamic_prior.clone()
        target_shape = tuple(dynamic_score.shape[-2:])

        if temporal_params.get("insertion_use_temporal_prior_support", False):
            temporal_prior = self._get_temporal_prior_score(viewpoint, target_shape)
            if temporal_prior is not None:
                temporal_prior_gain = float(
                    temporal_params.get("insertion_temporal_prior_gain", 1.0)
                )
                dynamic_score = torch.maximum(
                    dynamic_score,
                    torch.clamp(temporal_prior_gain * temporal_prior, min=0.0, max=1.0),
                )

        insertion_uncertainty_threshold = float(
            temporal_params.get("insertion_uncertainty_threshold", -1.0)
        )
        if insertion_uncertainty_threshold > 0:
            uncertainty_map = self.get_viewpoint_uncertainty_no_grad(viewpoint).squeeze(0)
            uncertainty_scale = float(
                temporal_params.get("insertion_uncertainty_scale", 1.0)
            )
            uncertainty_score = torch.clamp(
                (uncertainty_map - insertion_uncertainty_threshold)
                / max(uncertainty_scale, 1e-6),
                min=0.0,
                max=1.0,
            )
            dynamic_score = torch.maximum(dynamic_score, uncertainty_score)

        person_mask = None
        person_mask_mode = str(
            temporal_params.get("insertion_person_mask_mode", "intersect")
        )
        if temporal_params.get("insertion_use_person_segmentation", False):
            person_mask = self._get_person_mask_from_config(
                viewpoint,
                target_shape=target_shape,
                device=self.device,
                prefix="insertion",
            )
            if person_mask is not None and person_mask_mode != "intersect":
                person_min_dynamic = float(
                    temporal_params.get(
                        "insertion_person_min_dynamic", dynamic_threshold
                    )
                )
                dynamic_score = torch.maximum(
                    dynamic_score,
                    person_mask.float() * person_min_dynamic,
                )

        dilation_radius = int(temporal_params.get("insertion_dilation_radius", 0))
        if dilation_radius > 0:
            dynamic_score = temporal_utils.max_pool_spatial_map(
                dynamic_score, dilation_radius
            )

        dynamic_mask = dynamic_score > dynamic_threshold

        valid_depth = torch.as_tensor(
            viewpoint.depth, device=self.device, dtype=torch.float32
        ) > 0
        valid_depth_count = valid_depth.float().sum().item()
        if valid_depth_count < 1.0:
            return None, False

        if person_mask is not None and person_mask_mode == "intersect":
            dynamic_mask = dynamic_mask & person_mask

        raw_dynamic_ratio_valid = (dynamic_mask & valid_depth).float().sum().item()
        raw_dynamic_ratio_valid /= valid_depth_count

        fail_closed_skip = bool(
            temporal_params.get("insertion_fail_closed_skip", False)
        )
        fail_closed_ratio = float(
            temporal_params.get("insertion_fail_closed_ratio", -1.0)
        )
        if (
            fail_closed_skip
            and fail_closed_ratio > 0
            and raw_dynamic_ratio_valid > fail_closed_ratio
        ):
            return (~dynamic_mask).float(), True

        max_mask_ratio = float(temporal_params.get("insertion_max_mask_ratio", 0.25))
        if (
            max_mask_ratio > 0
            and temporal_params.get("insertion_enable_adaptive_threshold", True)
        ):
            if raw_dynamic_ratio_valid > max_mask_ratio:
                valid_scores = dynamic_score[valid_depth]
                if valid_scores.numel() > 0:
                    quantile_target = max(0.0, min(1.0, 1.0 - max_mask_ratio))
                    adaptive_threshold = float(
                        torch.quantile(valid_scores, quantile_target).item()
                    )
                    dynamic_threshold = max(dynamic_threshold, adaptive_threshold)
                    dynamic_mask = dynamic_score > dynamic_threshold
                    if person_mask is not None and person_mask_mode == "intersect":
                        dynamic_mask = dynamic_mask & person_mask

        dynamic_ratio = (dynamic_mask & valid_depth).float().sum().item()
        dynamic_ratio /= valid_depth_count
        if dynamic_ratio <= 0.0:
            return None, False

        kept_depth_ratio = ((~dynamic_mask) & valid_depth).float().sum().item()
        kept_depth_ratio /= valid_depth_count
        min_remaining_depth_ratio = float(
            temporal_params.get("insertion_min_remaining_depth_ratio", 0.2)
        )
        if (
            kept_depth_ratio < min_remaining_depth_ratio
            and temporal_params.get(
                "insertion_enable_min_remaining_adaptive_threshold", True
            )
        ):
            valid_scores = dynamic_score[valid_depth]
            if valid_scores.numel() > 0:
                safe_quantile = max(0.0, min(1.0, min_remaining_depth_ratio))
                adaptive_threshold = float(
                    torch.quantile(valid_scores, safe_quantile).item()
                )
                dynamic_threshold = max(dynamic_threshold, adaptive_threshold)
                dynamic_mask = dynamic_score > dynamic_threshold
                if person_mask is not None and person_mask_mode == "intersect":
                    dynamic_mask = dynamic_mask & person_mask
                kept_depth_ratio = ((~dynamic_mask) & valid_depth).float().sum().item()
                kept_depth_ratio /= valid_depth_count

        if kept_depth_ratio < min_remaining_depth_ratio:
            keep_mask = (~dynamic_mask).float()
            if fail_closed_skip:
                return keep_mask, True
            return None, False

        return (~dynamic_mask).float(), False

    def _run_post_refine_dynamic_gaussian_cleanup(self) -> None:
        temporal_params = getattr(self.video, "temporal_params", {})
        if not temporal_params.get("post_cleanup_activate", False):
            return

        cleanup_iters = int(temporal_params.get("post_cleanup_iters", 0))
        if cleanup_iters <= 0:
            return

        viewpoint_stack = [
            viewpoint
            for kf_idx, viewpoint in self.cameras.items()
            if self.is_kf.get(kf_idx, False)
        ]
        if len(viewpoint_stack) == 0:
            return

        opacity_weight = float(temporal_params.get("post_cleanup_opacity_weight", 3.0))
        static_l1_weight = float(temporal_params.get("post_cleanup_static_l1_weight", 0.05))
        optimize_exposure = bool(
            temporal_params.get("post_cleanup_optimize_exposure", False)
        )
        opacity_only = bool(temporal_params.get("post_cleanup_opacity_only", False))

        frozen_param_tensors = []
        if opacity_only:
            for param_tensor in [
                self.gaussians._xyz,
                self.gaussians._features_dc,
                self.gaussians._features_rest,
                self.gaussians._scaling,
                self.gaussians._rotation,
            ]:
                if param_tensor.requires_grad:
                    param_tensor.requires_grad_(False)
                    frozen_param_tensors.append(param_tensor)

        self.printer.print(
            f"Starting post-refine dynamic cleanup ({cleanup_iters} iters)",
            FontColor.MAPPER,
        )

        used_iters = 0
        avg_dynamic_opacity = 0.0
        avg_static_l1 = 0.0

        try:
            for _ in tqdm(range(cleanup_iters)):
                self.iteration_count += 1

                viewpoint = viewpoint_stack[np.random.choice(range(len(viewpoint_stack)))]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                image = render_pkg["render"]
                opacity = render_pkg["opacity"]
                if opacity.dim() == 3:
                    opacity = opacity.squeeze(0)

                if self.config["mapping"]["full_resolution"]:
                    image = F.interpolate(
                        image.unsqueeze(0),
                        size=viewpoint.depth.shape,
                        mode="bicubic",
                    ).squeeze(0)
                    opacity = F.interpolate(
                        opacity.unsqueeze(0).unsqueeze(0),
                        size=viewpoint.depth.shape,
                        mode="bilinear",
                    ).squeeze(0).squeeze(0)

                target_shape = tuple(opacity.shape[-2:])
                cleanup_mask = self._get_dynamic_cleanup_mask(
                    viewpoint, target_shape, image.device
                )
                if cleanup_mask is None:
                    continue

                gt_image = viewpoint.original_image.to(image.device)
                if tuple(gt_image.shape[-2:]) != target_shape:
                    gt_image = F.interpolate(
                        gt_image.unsqueeze(0),
                        size=target_shape,
                        mode="bilinear",
                    ).squeeze(0)

                rendered_exposed = torch.exp(viewpoint.exposure_a) * image + viewpoint.exposure_b
                static_mask = (~cleanup_mask).float()
                
                roi_radius = int(temporal_params.get("post_cleanup_roi_radius", 0))
                if roi_radius > 0:
                    roi_mask = temporal_utils.max_pool_spatial_map(cleanup_mask.float(), roi_radius) > 0.5
                    static_mask = static_mask * roi_mask.float()
                    
                static_mask_3 = static_mask.unsqueeze(0).expand_as(rendered_exposed)
                static_denom = static_mask_3.sum().clamp(min=1.0)
                static_l1 = (
                    torch.abs(rendered_exposed - gt_image) * static_mask_3
                ).sum() / static_denom

                dynamic_opacity = opacity[cleanup_mask].mean()
                cleanup_loss = opacity_weight * dynamic_opacity + static_l1_weight * static_l1

                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if optimize_exposure and self.keyframe_optimizers is not None:
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                cleanup_loss.backward()

                with torch.no_grad():
                    self.gaussians.optimizer.step()
                    self.gaussians.update_learning_rate(self.iteration_count)
                    if optimize_exposure and self.keyframe_optimizers is not None:
                        self.keyframe_optimizers.step()

                used_iters += 1
                avg_dynamic_opacity += float(dynamic_opacity.detach().item())
                avg_static_l1 += float(static_l1.detach().item())
        finally:
            for param_tensor in frozen_param_tensors:
                param_tensor.requires_grad_(True)

        if used_iters > 0:
            avg_dynamic_opacity /= used_iters
            avg_static_l1 /= used_iters
            self.printer.print(
                "Post-refine cleanup done: "
                f"iters={used_iters}, avg_dyn_opacity={avg_dynamic_opacity:.4f}, "
                f"avg_static_l1={avg_static_l1:.4f}",
                FontColor.MAPPER,
            )
        else:
            self.printer.print(
                "Post-refine cleanup skipped (no eligible dynamic masks).",
                FontColor.MAPPER,
            )

    @torch.no_grad()
    def _collect_post_cleanup_dynamic_point_candidates(
        self,
        activation_key: str,
        min_opacity_key: str,
        erosion_radius_key: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        temporal_params = getattr(self.video, "temporal_params", {})
        if not temporal_params.get(activation_key, False):
            return None, None, 0

        xyz = self.gaussians.get_xyz.detach()
        if xyz.shape[0] == 0:
            return None, None, 0

        viewpoint_stack = [
            viewpoint
            for kf_idx, viewpoint in self.cameras.items()
            if self.is_kf.get(kf_idx, False)
        ]
        if len(viewpoint_stack) == 0:
            return None, None, 0

        gaussian_opacity = self.gaussians.get_opacity.detach().squeeze(-1)
        anchor_kf_ids = self.gaussians.unique_kfIDs.to(
            device=xyz.device, dtype=torch.long
        )
        candidate_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)

        min_opacity = float(temporal_params.get(min_opacity_key, 0.05))
        erosion_radius = int(temporal_params.get(erosion_radius_key, 1))
        used_viewpoints = 0

        for viewpoint in viewpoint_stack:
            target_shape = (viewpoint.image_height, viewpoint.image_width)
            dynamic_mask = self._get_dynamic_cleanup_mask(
                viewpoint, target_shape, xyz.device
            )
            if dynamic_mask is None:
                continue

            dynamic_mask = self._erode_binary_mask(dynamic_mask, erosion_radius)
            if dynamic_mask.float().mean().item() <= 0.0:
                continue

            anchored_points_mask = anchor_kf_ids == int(viewpoint.uid)
            if not bool(anchored_points_mask.any().item()):
                continue

            point_indices = torch.nonzero(
                anchored_points_mask, as_tuple=False
            ).squeeze(1)
            points_world = xyz[point_indices]

            cam_rot = viewpoint.R.to(xyz.device, dtype=torch.float32)
            cam_trans = viewpoint.T.to(xyz.device, dtype=torch.float32)
            points_cam = (cam_rot @ points_world.T).T + cam_trans
            depth = points_cam[:, 2]
            safe_depth = torch.clamp(depth, min=1e-6)

            x_proj = viewpoint.fx * (points_cam[:, 0] / safe_depth) + viewpoint.cx
            y_proj = viewpoint.fy * (points_cam[:, 1] / safe_depth) + viewpoint.cy
            x_pix = torch.round(x_proj).long()
            y_pix = torch.round(y_proj).long()

            valid_proj = (
                (depth > 1e-3)
                & (x_pix >= 0)
                & (x_pix < target_shape[1])
                & (y_pix >= 0)
                & (y_pix < target_shape[0])
            )
            if not bool(valid_proj.any().item()):
                continue

            dynamic_hit = torch.zeros_like(valid_proj)
            dynamic_hit[valid_proj] = dynamic_mask[
                y_pix[valid_proj], x_pix[valid_proj]
            ]
            if min_opacity > 0:
                dynamic_hit = dynamic_hit & (gaussian_opacity[point_indices] > min_opacity)

            if bool(dynamic_hit.any().item()):
                candidate_mask[point_indices[dynamic_hit]] = True
                used_viewpoints += 1

        return candidate_mask, gaussian_opacity, used_viewpoints

    @torch.no_grad()
    def _get_viewpoint_depth_tensor(
        self, viewpoint: Camera, target_shape: Tuple[int, int], device
    ) -> torch.Tensor:
        depth_tensor = torch.as_tensor(viewpoint.depth, device=device, dtype=torch.float32)
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.squeeze(0)
        if tuple(depth_tensor.shape[-2:]) != tuple(target_shape):
            depth_tensor = F.interpolate(
                depth_tensor.view(1, 1, *depth_tensor.shape[-2:]),
                size=target_shape,
                mode="nearest",
            ).squeeze(0).squeeze(0)
        return depth_tensor

    @torch.no_grad()
    def _collect_post_cleanup_gaussian_evidence_candidates(
        self,
        min_opacity_key: str,
        erosion_radius_key: str,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        int,
        Optional[dict],
    ]:
        temporal_params = getattr(self.video, "temporal_params", {})
        xyz = self.gaussians.get_xyz.detach()
        if xyz.shape[0] == 0:
            return None, None, None, 0, None

        viewpoint_stack = [
            viewpoint
            for kf_idx, viewpoint in self.cameras.items()
            if self.is_kf.get(kf_idx, False)
        ]
        viewpoint_stack = sorted(viewpoint_stack, key=lambda viewpoint: int(viewpoint.uid))
        if len(viewpoint_stack) == 0:
            empty_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
            empty_scores = torch.zeros(xyz.shape[0], dtype=torch.float32, device=xyz.device)
            gaussian_opacity = self.gaussians.get_opacity.detach().squeeze(-1)
            self.gaussians.reset_dynamic_evidence()
            return empty_mask, gaussian_opacity, empty_scores, 0, {
                "mode": "gaussian_evidence",
                "status": "no_viewpoints",
                "active_count": 0,
                "observed_count": 0,
                "candidate_count_before_cap": 0,
            }

        gaussian_opacity = self.gaussians.get_opacity.detach().squeeze(-1)
        min_opacity = float(temporal_params.get(min_opacity_key, 0.05))
        view_min_opacity = float(
            temporal_params.get("post_cleanup_evidence_view_min_opacity", min_opacity)
        )
        erosion_radius = int(temporal_params.get(erosion_radius_key, 1))
        depth_abs_tolerance = float(
            temporal_params.get("post_cleanup_evidence_depth_abs_tolerance", 0.08)
        )
        depth_rel_tolerance = float(
            temporal_params.get("post_cleanup_evidence_depth_rel_tolerance", 0.05)
        )
        min_observations = int(
            temporal_params.get("post_cleanup_evidence_min_observations", 2)
        )
        min_dynamic_hits = int(
            temporal_params.get("post_cleanup_evidence_min_dynamic_hits", 2)
        )
        dynamic_ratio_threshold = float(
            temporal_params.get("post_cleanup_evidence_dynamic_ratio_threshold", 0.6)
        )
        max_static_ratio = float(
            temporal_params.get("post_cleanup_evidence_max_static_ratio", 0.35)
        )
        dynamic_margin = float(
            temporal_params.get("post_cleanup_evidence_dynamic_margin", 1.0)
        )

        active_indices = torch.nonzero(
            gaussian_opacity > view_min_opacity, as_tuple=False
        ).squeeze(1)
        dynamic_hits = torch.zeros(xyz.shape[0], device=xyz.device, dtype=torch.float32)
        static_hits = torch.zeros_like(dynamic_hits)
        observation_hits = torch.zeros_like(dynamic_hits)
        dynamic_weighted_hits = torch.zeros_like(dynamic_hits)
        static_weighted_hits = torch.zeros_like(dynamic_hits)
        observation_weighted_hits = torch.zeros_like(dynamic_hits)

        if active_indices.numel() == 0:
            empty_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
            empty_scores = torch.zeros(xyz.shape[0], dtype=torch.float32, device=xyz.device)
            self.gaussians.set_dynamic_evidence(
                dynamic_hits=dynamic_hits,
                static_hits=static_hits,
                observation_hits=observation_hits,
                dynamic_weighted_hits=dynamic_weighted_hits,
                static_weighted_hits=static_weighted_hits,
                observation_weighted_hits=observation_weighted_hits,
            )
            return empty_mask, gaussian_opacity, empty_scores, 0, {
                "mode": "gaussian_evidence",
                "status": "no_visible_seed_gaussians",
                "active_count": 0,
                "observed_count": 0,
                "candidate_count_before_cap": 0,
                "view_min_opacity": view_min_opacity,
            }

        points_world = xyz[active_indices]
        used_viewpoints = 0

        for viewpoint in viewpoint_stack:
            target_shape = (viewpoint.image_height, viewpoint.image_width)
            dynamic_mask = self._get_dynamic_cleanup_mask(
                viewpoint, target_shape, xyz.device
            )
            if dynamic_mask is None:
                continue

            dynamic_mask = self._erode_binary_mask(dynamic_mask, erosion_radius)
            if dynamic_mask.float().mean().item() <= 0.0:
                continue

            depth_tensor = self._get_viewpoint_depth_tensor(
                viewpoint, target_shape, xyz.device
            )
            cam_rot = viewpoint.R.to(xyz.device, dtype=torch.float32)
            cam_trans = viewpoint.T.to(xyz.device, dtype=torch.float32)
            points_cam = (cam_rot @ points_world.T).T + cam_trans
            depth = points_cam[:, 2]
            safe_depth = torch.clamp(depth, min=1e-6)

            x_proj = viewpoint.fx * (points_cam[:, 0] / safe_depth) + viewpoint.cx
            y_proj = viewpoint.fy * (points_cam[:, 1] / safe_depth) + viewpoint.cy
            x_pix = torch.round(x_proj).long()
            y_pix = torch.round(y_proj).long()

            valid_proj = (
                (depth > 1e-3)
                & (x_pix >= 0)
                & (x_pix < target_shape[1])
                & (y_pix >= 0)
                & (y_pix < target_shape[0])
            )
            if not bool(valid_proj.any().item()):
                continue

            visible_indices = active_indices[valid_proj]
            depth_valid = depth[valid_proj]
            x_valid = x_pix[valid_proj]
            y_valid = y_pix[valid_proj]
            sampled_depth = depth_tensor[y_valid, x_valid]
            depth_tolerance = torch.clamp(
                sampled_depth * depth_rel_tolerance, min=depth_abs_tolerance
            )
            depth_consistent = (sampled_depth > 0) & (
                torch.abs(depth_valid - sampled_depth) <= depth_tolerance
            )
            if not bool(depth_consistent.any().item()):
                continue

            visible_indices = visible_indices[depth_consistent]
            x_visible = x_valid[depth_consistent]
            y_visible = y_valid[depth_consistent]
            visible_dynamic = dynamic_mask[y_visible, x_visible]
            static_visible = ~visible_dynamic

            observation_hits[visible_indices] += 1.0
            observation_weighted_hits[visible_indices] += 1.0
            if bool(visible_dynamic.any().item()):
                dynamic_indices = visible_indices[visible_dynamic]
                dynamic_hits[dynamic_indices] += 1.0
                dynamic_weighted_hits[dynamic_indices] += 1.0
            if bool(static_visible.any().item()):
                static_indices = visible_indices[static_visible]
                static_hits[static_indices] += 1.0
                static_weighted_hits[static_indices] += 1.0
            used_viewpoints += 1

        self.gaussians.set_dynamic_evidence(
            dynamic_hits=dynamic_hits,
            static_hits=static_hits,
            observation_hits=observation_hits,
            dynamic_weighted_hits=dynamic_weighted_hits,
            static_weighted_hits=static_weighted_hits,
            observation_weighted_hits=observation_weighted_hits,
        )

        observed_mask = observation_hits > 0
        weighted_observed_mask = observation_weighted_hits > 0
        raw_dynamic_ratio = torch.zeros_like(dynamic_hits)
        raw_static_ratio = torch.zeros_like(static_hits)
        dynamic_ratio = torch.zeros_like(dynamic_hits)
        static_ratio = torch.zeros_like(static_hits)
        raw_dynamic_ratio[observed_mask] = (
            dynamic_hits[observed_mask] / observation_hits[observed_mask]
        )
        raw_static_ratio[observed_mask] = (
            static_hits[observed_mask] / observation_hits[observed_mask]
        )
        dynamic_ratio[weighted_observed_mask] = (
            dynamic_weighted_hits[weighted_observed_mask]
            / observation_weighted_hits[weighted_observed_mask]
        )
        static_ratio[weighted_observed_mask] = (
            static_weighted_hits[weighted_observed_mask]
            / observation_weighted_hits[weighted_observed_mask]
        )

        base_candidate_mask = observation_hits >= float(max(min_observations, 1))
        base_candidate_mask = base_candidate_mask & (
            dynamic_hits >= float(max(min_dynamic_hits, 1))
        )
        base_candidate_mask = base_candidate_mask & weighted_observed_mask
        if min_opacity > 0:
            base_candidate_mask = base_candidate_mask & (gaussian_opacity > min_opacity)

        candidate_mask = base_candidate_mask & (dynamic_ratio >= dynamic_ratio_threshold)
        if max_static_ratio >= 0:
            candidate_mask = candidate_mask & (static_ratio <= max_static_ratio)
        if dynamic_margin > 0:
            candidate_mask = candidate_mask & (
                (dynamic_hits - static_hits) >= dynamic_margin
            )

        candidate_scores = dynamic_weighted_hits + dynamic_ratio + 0.1 * gaussian_opacity
        observed_count = int(observed_mask.sum().item())
        candidate_count = int(candidate_mask.sum().item())
        avg_observations = (
            float(observation_hits[observed_mask].mean().item()) if observed_count > 0 else 0.0
        )
        avg_dynamic_ratio = (
            float(dynamic_ratio[candidate_mask].mean().item()) if candidate_count > 0 else 0.0
        )
        avg_raw_dynamic_ratio = (
            float(raw_dynamic_ratio[candidate_mask].mean().item())
            if candidate_count > 0
            else 0.0
        )
        avg_dynamic_hits = (
            float(dynamic_hits[candidate_mask].mean().item()) if candidate_count > 0 else 0.0
        )
        avg_weighted_dynamic_hits = (
            float(dynamic_weighted_hits[candidate_mask].mean().item())
            if candidate_count > 0
            else 0.0
        )
        evidence_summary = {
            "mode": "gaussian_evidence",
            "status": "collected",
            "active_count": int(active_indices.numel()),
            "observed_count": observed_count,
            "candidate_count_before_cap": candidate_count,
            "used_viewpoints": int(used_viewpoints),
            "view_min_opacity": view_min_opacity,
            "min_observations": int(max(min_observations, 1)),
            "min_dynamic_hits": int(max(min_dynamic_hits, 1)),
            "dynamic_ratio_threshold": dynamic_ratio_threshold,
            "max_static_ratio": max_static_ratio,
            "dynamic_margin": dynamic_margin,
            "depth_abs_tolerance": depth_abs_tolerance,
            "depth_rel_tolerance": depth_rel_tolerance,
            "avg_observations_per_observed": avg_observations,
            "avg_dynamic_ratio_candidate": avg_dynamic_ratio,
            "avg_raw_dynamic_ratio_candidate": avg_raw_dynamic_ratio,
            "avg_dynamic_hits_candidate": avg_dynamic_hits,
            "avg_weighted_dynamic_hits_candidate": avg_weighted_dynamic_hits,
        }
        return (
            candidate_mask,
            gaussian_opacity,
            candidate_scores,
            used_viewpoints,
            evidence_summary,
        )

    @torch.no_grad()
    def _run_post_refine_dynamic_point_prune(self) -> None:
        temporal_params = getattr(self.video, "temporal_params", {})
        if not temporal_params.get("post_cleanup_prune_activate", False):
            return

        self.printer.print(
            "Starting post-refine dynamic point prune",
            FontColor.MAPPER,
        )

        prune_mask, gaussian_opacity, used_viewpoints = (
            self._collect_post_cleanup_dynamic_point_candidates(
                activation_key="post_cleanup_prune_activate",
                min_opacity_key="post_cleanup_prune_min_opacity",
                erosion_radius_key="post_cleanup_prune_erosion_radius",
            )
        )
        if prune_mask is None or gaussian_opacity is None:
            return

        candidate_count = int(prune_mask.sum().item())
        if candidate_count == 0:
            self.post_cleanup_stats["point_prune"] = {
                "enabled": True,
                "status": "no_dynamic_candidates",
                "candidate_count": 0,
                "used_viewpoints": int(used_viewpoints),
            }
            self.printer.print(
                "Post-refine point prune skipped (no dynamic candidates).",
                FontColor.MAPPER,
            )
            return

        total_count = int(prune_mask.shape[0])
        max_ratio = float(temporal_params.get("post_cleanup_prune_max_ratio", 0.08))
        max_count = int(max_ratio * total_count) if max_ratio > 0 else candidate_count
        if max_count > 0 and candidate_count > max_count:
            candidate_indices = torch.nonzero(prune_mask, as_tuple=False).squeeze(1)
            candidate_opacity = gaussian_opacity[candidate_indices]
            topk_indices = torch.topk(candidate_opacity, k=max_count).indices
            limited_mask = torch.zeros_like(prune_mask)
            limited_mask[candidate_indices[topk_indices]] = True
            prune_mask = limited_mask

        prune_count = int(prune_mask.sum().item())
        if prune_count <= 0:
            self.post_cleanup_stats["point_prune"] = {
                "enabled": True,
                "status": "skipped_after_ratio_cap",
                "candidate_count": candidate_count,
                "prune_count": 0,
                "used_viewpoints": int(used_viewpoints),
            }
            self.printer.print(
                "Post-refine point prune skipped after ratio cap.",
                FontColor.MAPPER,
            )
            return

        self.gaussians.prune_points(prune_mask)
        self.post_cleanup_stats["point_prune"] = {
            "enabled": True,
            "status": "applied",
            "candidate_count": candidate_count,
            "prune_count": prune_count,
            "total_count": total_count,
            "used_viewpoints": int(used_viewpoints),
        }
        self.printer.print(
            "Post-refine point prune done: "
            f"removed={prune_count}/{total_count}, viewpoints={used_viewpoints}",
            FontColor.MAPPER,
        )

    @torch.no_grad()
    def _run_post_refine_dynamic_point_opacity_decay(self) -> None:
        temporal_params = getattr(self.video, "temporal_params", {})
        if not temporal_params.get("post_cleanup_opacity_decay_activate", False):
            return

        self.printer.print(
            "Starting post-refine dynamic point opacity decay",
            FontColor.MAPPER,
        )

        decay_mode = str(
            temporal_params.get("post_cleanup_opacity_decay_mode", "anchor_mask")
        )
        evidence_summary = None
        candidate_scores = None
        if decay_mode == "gaussian_evidence":
            (
                decay_mask,
                gaussian_opacity,
                candidate_scores,
                used_viewpoints,
                evidence_summary,
            ) = self._collect_post_cleanup_gaussian_evidence_candidates(
                min_opacity_key="post_cleanup_opacity_decay_min_opacity",
                erosion_radius_key="post_cleanup_opacity_decay_erosion_radius",
            )
        else:
            decay_mask, gaussian_opacity, used_viewpoints = (
                self._collect_post_cleanup_dynamic_point_candidates(
                    activation_key="post_cleanup_opacity_decay_activate",
                    min_opacity_key="post_cleanup_opacity_decay_min_opacity",
                    erosion_radius_key="post_cleanup_opacity_decay_erosion_radius",
                )
            )
            candidate_scores = gaussian_opacity
        if decay_mask is None or gaussian_opacity is None:
            return

        candidate_count = int(decay_mask.sum().item())
        if candidate_count == 0:
            payload = {
                "enabled": True,
                "mode": decay_mode,
                "status": "no_dynamic_candidates",
                "candidate_count": 0,
                "used_viewpoints": int(used_viewpoints),
            }
            if evidence_summary is not None:
                payload["gaussian_evidence"] = evidence_summary
            self.post_cleanup_stats["opacity_decay"] = payload
            self.printer.print(
                "Post-refine opacity decay skipped (no dynamic candidates).",
                FontColor.MAPPER,
            )
            return

        total_count = int(decay_mask.shape[0])
        max_ratio = float(
            temporal_params.get("post_cleanup_opacity_decay_max_ratio", 0.04)
        )
        max_count = int(max_ratio * total_count) if max_ratio > 0 else candidate_count
        if max_count > 0 and candidate_count > max_count:
            candidate_indices = torch.nonzero(decay_mask, as_tuple=False).squeeze(1)
            ranking_scores = (
                candidate_scores[candidate_indices]
                if candidate_scores is not None
                else gaussian_opacity[candidate_indices]
            )
            topk_indices = torch.topk(ranking_scores, k=max_count).indices
            limited_mask = torch.zeros_like(decay_mask)
            limited_mask[candidate_indices[topk_indices]] = True
            decay_mask = limited_mask

        decay_count = int(decay_mask.sum().item())
        if decay_count <= 0:
            payload = {
                "enabled": True,
                "mode": decay_mode,
                "status": "skipped_after_ratio_cap",
                "candidate_count": candidate_count,
                "attenuated_count": 0,
                "used_viewpoints": int(used_viewpoints),
            }
            if evidence_summary is not None:
                payload["gaussian_evidence"] = evidence_summary
            self.post_cleanup_stats["opacity_decay"] = payload
            self.printer.print(
                "Post-refine opacity decay skipped after ratio cap.",
                FontColor.MAPPER,
            )
            return

        decay_factor = float(
            temporal_params.get("post_cleanup_opacity_decay_factor", 0.45)
        )
        opacity_floor = float(
            temporal_params.get("post_cleanup_opacity_decay_floor", 0.03)
        )
        selected_indices = torch.nonzero(decay_mask, as_tuple=False).squeeze(1)
        avg_before = float(gaussian_opacity[selected_indices].mean().item())
        self.gaussians.attenuate_opacity(
            decay_mask,
            decay_factor=decay_factor,
            min_opacity=opacity_floor,
        )
        gaussian_opacity_after = self.gaussians.get_opacity.detach().squeeze(-1)
        avg_after = float(gaussian_opacity_after[selected_indices].mean().item())
        payload = {
            "enabled": True,
            "mode": decay_mode,
            "status": "applied",
            "candidate_count": candidate_count,
            "attenuated_count": decay_count,
            "total_count": total_count,
            "used_viewpoints": int(used_viewpoints),
            "decay_factor": decay_factor,
            "opacity_floor": opacity_floor,
            "avg_opacity_before": avg_before,
            "avg_opacity_after": avg_after,
        }
        if evidence_summary is not None:
            payload["gaussian_evidence"] = evidence_summary
        self.post_cleanup_stats["opacity_decay"] = payload
        self.printer.print(
            "Post-refine opacity decay done: "
            f"mode={decay_mode}, attenuated={decay_count}/{total_count}, "
            f"avg_opacity={avg_before:.4f}->{avg_after:.4f}, "
            f"viewpoints={used_viewpoints}",
            FontColor.MAPPER,
        )

    @torch.no_grad()
    def _get_dynamic_visual_eraser_mask(
        self, viewpoint: Camera, target_shape: Tuple[int, int], device
    ) -> Optional[torch.Tensor]:
        temporal_params = getattr(self.video, "temporal_params", {})
        if not temporal_params.get("demo_dynamic_eraser_activate", False):
            return None

        dynamic_score = torch.zeros(target_shape, device=device, dtype=torch.float32)
        dynamic_prior = self._get_temporal_dynamic_score(viewpoint)
        temporal_seed_mask = None
        if dynamic_prior is not None:
            if tuple(dynamic_prior.shape[-2:]) != tuple(target_shape):
                dynamic_prior = map_utils.resample_tensor_to_shape(
                    dynamic_prior, target_shape
                )
            dynamic_score = torch.maximum(dynamic_score, dynamic_prior)
            support_threshold = float(
                temporal_params.get("demo_dynamic_eraser_temporal_support_threshold", 0.2)
            )
            temporal_seed_mask = dynamic_prior > support_threshold

        uncertainty_threshold = float(
            temporal_params.get("demo_dynamic_eraser_uncertainty_threshold", -1.0)
        )
        if uncertainty_threshold > 0:
            uncertainty_map = self.get_viewpoint_uncertainty_no_grad(viewpoint).squeeze(0)
            if tuple(uncertainty_map.shape[-2:]) != tuple(target_shape):
                uncertainty_map = map_utils.resample_tensor_to_shape(
                    uncertainty_map, target_shape
                )
            uncertainty_scale = float(
                temporal_params.get("demo_dynamic_eraser_uncertainty_scale", 1.0)
            )
            uncertainty_score = torch.clamp(
                (uncertainty_map - uncertainty_threshold) / max(uncertainty_scale, 1e-6),
                min=0.0,
                max=1.0,
            )
            support_radius = int(
                temporal_params.get("demo_dynamic_eraser_temporal_support_radius", 0)
            )
            if support_radius > 0 and temporal_seed_mask is not None:
                temporal_support_mask = (
                    temporal_utils.max_pool_spatial_map(
                        temporal_seed_mask.float(), support_radius
                    )
                    > 0.5
                )
                uncertainty_score = uncertainty_score * temporal_support_mask.float()
            dynamic_score = torch.maximum(dynamic_score, uncertainty_score)

        dilation_radius = int(
            temporal_params.get("demo_dynamic_eraser_dilation_radius", 0)
        )
        if dilation_radius > 0:
            dynamic_score = temporal_utils.max_pool_spatial_map(
                dynamic_score, dilation_radius
            )

        dynamic_threshold = float(
            temporal_params.get(
                "demo_dynamic_eraser_dynamic_threshold",
                temporal_params.get("insertion_dynamic_threshold", 0.4),
            )
        )
        dynamic_mask = dynamic_score > dynamic_threshold

        valid_depth = torch.as_tensor(
            viewpoint.depth, device=device, dtype=torch.float32
        ) > 0
        if tuple(valid_depth.shape[-2:]) != tuple(target_shape):
            valid_depth = (
                F.interpolate(
                    valid_depth.float().view(1, 1, *valid_depth.shape[-2:]),
                    size=target_shape,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                > 0
            )
        dynamic_mask = dynamic_mask & valid_depth

        valid_depth_count = valid_depth.float().sum().item()
        if valid_depth_count < 1.0:
            return None

        max_mask_ratio = float(
            temporal_params.get("demo_dynamic_eraser_max_mask_ratio", 0.55)
        )
        if max_mask_ratio > 0:
            dynamic_ratio_valid = (dynamic_mask & valid_depth).float().sum().item()
            dynamic_ratio_valid /= valid_depth_count
            if dynamic_ratio_valid > max_mask_ratio:
                valid_scores = dynamic_score[valid_depth]
                if valid_scores.numel() > 0:
                    quantile_target = max(0.0, min(1.0, 1.0 - max_mask_ratio))
                    adaptive_threshold = float(
                        torch.quantile(valid_scores, quantile_target).item()
                    )
                    dynamic_mask = (dynamic_score > max(dynamic_threshold, adaptive_threshold)) & valid_depth

        min_mask_ratio = float(
            temporal_params.get("demo_dynamic_eraser_min_mask_ratio", 0.008)
        )
        if dynamic_mask.float().mean().item() < min_mask_ratio:
            return None

        if temporal_params.get("demo_dynamic_eraser_use_person_segmentation", False):
            person_mask = self._get_demo_person_mask(viewpoint, target_shape, device)
            if person_mask is not None:
                person_gate_mask = person_mask
                if dynamic_prior is not None:
                    person_prior_threshold = float(
                        temporal_params.get(
                            "demo_dynamic_eraser_person_prior_threshold", 0.25
                        )
                    )
                    person_gate_mask = person_gate_mask & (
                        dynamic_prior > person_prior_threshold
                    )

                overlap_numer = (dynamic_mask & person_gate_mask).float().sum().item()
                overlap_denom = dynamic_mask.float().sum().item()
                overlap_ratio = overlap_numer / max(overlap_denom, 1.0)
                min_overlap_ratio = float(
                    temporal_params.get("demo_dynamic_eraser_min_person_overlap_ratio", 0.05)
                )
                if overlap_ratio >= min_overlap_ratio:
                    dynamic_mask = dynamic_mask & person_gate_mask

        if dynamic_mask.float().mean().item() < min_mask_ratio:
            return None

        return dynamic_mask

    @torch.no_grad()
    def _load_demo_dynamic_eraser_model(self):
        if self.demo_eraser_model is not None and self.demo_eraser_preprocess is not None:
            return
        weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        self.demo_eraser_preprocess = weights.transforms()
        self.demo_eraser_model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=weights
        ).to(self.device)
        self.demo_eraser_model.eval()

    @torch.no_grad()
    def _get_person_mask_from_config(
        self,
        viewpoint: Camera,
        target_shape: Tuple[int, int],
        device,
        prefix: str,
    ) -> Optional[torch.Tensor]:
        temporal_params = getattr(self.video, "temporal_params", {})
        self._load_demo_dynamic_eraser_model()

        gt_image = viewpoint.original_image.detach().cpu().permute(1, 2, 0).numpy()
        gt_u8 = np.clip(gt_image * 255.0, 0, 255).astype(np.uint8)
        if gt_u8.size == 0:
            return None

        input_tensor = self.demo_eraser_preprocess(Image.fromarray(gt_u8)).unsqueeze(0).to(
            self.device
        )
        logits = self.demo_eraser_model(input_tensor)["out"]
        logits = F.interpolate(
            logits, size=target_shape, mode="bilinear", align_corners=False
        )

        person_class_id = int(temporal_params.get(f"{prefix}_person_class_id", 15))
        confidence_threshold = float(
            temporal_params.get(f"{prefix}_person_confidence_threshold", 0.35)
        )
        person_prob = torch.softmax(logits, dim=1)[:, person_class_id : person_class_id + 1]
        person_mask = person_prob.squeeze(0).squeeze(0) > confidence_threshold

        dilation_radius = int(temporal_params.get(f"{prefix}_person_dilation_radius", 2))
        if dilation_radius > 0:
            person_mask = (
                temporal_utils.max_pool_spatial_map(person_mask.float(), dilation_radius)
                > 0.5
            )

        erosion_radius = int(temporal_params.get(f"{prefix}_person_erosion_radius", 0))
        if erosion_radius > 0:
            person_mask = self._erode_binary_mask(person_mask, erosion_radius)

        if person_mask.float().mean().item() <= 0:
            return None

        return person_mask.to(device=device)

    @torch.no_grad()
    def _get_demo_person_mask(
        self, viewpoint: Camera, target_shape: Tuple[int, int], device
    ) -> Optional[torch.Tensor]:
        return self._get_person_mask_from_config(
            viewpoint,
            target_shape=target_shape,
            device=device,
            prefix="demo_dynamic_eraser",
        )

    @torch.no_grad()
    def _apply_visual_dynamic_eraser(
        self,
        viewpoint: Camera,
        rendered_img: torch.Tensor,
        rendered_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_params = getattr(self.video, "temporal_params", {})
        if not temporal_params.get("demo_dynamic_eraser_activate", False):
            return rendered_img, rendered_depth

        target_shape = tuple(rendered_img.shape[-2:])
        dynamic_mask = self._get_dynamic_visual_eraser_mask(
            viewpoint, target_shape, rendered_img.device
        )
        if dynamic_mask is None:
            return rendered_img, rendered_depth

        inpaint_radius = int(temporal_params.get("demo_dynamic_eraser_inpaint_radius", 3))
        if inpaint_radius <= 0:
            return rendered_img, rendered_depth

        mask_np = (dynamic_mask.detach().cpu().numpy().astype(np.uint8)) * 255
        if mask_np.max() <= 0:
            return rendered_img, rendered_depth

        pred_np = (
            rendered_img.detach().cpu().permute(1, 2, 0).numpy() * 255.0
        ).clip(0, 255).astype(np.uint8)
        inpaint_method = str(
            temporal_params.get("demo_dynamic_eraser_inpaint_method", "ns")
        ).lower()
        inpaint_flag = cv2.INPAINT_NS if inpaint_method == "ns" else cv2.INPAINT_TELEA
        inpaint_np = cv2.inpaint(pred_np, mask_np, inpaint_radius, inpaint_flag)

        feather_sigma = float(
            temporal_params.get("demo_dynamic_eraser_feather_sigma", 1.5)
        )
        if feather_sigma > 0:
            alpha = cv2.GaussianBlur(
                (mask_np.astype(np.float32) / 255.0),
                ksize=(0, 0),
                sigmaX=feather_sigma,
                sigmaY=feather_sigma,
            )
            alpha = np.clip(alpha, 0.0, 1.0)[..., None]
            pred_np = (
                pred_np.astype(np.float32) * (1.0 - alpha)
                + inpaint_np.astype(np.float32) * alpha
            ).clip(0, 255).astype(np.uint8)
        else:
            pred_np = inpaint_np
        rendered_img = (
            torch.from_numpy(pred_np.astype(np.float32) / 255.0)
            .to(rendered_img.device)
            .permute(2, 0, 1)
        )

        if temporal_params.get("demo_dynamic_eraser_inpaint_depth", True):
            depth_map = rendered_depth.squeeze(0)
            depth_np = depth_map.detach().cpu().numpy().astype(np.float32)
            depth_np = cv2.inpaint(depth_np, mask_np, inpaint_radius, cv2.INPAINT_NS)
            rendered_depth = (
                torch.from_numpy(depth_np)
                .to(device=rendered_depth.device, dtype=rendered_depth.dtype)
                .unsqueeze(0)
            )

        return rendered_img.clamp(0.0, 1.0), rendered_depth

    @torch.no_grad()
    def get_viewpoint_uncertainty_no_grad(self, viewpoint: Camera) -> torch.Tensor:
        """
        Compute the uncertainty for a given viewpoint without gradient computation.
        """
        if getattr(self.video, "temporal_uncertainty_aware", False):
            return self.video.get_uncertainty_map(
                viewpoint.uid,
                (viewpoint.image_height, viewpoint.image_width),
                self.device,
                squared=True,
            )

        features = viewpoint.features.to(self.device)
        with Lock():
            uncertainty = self.uncer_network(features)

        # Process uncertainty values
        uncertainty = torch.clip(uncertainty, min=0.1) + 1e-3
        target_shape = (viewpoint.image_height, viewpoint.image_width)
        uncertainty_resized = map_utils.resample_tensor_to_shape(
            uncertainty, target_shape
        )

        # Apply data rate adjustment, the same as how we calculate the loss function
        train_frac = self.uncer_params["train_frac_fix"]
        data_rate = 1 + map_utils.compute_bias_factor(train_frac, 0.8)
        uncertainty_adjusted = (uncertainty_resized - 0.1) * data_rate + 0.1

        return uncertainty_adjusted ** 2

    @torch.no_grad()
    def save_fig_everything(
        self,
        keyframe_idx: int,
        plot_dir: str,
        suffix: str = "",
        depth_max: float = 10.0,
        render_dir: Optional[str] = None,
    ):
        """
        Saves various visualizations for a specific keyframe.

        This function renders the scene from a given viewpoint, compares the rendered image
        and depth to ground truth, calculates quality metrics, and saves visualizations.
        If uncertainty-aware mode is enabled, it also includes uncertainty visualizations.

        Parameters:
        - keyframe_idx (int): Index of the keyframe to visualize.
        - plot_dir (str): Directory where the visualizations will be saved.
        - suffix (str, optional): Additional string to append to the saved filename. Defaults to "".
                                  The saved image will be named as "{keyframe_idx}{suffix}.png".
        - depth_max (float, optional): Maximum depth value for visualization. Defaults to 10.0.
        - render_dir (str, optional): Directory where raw rendered RGB frames are saved.
        """
        viewpoint = self.cameras[keyframe_idx]
        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        (rendered_img, rendered_depth,) = (
            render_pkg["render"].detach(),
            render_pkg["depth"].detach(),
        )
        if self.config["mapping"]["full_resolution"]:
            rendered_depth = F.interpolate(
                rendered_depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
            ).squeeze(0)
        rendered_img, rendered_depth = self._apply_visual_dynamic_eraser(
            viewpoint, rendered_img, rendered_depth
        )
        gt_image = viewpoint.original_image
        gt_depth = viewpoint.depth

        rendered_img = torch.clamp(rendered_img, 0.0, 1.0)
        raw_render_rgb = (
            rendered_img.detach().cpu().numpy().transpose((1, 2, 0)) * 255
        ).astype(np.uint8)
        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = raw_render_rgb.copy()
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        mask = gt_image > 0
        psnr_score = psnr(
            (rendered_img[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0)
        )
        diff_rgb=np.abs(gt - pred)
        diff_depth_l1 = torch.abs(rendered_depth.detach().cpu() - gt_depth)
        diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
        depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()
        diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)

        if self.uncertainty_aware:
            # Add plotting 2x4 grid with additional figures for uncertainty
            # Estimated uncertainty map
            uncertainty_map = self.get_viewpoint_uncertainty_no_grad(viewpoint)
            uncertainty_map = uncertainty_map.cpu().squeeze(0)

            # SSIM loss
            opacity = render_pkg["opacity"].detach().squeeze()
            ssim_loss = self._get_uncertainty_ssim_loss_vis(
                gt_image, rendered_img, opacity
            )
            ssim_loss = ssim_loss.cpu().squeeze(0)
        else:
            # All white
            uncertainty_map = torch.ones_like(rendered_img)
            ssim_loss = torch.ones_like(rendered_img)

        # Make the plot
        # Determine Plot Aspect Ratio
        aspect_ratio = gt_image.shape[2] / gt_image.shape[1]
        fig_height = 8
        fig_width = 11
        fig_width = fig_width * aspect_ratio

        # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
        fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(gt_image.cpu().permute(1, 2, 0))
        axs[0, 0].set_title("Ground Truth RGB", fontsize=16)
        axs[0, 1].imshow(gt_depth, cmap='jet', vmin=0, vmax=depth_max)
        axs[0, 1].set_title(f"Metric Depth, vmax:{depth_max:.2f}", fontsize=16)
        axs[1, 0].imshow(rendered_img.cpu().permute(1, 2, 0))
        axs[1, 0].set_title("Rendered RGB, PSNR: {:.2f}".format(psnr_score.item()), fontsize=16)
        axs[1, 1].imshow(rendered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=depth_max)
        axs[1, 1].set_title("Rendered Depth, L1: {:.2f}".format(depth_l1), fontsize=16)
        axs[0, 2].imshow(diff_rgb, cmap='jet', vmin=0, vmax=diff_rgb.max())
        axs[0, 2].set_title(f"Diff RGB L1, vmax:{diff_rgb.max():.2f}", fontsize=16)
        axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=depth_max/5.0)
        axs[1, 2].set_title(f"Diff Depth L1, vmax:{depth_max/5.0:.2f}", fontsize=16)
        axs[0, 3].imshow(uncertainty_map, cmap='jet', vmin=0, vmax=5)
        axs[0, 3].set_title("Uncertainty", fontsize=16)
        axs[1, 3].imshow(ssim_loss, cmap='jet', vmin=0, vmax=5)
        axs[1, 3].set_title("ssim_loss", fontsize=16)
        
        for i in range(2):
            for j in range(4):
                axs[i, j].axis('off')
                axs[i, j].grid(False)

        frame_idx = int(self.video.timestamp[keyframe_idx])
        if render_dir is not None:
            os.makedirs(render_dir, exist_ok=True)
            render_path = os.path.join(
                render_dir, f"video_idx_{keyframe_idx}_kf_idx_{frame_idx}{suffix}.png"
            )
            Image.fromarray(raw_render_rgb).save(render_path)

        fig.suptitle(f"Key Frame idx ({keyframe_idx}), Frame idx ({frame_idx}), Plot{suffix}", y=0.95, fontsize=20)
        fig.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"video_idx_{keyframe_idx}_kf_idx_{frame_idx}{suffix}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def save_all_kf_figs(
        self,
        save_dir: str,
        iteration: Union[str, float] ="after_refine",
    ): 
        """
        Save figures for all keyframes in the specified directory.

        This function saves figures for each keyframe in the video sequence,
        creates a directory for plots, and generates a gif from the saved figures.

        Args:
            save_dir (str): The base directory where figures will be saved.
            iteration (Union[str, float]): A string or float representing the 
                                        iteration or stage of the process. 
                                        Default is "after_refine".
        """
        video_idxs = self.video_idxs

        plot_dir = os.path.join(save_dir, "plots_" + iteration)
        render_dir = os.path.join(save_dir, "rendered_rgb_" + iteration)
        mkdir_p(plot_dir)
        mkdir_p(render_dir)

        for kf_idx in video_idxs:
            self.save_fig_everything(kf_idx, plot_dir, render_dir=render_dir)
        # Create gif
        create_gif_from_directory(plot_dir, plot_dir + '/output.gif', online=True)
        create_video_from_directory(
            render_dir,
            os.path.join(save_dir, f"rendered_rgb_{iteration}.mp4"),
            fps=10.0,
            online=True,
        )

    @torch.no_grad()
    def _vis_uncertainty_mask_all(self, n_rows=8, n_cols=8, is_final=False):
        """Used to inspect the uncertainty"""
        assert (
            n_rows % 2 == 0
        )  # one row for uncertainty, one for imgs, the other for uncertainty

        n_img = int(n_rows * n_cols / 2)
        if n_img >= len(self.cameras):
            keyframe_idxs = list(self.cameras.keys())
        else:
            keyframe_idxs = (
                list(self.cameras.keys())[:n_cols]
                + list(self.cameras.keys())[-(n_img - n_cols) :]
            )

        h = self.cameras[keyframe_idxs[0]].image_height
        w = self.cameras[keyframe_idxs[0]].image_width
        all_white = (np.ones((h, w, 3)) * 255).astype(np.uint8)

        aspect_ratio = w / h
        fig_height = 8
        fig_width = 8.5
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * fig_width * aspect_ratio, n_rows * fig_height),
        )
        for i in range(0, n_rows // 2):
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx >= len(keyframe_idxs):
                    axs[2 * i, j].imshow(all_white)
                    axs[2 * i + 1, j].imshow(all_white)
                else:
                    viewpoint = self.cameras[keyframe_idxs[idx]]
                    rgb = viewpoint.original_image.cpu().permute(1, 2, 0).numpy()
                    rgb = (rgb * 255.0).astype(np.uint8)
                    uncer_resized = self.get_viewpoint_uncertainty_no_grad(viewpoint)
                    uncer_resized = uncer_resized.cpu().squeeze(0)

                    axs[2 * i, j].imshow(rgb)
                    if keyframe_idxs[idx] in self.current_window:
                        # used for highlight
                        rect = patches.Rectangle(
                            (0, 0),
                            1,
                            1,
                            linewidth=30,
                            edgecolor="red",
                            facecolor="none",
                            transform=axs[2 * i, j].transAxes,
                        )
                        axs[2 * i, j].add_patch(rect)
                    axs[2 * i + 1, j].imshow(
                        uncer_resized ** 2, cmap="jet", vmin=0, vmax=5
                    )
                    axs[2 * i + 1, j].grid(False)
                axs[2 * i, j].axis("off")
                axs[2 * i + 1, j].axis("off")

        fig.tight_layout()
        cur_idx = self.current_window[np.array(self.current_window).argmax()]
        os.makedirs(os.path.join(self.save_dir, "online_uncer"), exist_ok=True)
        if is_final:
            save_path = os.path.join(
                self.save_dir, "online_uncer", f"after_final_refine.png"
            )
        else:
            save_path = os.path.join(self.save_dir, "online_uncer", f"{cur_idx}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
