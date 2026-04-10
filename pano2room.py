import torch
import os
import cv2
import json
from PIL import Image, ImageDraw
import numpy as np
from tqdm.auto import tqdm


from modules.mesh_fusion.render import (
    features_to_world_space_mesh,
    render_mesh,
    edge_threshold_filter,
    unproject_points,
)
from utils.common_utils import (
    visualize_depth_numpy,
    save_rgbd,
)

import time
from utils.camera_utils import *

import utils.functions as functions
from utils.functions import rot_x_world_to_cam, rot_y_world_to_cam, rot_z_world_to_cam, colorize_single_channel_image, write_video
from modules.equilib import equi2pers, cube2equi, equi2cube

from modules.geo_predictors.PanoFusionDistancePredictor import PanoFusionDistancePredictor
from modules.inpainters import PanoPersFusionInpainter
from modules.geo_predictors import PanoJointPredictor
from modules.mesh_fusion.sup_info import SupInfoPool
from modules.ambiguity import (
    AmbiguityManager,
    apply_constraints,
    dump_queries_json,
    load_answer_payload,
    parse_answers,
    answers_to_constraints,
)
from kornia.morphology import erosion, dilation
from scene.arguments import GSParams, CameraParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.graphics import focal2fov
from utils.loss import l1_loss, ssim
from random import randint

@torch.no_grad()
class Pano2RoomPipeline(torch.nn.Module):
    def __init__(self, attempt_idx="", mode="full"):
        super().__init__()
        self.mode = mode

        # renderer setting
        self.blur_radius = 0
        self.faces_per_pixel = 8
        self.fov = 90
        self.R, self.T = torch.Tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]), torch.Tensor([[0., 0., 0.]])
        self.pano_width, self.pano_height = 1024 * 2, 512 * 2
        self.H, self.W = 512, 512
        self.device = "cuda:0"

        # initialize
        self.rendered_depth = torch.zeros((self.H, self.W), device=self.device) 
        self.inpaint_mask = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)  
        self.vertices = torch.empty((3, 0), device=self.device, requires_grad=False)
        self.colors = torch.empty((3, 0), device=self.device, requires_grad=False)
        self.faces = torch.empty((3, 0), device=self.device, dtype=torch.long, requires_grad=False)
        self.pix_to_face = None

        self.pose_scale = 0.6
        self.pano_center_offset = (-0.2,0.3)
        self.inpaint_frame_stride = 20   

        # create exp dir
        self.setting = f""
        apply_timestamp = True
        if apply_timestamp:
            timestamp = str(int(time.time()))[-8:]
            self.setting += f"-{timestamp}"
        self.save_path = f'output/Pano2Room-results'
        self.save_details = False

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print("makedir:", self.save_path)

        self.world_to_cam = torch.eye(4, dtype=torch.float32, device=self.device)
        self.cubemap_w2c_list = functions.get_cubemap_views_world_to_cam()

        self.inpainter = None
        self.geo_predictor = None
        self.load_modules_for_mode()

    def load_modules_for_mode(self):
        if self.mode in ["full", "resume"]:
            self.ensure_geo_predictor_loaded()
            # Keep full/resume behavior compatible by loading inpainter eagerly.
            self.ensure_inpainter_loaded()
        elif self.mode == "probe":
            # Probe mode intentionally avoids inpainting stack and predictors only needed post-probe.
            return
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def ensure_geo_predictor_loaded(self):
        if self.geo_predictor is None:
            self.geo_predictor = PanoJointPredictor(save_path=self.save_path)
        return self.geo_predictor

    def _validate_lama_checkpoints(self):
        expected_files = [
            "./checkpoints/big-lama-config.yaml",
            "./checkpoints/big-lama.ckpt",
        ]
        missing = [p for p in expected_files if not os.path.exists(p)]
        if missing:
            missing_bullet = "\n".join([f"  - {p}" for p in missing])
            expected_bullet = "\n".join([f"  - {p}" for p in expected_files])
            raise RuntimeError(
                "Missing LaMa checkpoint files required for inpainting in full/resume mode.\n"
                "Expected files:\n"
                f"{expected_bullet}\n"
                "Expected path: ./checkpoints/\n"
                "Missing files:\n"
                f"{missing_bullet}\n"
                "Probe mode (--mode probe) does not require LaMa."
            )

    def ensure_inpainter_loaded(self):
        if self.inpainter is None:
            self._validate_lama_checkpoints()
            self.inpainter = PanoPersFusionInpainter(save_path=self.save_path)
        return self.inpainter

    def project(self, world_to_cam):
        # project mesh into pose and render (rgb, depth, mask)
        rendered_image_tensor, self.rendered_depth, self.inpaint_mask, self.pix_to_face, self.z_buf, self.mesh = render_mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_features=self.colors,
            H=self.H,
            W=self.W,
            fov_in_degrees=self.fov,
            RT=world_to_cam,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel
        )
        # mask rendered_image_tensor
        rendered_image_tensor = rendered_image_tensor * ~self.inpaint_mask

        # stable diffusion models want the mask and image as PIL images
        rendered_image_pil = Image.fromarray((rendered_image_tensor.permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
        self.inpaint_mask_pil = Image.fromarray(self.inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")

        self.inpaint_mask_restore = self.inpaint_mask
        self.inpaint_mask_pil_restore = self.inpaint_mask_pil

        return rendered_image_tensor[:3, ...], rendered_image_pil

    def render_pano(self, pose):
        cubemap_list = [] 
        for cubemap_pose in self.cubemap_w2c_list:
            pose_tmp = pose.clone()
            pose_tmp = cubemap_pose.cuda() @ pose_tmp
            rendered_image_tensor, rendered_image_pil = self.project(pose_tmp.cuda())

            rgb_CHW = rendered_image_tensor.squeeze(0).cuda()
            depth_CHW = self.rendered_depth.unsqueeze(0).cuda()
            distance_CHW = functions.depth_to_distance(depth_CHW)
            mask_CHW = self.inpaint_mask.unsqueeze(0).cuda()
            cubemap_list += [torch.cat([rgb_CHW, distance_CHW, mask_CHW], axis=0)]

        torch.set_default_tensor_type('torch.FloatTensor')
        pano_rgbd = cube2equi(cubemap_list,
                                "list",
                                1024,2048) #CHW
        pano_rgb = pano_rgbd[:3,:,:]
        pano_depth =  pano_rgbd[3:4,:,:].squeeze(0)
        pano_mask =  pano_rgbd[4:,:,:].squeeze(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return pano_rgb, pano_depth, pano_mask  # CHW, HW, HW

    def rgbd_to_mesh(self, rgb, depth, world_to_cam=None, mask=None, pix_to_face=None, using_distance_map=False):
        predicted_depth = depth.cuda()
        rgb = rgb.squeeze(0).cuda()
        if world_to_cam is None:
            world_to_cam = torch.eye(4, dtype=torch.float32)
        world_to_cam = world_to_cam.cuda()
        if pix_to_face is not None:
            self.pix_to_face = pix_to_face
        if mask is None:
            self.inpaint_mask = torch.ones_like(predicted_depth)
        else:
            self.inpaint_mask = mask

        if self.inpaint_mask.sum() == 0:
            return
        
        vertices, faces, colors = features_to_world_space_mesh(
            colors=rgb,
            depth=predicted_depth,
            fov_in_degrees=self.fov,
            world_to_cam=world_to_cam,
            mask=self.inpaint_mask,
            pix_to_face=self.pix_to_face,
            faces=self.faces,
            vertices=self.vertices,
            using_distance_map=using_distance_map,
            edge_threshold=0.05
        )

        faces += self.vertices.shape[1] 

        self.vertices_restore = self.vertices.clone()
        self.colors_restore = self.colors.clone()
        self.faces_restore = self.faces.clone()

        self.vertices = torch.cat([self.vertices, vertices], dim=1)
        self.colors = torch.cat([self.colors, colors], dim=1)
        self.faces = torch.cat([self.faces, faces], dim=1)

    def find_depth_edge(self, depth, dilate_iter=0):
        gray = (depth/depth.max() * 255).astype(np.uint8)
        edges = cv2.Canny(gray, 60, 150)
        if dilate_iter > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
        return edges

    def pano_distance_to_mesh(self, pano_rgb, pano_distance, depth_edge_inpaint_mask, pose=None):
        self.rgbd_to_mesh(pano_rgb, pano_distance, mask=depth_edge_inpaint_mask, using_distance_map=True, world_to_cam=pose)
 
    def load_inpaint_poses(self):
        pano_rgb, pano_distance, pano_mask = self.render_pano(self.world_to_cam)

        pose_dict = {} # {idx:pose, ...} # pose are c2w
        key = 0

        sampled_inpaint_poses = self.poses[::self.inpaint_frame_stride]
        for anchor_idx in range(len(sampled_inpaint_poses)):
            pose = torch.eye(4).float() # pano pose dosen't support rotations

            pose_44 = sampled_inpaint_poses[anchor_idx].clone()
            pose_44 = pose_44.float()
            Rw2c = pose_44[:3,:3].cpu().numpy()
            Tw2c = pose_44[:3,3:].cpu().numpy()
            yz_reverse = np.array([[1,0,0], [0,1,0], [0,0,1]])
            Rc2w = np.matmul(yz_reverse, Rw2c).T
            Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
            Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0) 
            pose[:3, 3] = torch.tensor(Pc2w[:3, 3]).cuda().float()
            pose[:3, 3] *= -1
            pose_dict[key] = pose.clone()

            key += 1
        return  pose_dict

    def stage_inpaint_pano_greedy_search(self, pose_dict): 
        print("stage_inpaint_pano_greedy_search")
        pano_rgb, pano_distance, pano_mask = self.render_pano(self.world_to_cam)

        inpainted_panos_and_poses = []
        while len(pose_dict) > 0:
            print(f"len(pose_dict):{len(pose_dict)}")

            values_sampled_poses = []
            keys = list(pose_dict.keys())
            for key in keys:
                pose = pose_dict[key]
                pano_rgb, pano_distance, pano_mask = self.render_pano(pose.cuda())
                view_completeness = torch.sum((1 - pano_mask * 1))/(pano_mask.shape[0] * pano_mask.shape[1])
                
                values_sampled_poses += [(key, view_completeness, pose)]
                torch.cuda.empty_cache() 
            if len(values_sampled_poses) < 1:
                break

            # find inpainting with least view completeness
            values_sampled_poses = sorted(values_sampled_poses, key=lambda item: item[1])
            # least_complete_view = values_sampled_poses[0]
            least_complete_view = values_sampled_poses[len(values_sampled_poses)*2//3]

            key, view_completeness, pose = least_complete_view
            print(f"least_complete_view:{view_completeness}")
            del pose_dict[key]

            # rendering rgb depth mask
            pano_rgb, pano_distance, pano_mask = self.render_pano(pose.cuda())

            # inpaint pano
            colors = pano_rgb.permute(1,2,0).clone()
            distances = pano_distance.unsqueeze(-1).clone()
            pano_inpaint_mask = pano_mask.clone()

            if pano_inpaint_mask.min().item() < .5:
                # inpainting pano
                colors, distances, normals = self.inpaint_new_panorama(idx=key, colors=colors, distances=distances, pano_mask=pano_inpaint_mask) # HWC, HWC, HW
                
                #apply_GeoCheck:
                perf_pose = pose.clone()
                perf_pose[0,3], perf_pose[1,3], perf_pose[2,3] = -pose[0,3], pose[2,3], 0 
                rays = gen_pano_rays(perf_pose, self.pano_height, self.pano_width)
                conflict_mask = self.sup_pool.geo_check(rays, distances.unsqueeze(-1))    # 0 conflict, 1 not conflict
                pano_inpaint_mask = pano_inpaint_mask * conflict_mask
                    
            # add new mesh
            self.pano_distance_to_mesh(colors.permute(2,0,1), distances, pano_inpaint_mask, pose=pose) #CHW, HW, HW

            # apply_GeoCheck:
            sup_mask = pano_inpaint_mask.clone()
            self.sup_pool.register_sup_info(pose=perf_pose, mask=sup_mask, rgb=colors, distance=distances.unsqueeze(-1), normal=normals)
            
            # save renderred
            panorama_tensor_pil = functions.tensor_to_pil(pano_rgb.unsqueeze(0))
            panorama_tensor_pil.save(f"{self.save_path}/renderred_pano_{key}.png")
            if self.save_details:
                depth_pil = Image.fromarray(colorize_single_channel_image(pano_distance.unsqueeze(0)/self.scene_depth_max))
                depth_pil.save(f"{self.save_path}/renderred_depth_{key}.png")        
                inpaint_mask_pil = Image.fromarray(pano_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
                inpaint_mask_pil.save(f"{self.save_path}/mask_{key}.png")  
                inpaint_mask_pil = Image.fromarray(pano_inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
                inpaint_mask_pil.save(f"{self.save_path}/inpaint_mask_{key}.png")  

            # save inpainted
            panorama_tensor_pil = functions.tensor_to_pil(colors.permute(2,0,1).unsqueeze(0))
            panorama_tensor_pil.save(f"{self.save_path}/inpainted_pano_{key}.png")
            depth_pil = Image.fromarray(colorize_single_channel_image(distances.unsqueeze(0)/self.scene_depth_max))
            depth_pil.save(f"{self.save_path}/inpainted_depth_{key}.png") 

            # collect pano images for GS training
            inpainted_panos_and_poses += [(colors.permute(2,0,1).unsqueeze(0), pose.clone())] #BCHW, 44
            
        return inpainted_panos_and_poses

    def inpaint_new_panorama(self, idx, colors, distances, pano_mask):
        print(f"inpaint_new_panorama")
        self.ensure_inpainter_loaded()
        self.ensure_geo_predictor_loaded()

        # must dilate mask first
        mask = pano_mask.unsqueeze(-1)
        s_size = (9, 9)
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, s_size)
        kernel_s = torch.from_numpy(kernel_s).to(torch.float32).to(mask.device)
        mask = (mask[None, :, :, :] > 0.5).float()
        mask = mask.permute(0, 3, 1, 2)
        mask = dilation(mask, kernel=kernel_s)
        mask.permute(0, 2, 3, 1).contiguous().squeeze(0).squeeze(-1)

        distances = distances.squeeze()[..., None]
        mask = mask.squeeze()[..., None]

        inpainted_distances = None
        inpainted_normals = None

        inpainted_img = self.inpainter.inpaint(idx, colors, mask)

        # Keep renderred part
        inpainted_img = colors * (1 - mask) + inpainted_img * mask
        inpainted_img = inpainted_img.cuda()

        inpainted_distances, inpainted_normals = self.geo_predictor(idx,
                                                                    inpainted_img,
                                                                    distances,
                                                                    mask=mask,
                                                                    reg_loss_weight=0.,
                                                                    normal_loss_weight=5e-2,
                                                                    normal_tv_loss_weight=5e-2)

        inpainted_distances = inpainted_distances.squeeze()
        return inpainted_img, inpainted_distances, inpainted_normals

    def load_pano(self):
        image_path = f"input/input_panorama.png"
        image = Image.open(image_path)
        if image.size[0] < image.size[1]:
            image = image.transpose(Image.TRANSPOSE)
        image = functions.resize_image_with_aspect_ratio(image, new_width=self.pano_width)
        panorama_tensor = torch.tensor(np.array(image))[...,:3].permute(2,0,1).unsqueeze(0).float()/255
        panorama_image_pil = functions.tensor_to_pil(panorama_tensor)

        depth_scale_factor = 3.4092

        # get panofusion_distance
        pano_fusion_distance_predictor = PanoFusionDistancePredictor()
        depth = pano_fusion_distance_predictor.predict(panorama_tensor.squeeze(0).permute(1,2,0)) #input:HW3
        depth = depth/depth.max() * depth_scale_factor
        print(f"pano_fusion_distance...[{depth.min(), depth.mean(),depth.max()}]")
        
        return panorama_tensor, depth# panorama_tensor:BCHW, depth:HW

    def load_camera_poses(self, pano_center_offset=[0,0]):
        subset_path = self._camera_trajectory_path() # initial 6 poses are cubemaps poses
        if not os.path.isdir(subset_path):
            raise FileNotFoundError(
                f"Camera trajectory directory not found: {subset_path}. "
                "Full generation requires camera_pose*.txt files. "
                "Probe mode can run without this directory."
            )
        files = os.listdir(subset_path)

        self.scene_depth_max = 4.0228885328450446

        pano_pose_44 = None
        pose_files = [f for f in files if f.startswith('camera_pose')]
        pose_files = sorted(pose_files)
        poses_name = pose_files
        poses = []
        for i, pose_name in enumerate(poses_name):
            with open(f'{subset_path}/{pose_name}', 'r') as f: 
                lines = f.readlines()
            pose_44 = []
            for line in lines:
                pose_44 += line.split()
            pose_44 = np.array(pose_44).reshape(4, 4).astype(float)
            if pano_pose_44 is None:
                pano_pose_44 = pose_44.copy()
                pano_pose_44_cubemaps = pose_44.copy()
                pano_pose_44[0,3] += pano_center_offset[0]
                pano_pose_44[2,3] += pano_center_offset[1]
            
            if i < 6:
                pose_relative_44 = pose_44 @ np.linalg.inv(pano_pose_44_cubemaps)  
            else:
                ### convert gt_pose to gt_relative_pose with pano_pose
                pose_relative_44 = pose_44 @ np.linalg.inv(pano_pose_44)

            pose_relative_44 = np.vstack((-pose_relative_44[0:1,:], -pose_relative_44[1:2,:], pose_relative_44[2:3,:], pose_relative_44[3:4,:]))
            pose_relative_44 = pose_relative_44 @ rot_z_world_to_cam(180).cpu().numpy()

            pose_relative_44[:3,3] *= self.pose_scale
            poses += [torch.tensor(pose_relative_44).float()] # w2c

        return pano_pose_44, poses

    def pano_to_perpective(self, pano_bchw, pitch, yaw, fov):
        rots = {
            'roll': 0.,
            'pitch': pitch,  # rotate vertical
            'yaw': yaw,  # rotate horizontal
        }

        perspective = equi2pers(
            equi=pano_bchw.squeeze(0),
            rots=rots,
            height=self.H,
            width=self.W,
            fov_x=fov,
            mode="bilinear",
        ).unsqueeze(0) # BCHW

        return perspective

    def pano_to_cubemap(self, pano_tensor, pano_depth_tensor=None): #BCHW, HW
        cubemaps_pitch_yaw = [(0, 0), (0, 3/2 * np.pi), (0, 1 * np.pi), (0, 1/2 * np.pi),\
                            (-1/2 * np.pi, 0), (1/2 * np.pi, 0)]
        pitch_yaw_list = cubemaps_pitch_yaw

        cubemaps = []
        cubemaps_depth = []
        # collect fov 90 cubemaps
        for view_idx, (pitch, yaw) in enumerate(pitch_yaw_list):
            view_rgb = self.pano_to_perpective(pano_tensor, pitch, yaw, 90)
            cubemaps += [view_rgb.cpu().clone()]
            if pano_depth_tensor is not None:
                view_depth = self.pano_to_perpective(pano_depth_tensor.unsqueeze(0).unsqueeze(0), pitch, yaw, 90)
                cubemaps_depth += [view_depth.cpu().clone()]
        return cubemaps, cubemaps_depth  # BCHW, BCHW

    def train_GS(self):
        if not self.scene:
            raise('Build 3D Scene First!')
        
        iterable_gauss = range(1, self.opt.iterations + 1)

        for iteration in iterable_gauss:
            self.gaussians.update_learning_rate(iteration)

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam, mesh_pose = viewpoint_stack[iteration%len(viewpoint_stack)]

            # Render GS
            render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            render_image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
            
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(render_image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(render_image, gt_image))
            loss.backward()

            if self.save_details:
                if iteration % 200 == 0:
                    functions.write_image(f"{self.save_path}/Train_Ref_rgb_{iteration}.png", gt_image.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.)
                    functions.write_image(f"{self.save_path}/Train_GS_rgb_{iteration}.png", render_image.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.)

            with torch.no_grad():
                # Densification
                if iteration < self.opt.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    
                    if (iteration % self.opt.opacity_reset_interval == 0 
                        or (self.opt.white_background and iteration == self.opt.densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

    def eval_GS(self, eval_GS_cams):
        viewpoint_stack = eval_GS_cams
        l1_val = 0
        ssim_val = 0
        psnr_val = 0
        framelist = []
        depthlist = []
        for i in range(len(viewpoint_stack)):
            viewpoint_cam, mesh_pose = viewpoint_stack[i]

            results = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            frame, depth = results['render'], results['depth'].detach().cpu()

            framelist.append(
                np.round(frame.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            depthlist.append(colorize_single_channel_image(depth.detach().cpu()/self.scene_depth_max))

        if self.save_details:
            for i, frame in enumerate(framelist):
                image = Image.fromarray(frame, mode="RGB")
                image.save(os.path.join(self.save_path, f"Eval_render_rgb_{i}.png"))
                functions.write_image(f"{self.save_path}/Eval_render_depth_{i}.png", depthlist[i])
        
        write_video(f"{self.save_path}/GS_render_video.mp4", framelist[6:], fps=30)
        write_video(f"{self.save_path}/GS_depth_video.mp4", depthlist[6:], fps=30)
        print("Result saved at: ", self.save_path)

    def _runtime_settings(self):
        return {
            "resolution": int(self.H),
            "fov": float(self.fov),
            "inpaint_stride": int(self.inpaint_frame_stride),
            "offset_x": float(self.pano_center_offset[0]),
            "offset_z": float(self.pano_center_offset[1]),
            "room_scale": float(self.pose_scale),
            "save_details": bool(self.save_details),
        }

    def _apply_runtime_settings(self, runtime_settings):
        self.H = int(runtime_settings.get("resolution", self.H))
        self.W = int(runtime_settings.get("resolution", self.W))
        self.fov = float(runtime_settings.get("fov", self.fov))
        self.inpaint_frame_stride = int(runtime_settings.get("inpaint_stride", self.inpaint_frame_stride))
        self.pano_center_offset = (
            float(runtime_settings.get("offset_x", self.pano_center_offset[0])),
            float(runtime_settings.get("offset_z", self.pano_center_offset[1])),
        )
        self.pose_scale = float(runtime_settings.get("room_scale", self.pose_scale))
        self.save_details = bool(runtime_settings.get("save_details", self.save_details))
        self.rendered_depth = torch.zeros((self.H, self.W), device=self.device)
        self.inpaint_mask = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)

    def _session_output_dir(self, stage_tag):
        folder = os.path.join(self.save_path, f"clarification-{stage_tag}-{str(int(time.time()))[-8:]}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _camera_trajectory_path(self):
        return "input/Camera_Trajectory"

    def _build_default_probe_poses(self):
        """
        Probe-only fallback when trajectory assets are unavailable.
        Uses identity pano pose plus cubemap anchor poses so we can persist a
        valid resume payload without depending on demo trajectory files.
        """
        pano_pose_44 = np.eye(4, dtype=float)
        poses = [pose.clone().float().cpu() for pose in self.cubemap_w2c_list]
        self.scene_depth_max = 4.0228885328450446
        return pano_pose_44, poses

    def prepare_ambiguity_probe(self, state_dir=None):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        trajectory_path = self._camera_trajectory_path()
        if self.mode == "probe":
            if os.path.isdir(trajectory_path):
                self.pano_pose, self.poses = self.load_camera_poses(self.pano_center_offset)
            else:
                self.pano_pose, self.poses = self._build_default_probe_poses()
        else:
            if not os.path.isdir(trajectory_path):
                raise FileNotFoundError(
                    "Missing camera trajectory assets required for full generation mode.\n"
                    f"Expected directory: {trajectory_path}\n"
                    "Use --mode probe to generate ambiguity artifacts without trajectory files."
                )
            self.pano_pose, self.poses = self.load_camera_poses(self.pano_center_offset)
        pano_rgb, pano_depth = self.load_pano()
        panorama_tensor, init_depth = pano_rgb.squeeze(0).cuda(), pano_depth.cuda()

        depth_edge = self.find_depth_edge(init_depth.cpu().detach().numpy(), dilate_iter=1)
        depth_edge_inpaint_mask = ~(torch.from_numpy(depth_edge).cuda().bool())

        stage_dir = state_dir or self._session_output_dir("probe")
        ambiguity_manager = AmbiguityManager(top_k=3)
        ambiguity_info = ambiguity_manager.compute(
            pano_rgb=panorama_tensor.permute(1, 2, 0).detach().cpu().numpy(),
            init_depth=init_depth.detach().cpu().numpy(),
            depth_edges=depth_edge.astype(np.uint8),
        )

        Image.fromarray(ambiguity_info["heatmap"]).save(os.path.join(stage_dir, "ambiguity_heatmap.png"))
        Image.fromarray(ambiguity_info["overlay"]).save(os.path.join(stage_dir, "ambiguity_overlay.png"))
        queries_path = os.path.join(stage_dir, "queries.json")
        dump_queries_json(queries_path, ambiguity_info["queries"])

        depth_edge_pil = Image.fromarray(depth_edge)
        depth_pil = Image.fromarray(visualize_depth_numpy(init_depth.cpu().detach().numpy())[0].astype(np.uint8))
        _, _ = save_rgbd(depth_pil, depth_edge_pil, f'depth_edge', 0, stage_dir)

        state_payload = {
            "pano_rgb": panorama_tensor.detach().cpu(),
            "init_depth": init_depth.detach().cpu(),
            "depth_edges": torch.from_numpy(depth_edge).cpu(),
            "depth_edge_inpaint_mask": depth_edge_inpaint_mask.detach().cpu(),
            "pano_pose": torch.from_numpy(self.pano_pose).float().cpu(),
            "poses": torch.stack([p.cpu() for p in self.poses], dim=0),
            "runtime_settings": self._runtime_settings(),
            "scene_depth_max": float(self.scene_depth_max),
        }
        state_path = os.path.join(stage_dir, "intermediate_state.pt")
        torch.save(state_payload, state_path)
        metadata = {
            "state_path": state_path,
            "queries_path": queries_path,
            "runtime_settings": self._runtime_settings(),
            "saved_tensors": [
                "pano_rgb",
                "init_depth",
                "depth_edges",
                "depth_edge_inpaint_mask",
                "pano_pose",
                "poses",
            ],
        }
        with open(os.path.join(stage_dir, "state_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return {
            "state_path": state_path,
            "state_dir": stage_dir,
            "queries": ambiguity_info["queries"],
            "queries_path": queries_path,
            "ambiguity_heatmap_path": os.path.join(stage_dir, "ambiguity_heatmap.png"),
            "ambiguity_overlay_path": os.path.join(stage_dir, "ambiguity_overlay.png"),
        }

    def _continue_from_initial_mesh(self, panorama_tensor, init_depth, depth_edge_inpaint_mask):
        self.sup_pool = SupInfoPool()
        self.sup_pool.register_sup_info(pose=torch.eye(4).cuda(),
                                        mask=torch.ones([self.pano_height, self.pano_width]),
                                        rgb=panorama_tensor.permute(1,2,0),
                                        distance=init_depth.unsqueeze(-1))
        self.sup_pool.gen_occ_grid(256)

        # Pano2Mesh
        self.pano_distance_to_mesh(panorama_tensor, init_depth, depth_edge_inpaint_mask)

        # Mesh Inpainting        
        pose_dict = self.load_inpaint_poses()
        print(f"start inpainting with poses #{len(self.poses)}")
        inpainted_panos_and_poses = self.stage_inpaint_pano_greedy_search(pose_dict)

        # Train 3DGS
        self.opt = GSParams()
        self.cam = CameraParams()
        self.gaussians = GaussianModel(self.opt.sh_degree)
        self.opt.white_background = True
        bg_color = [1, 1, 1] if self.opt.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        
        traindata = {
            'camera_angle_x': self.cam.fov[0],
            'W': self.W,
            'H': self.H,
            'pcd_points': self.vertices.detach().cpu(),
            'pcd_colors': self.colors.permute(1,0).detach().cpu(),
            'frames': [],
        }
        for inpainted_pano_images, pano_pose_44 in inpainted_panos_and_poses:
            cubemaps, cubemaps_depth = self.pano_to_cubemap(inpainted_pano_images) # BCHW
            for i in range(len(cubemaps)):
                inpainted_img = cubemaps[i] 

                mesh_pose = self.cubemap_w2c_list[i].cuda() @ pano_pose_44.clone()

                pose_44 = mesh_pose.clone()
                pose_44 = pose_44.float()
                pose_44[0:1,:] *= -1
                pose_44[1:2,:] *= -1

                Rw2c = pose_44[:3,:3].cpu().numpy()
                Tw2c = pose_44[:3,3:].cpu().numpy()
                yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

                Rc2w = np.matmul(yz_reverse, Rw2c).T
                Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
                Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)  

                traindata['frames'].append({
                    'image': functions.tensor_to_pil(inpainted_img),
                    'transform_matrix': Pc2w.tolist(), 
                    'fovx': focal2fov(256, inpainted_img.shape[-1]),
                    'mesh_pose': mesh_pose
                })


        self.scene = Scene(traindata, self.gaussians, self.opt)   
        self.train_GS()
        outfile = self.gaussians.save_ply(os.path.join(self.save_path, '3DGS.ply'))

        # Eval GS
        evaldata = {
            'camera_angle_x': self.cam.fov[0],
            'W': self.W,
            'H': self.H,
            'frames': [],
        }

        for i in range(len(self.poses)):
            gt_img = inpainted_img

            pose_44 = self.poses[i].clone()
            pose_44 = pose_44.float()
            pose_44[0:1,:] *= -1
            pose_44[1:2,:] *= -1

            Rw2c = pose_44[:3,:3].cpu().numpy()
            Tw2c = pose_44[:3,3:].cpu().numpy()
            yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

            Rc2w = np.matmul(yz_reverse, Rw2c).T
            Tc2w = -np.matmul(Rc2w, np.matmul(yz_reverse, Tw2c))
            Pc2w = np.concatenate((Rc2w, Tc2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)                  

            evaldata['frames'].append({
                'image': functions.tensor_to_pil(gt_img),
                'transform_matrix': Pc2w.tolist(), 
                'fovx': focal2fov(256, gt_img.shape[-1]),
                'mesh_pose': self.poses[i].clone()
            })
        from scene.dataset_readers import loadCamerasFromData
        eval_GS_cams = loadCamerasFromData(evaldata, self.opt.white_background)
        self.eval_GS(eval_GS_cams)

    def resume_from_clarification(self, state_path, answer_json="", answer_json_path=""):
        stage_dir = os.path.dirname(state_path)
        state_payload = torch.load(state_path, map_location="cpu")
        self._apply_runtime_settings(state_payload.get("runtime_settings", {}))
        self.pano_pose = state_payload["pano_pose"].cpu().numpy()
        self.poses = [p.cuda().float() for p in state_payload["poses"]]
        self.scene_depth_max = float(state_payload.get("scene_depth_max", 4.0228885328450446))

        panorama_tensor = state_payload["pano_rgb"].cuda()
        init_depth = state_payload["init_depth"].cuda()
        depth_edges = state_payload["depth_edges"].cuda()
        depth_edge_inpaint_mask = state_payload["depth_edge_inpaint_mask"].cuda().bool()

        queries_path = os.path.join(stage_dir, "queries.json")
        queries_payload = {"queries": []}
        if os.path.exists(queries_path):
            with open(queries_path, "r", encoding="utf-8") as f:
                queries_payload = json.load(f)

        answers_payload = load_answer_payload(answer_json=answer_json, answer_json_path=answer_json_path)
        answers = parse_answers(answers_payload)
        constraints = answers_to_constraints(answers, queries_payload)

        if answers:
            with open(os.path.join(stage_dir, "clarification_answers.json"), "w", encoding="utf-8") as f:
                json.dump(answers_payload, f, indent=2)

        if constraints:
            init_depth, depth_edges, depth_edge_inpaint_mask, debug_img = apply_constraints(
                init_depth=init_depth,
                depth_edges=depth_edges,
                depth_edge_inpaint_mask=depth_edge_inpaint_mask,
                constraints=constraints,
            )
            Image.fromarray(debug_img).save(os.path.join(stage_dir, "applied_constraints_debug.png"))
        # TODO: add optional update-stage escalation during greedy inpainting with iterative human feedback.

        self._continue_from_initial_mesh(panorama_tensor, init_depth, depth_edge_inpaint_mask)
        return {"state_dir": stage_dir, "status": "resume_complete"}

    def run(self):
        probe = self.prepare_ambiguity_probe()
        return self.resume_from_clarification(probe["state_path"])

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt_idx", type=str, default="")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "probe", "resume"])
    parser.add_argument("--state_path", type=str, default="")
    parser.add_argument("--answer_json", type=str, default="")
    parser.add_argument("--answer_json_path", type=str, default="")
    parser.add_argument("--inpaint_stride", type=int, default=20, help="Lower number = more holes fixed, but slower.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--room_scale", type=float, default=0.6)
    parser.add_argument("--offset_x", type=float, default=-0.2)
    parser.add_argument("--offset_z", type=float, default=0.3)
    args = parser.parse_args()

    pipeline = Pano2RoomPipeline(attempt_idx=args.attempt_idx, mode=args.mode)
    
    # Inject the settings from ComfyUI
    pipeline.inpaint_frame_stride = args.inpaint_stride
    pipeline.fov = args.fov
    pipeline.save_details = args.save_details
    pipeline.pose_scale = args.room_scale
    pipeline.pano_center_offset = (args.offset_x, args.offset_z)
    
    # Inject resolution and dynamically resize the canvas tensors
    pipeline.H = args.resolution
    pipeline.W = args.resolution
    pipeline.rendered_depth = torch.zeros((args.resolution, args.resolution), device=pipeline.device)
    pipeline.inpaint_mask = torch.ones((args.resolution, args.resolution), device=pipeline.device, dtype=torch.bool)

    if args.mode == "probe":
        output = pipeline.prepare_ambiguity_probe(state_dir=args.state_path or None)
        print(json.dumps(output))
    elif args.mode == "resume":
        if not args.state_path:
            raise ValueError("--state_path is required in resume mode.")
        output = pipeline.resume_from_clarification(
            state_path=args.state_path,
            answer_json=args.answer_json,
            answer_json_path=args.answer_json_path,
        )
        print(json.dumps(output))
    else:
        pipeline.run()
