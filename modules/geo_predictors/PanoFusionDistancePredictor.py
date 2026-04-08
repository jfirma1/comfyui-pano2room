import os.path
import numpy as np
import torch
import trimesh

from utils.camera_utils import *
from utils.plane_utils import fit_dominant_planes
from modules.geo_predictors import PanoFusionInvPredictor, PanoFusionNormalPredictor, PanoGeoRefiner, PanoJointPredictor

import torchvision
from PIL import Image

class PanoFusionDistance:
    def __init__(self):
        self.image_path = None
        self.ref_distance_path = None
        self.ref_normal_path = None
        self.ref_geometry_path = None
        self.image = None
        self.gt_distance = None
        self.ref_distance = None
        self.ref_normal = None
        self.pano_width, self.pano_height = 2048, 1024
        self.data_dir = None
        self.case_name = 'wp'

    def get_ref_distance(self):
        assert self.image is not None
        assert self.ref_distance_path is not None
        assert self.height > 0 and self.width > 0

        ref_distance = None
        if os.path.exists(self.ref_distance_path):
            ref_distance = np.load(self.ref_distance_path)
            ref_distance = torch.from_numpy(ref_distance.astype(np.float32)).cuda()
        else:
            distance_predictor = PanoFusionInvPredictor()
            ref_distance, _ = distance_predictor(self.image,
                                                 torch.zeros([self.height, self.width]),
                                                 torch.ones([self.height, self.width]))
        return ref_distance

    def get_ref_normal(self):

        normal_predictor = PanoFusionNormalPredictor()
        ref_normal = normal_predictor.inpaint_normal(self.image,
                                                        torch.ones([self.height, self.width, 3]) / np.sqrt(3.),
                                                        torch.ones([self.height, self.width]))

        return ref_normal

    def refine_geometry(self, distance_map, normal_map):
        refiner = PanoGeoRefiner()
        return refiner.refine(distance_map, normal_map)

    def get_joint_distance_normal(self, init_distance=None, init_mask=None, dominant_planes=None):

        joint_predictor = PanoJointPredictor()
        idx = 0
        ref_distance, ref_normal = joint_predictor(idx, self.image,
                                                    torch.ones([self.pano_height, self.pano_width, 1]),
                                                    torch.ones([self.pano_height, self.pano_width]),
                                                    dominant_planes=dominant_planes)

        return ref_distance, ref_normal

    def normalization(self):
        scale = self.ref_distance.max().item() * 1.05
        self.ref_distance /= scale

    def save_ref_geometry(self):
        if self.ref_distance_path is not None:
            np.save(self.ref_distance_path, self.ref_distance.cpu().numpy())
        if self.ref_normal_path is not None:
            np.save(self.ref_normal_path, self.ref_normal.cpu().numpy())

        # Save point cloud
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(self.height, self.width))
        pts = pano_dirs * self.ref_distance.squeeze()[..., None]
        pts = pts.cpu().numpy().reshape(-1, 3)
        if self.image is not None:
            pcd = trimesh.PointCloud(pts, vertex_colors=self.image.reshape(-1, 3).cpu().numpy())
        else:
            pcd = trimesh.PointCloud(pts)

        assert self.ref_geometry_path is not None and self.ref_geometry_path[-4:] == '.ply'
        pcd.export(self.ref_geometry_path)

    @torch.no_grad()
    def ref_point_cloud(self):
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(self.height, self.width))
        pts = pano_dirs * self.ref_distance.squeeze()[..., None]
        return pts


class PanoFusionDistancePredictor(PanoFusionDistance):
    def __init__(self):
        super().__init__()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def predict(self, pano_tensor, init_distance=None, init_mask=None, pano_width=2048, pano_height=1024):
        self.pano_width, self.pano_height = pano_width, pano_height
        self.image = pano_tensor.cuda()

        # If a prior depth estimate is available, fit planes once here and
        # thread them into the joint predictor to avoid redundant computation.
        dominant_planes = None
        if init_distance is not None:
            pano_dirs_pp   = img_coord_to_pano_direction(img_coord_from_hw(pano_height, pano_width))
            dist_flat_pp   = init_distance.reshape(-1).float().cpu()
            dirs_flat_pp   = pano_dirs_pp.reshape(-1, 3)
            valid_pp       = dist_flat_pp > 1e-2
            if int(valid_pp.sum()) >= 100:
                valid_pts_pp  = (dirs_flat_pp * dist_flat_pp[:, None])[valid_pp]
                valid_idx_pp  = valid_pp.nonzero(as_tuple=True)[0]
                N_full        = pano_height * pano_width
                dominant_planes = []
                for normal, d, local_mask in fit_dominant_planes(valid_pts_pp, n_planes=3):
                    full_mask = torch.zeros(N_full, dtype=torch.float32)
                    full_mask[valid_idx_pp[local_mask]] = 1.0
                    inlier_img = full_mask.reshape(1, 1, pano_height, pano_width)
                    dominant_planes.append((normal, d, inlier_img))
                if not dominant_planes:
                    dominant_planes = None

        self.ref_distance, self.ref_normal = self.get_joint_distance_normal(
            init_distance, init_mask, dominant_planes=dominant_planes)

        return self.ref_distance.squeeze(-1)
