import torch
import open3d as o3d
import numpy as np

from utils.graphics_utils import BasicPointCloud, getWorld2View2

class GaussianModel:
    def __init__(self, config):
        self._xyz = torch.empty(0, device="cuda")
        self._radius = torch.empty(0, device="cuda")
        self._color = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")

        self.config = config
        self.ply = None
    
    # Getters
    @property
    def xyz(self):
        return self._xyz
    
    @property
    def radius(self):
        return self._radius
    
    @property
    def color(self):
        return self._color
    
    @property
    def opacity(self):
        return self._opacity
    
    # Setters
    @xyz.setter
    def xyz(self, value):
        self._xyz = value

    @radius.setter
    def radius(self, value):
        self._radius = value

    @color.setter
    def color(self, value):
        self._color = value

    @opacity.setter
    def opacity(self, value):
        self._opacity = value

    def create_pcd_from_rgbd(self, viewpoint):
        rgb = o3d.geometry.Image(viewpoint.original_image.astype(np.uint8))
        depth = o3d.geometry.Image(viewpoint.depth.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(viewpoint.R, viewpoint.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                viewpoint.image_width,
                viewpoint.image_height,
                viewpoint.fx,
                viewpoint.fy,
                viewpoint.cx,
                viewpoint.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )

        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply = pcd

        return rgbd, pcd_tmp, pcd
    
    # MonoGS -> extend_from_pcd calls densicification calls to_dict
    # GS -> create_from pcd ; densification calls to_dict
    def create_from_pcd(self, pcd : BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        radiuses = 0.004 * torch.ones((fused_point_cloud.shape[0], 1)).float().cuda()
        opacities = 0.1 * torch.ones((fused_point_cloud.shape[0], 1)).float().cuda()

        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._radius = torch.nn.Parameter(radiuses.requires_grad_(True))
        self._color = torch.nn.Parameter(fused_color.requires_grad_(True))
        self._opacity = torch.nn.Parameter(opacities.requires_grad_(True))

        return fused_point_cloud, fused_color, radiuses, opacities


