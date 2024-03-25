import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.graphics_utils import getProjectionMatrix2, getWorld2View2

from gaussian_splatting.gaussian_model import GaussianModel

from utils.camera_utils import Camera



if __name__ == "__main__":
    config_file = "configs/rgbd/tum/fr1_desk.yaml"
    config = load_config(config_file)
    
    dataset = load_dataset(path=config_file, config=config)

    cur_frame_idx = 0
    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=dataset.fx,
        fy=dataset.fy,
        cx=dataset.cx,
        cy=dataset.cy,
        W=dataset.width,
        H=dataset.height,
    ).transpose(0, 1)

    viewpoint = Camera.init_from_dataset(dataset, cur_frame_idx, projection_matrix)
    print(type(viewpoint.original_image))
    print(type(viewpoint.depth))
    print(type(viewpoint.R_gt))
    print(type(viewpoint.T_gt))
    print(viewpoint.original_image.shape)
    print(viewpoint.depth.shape)

    gaussians = GaussianModel(config=config)
    rgbd, pcd_temp, pcd = gaussians.create_pcd_from_rgbd(viewpoint)

    '''    
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd.depth)
    plt.show()

    o3d.visualization.draw_geometries([pcd_temp])
    '''

    fused_point_cloud, fused_color, radius, opacities = gaussians.create_from_pcd(pcd)

    '''
    # Create an empty list to store sphere geometries
    sphere_geometries = []
    for i in range(0, fused_point_cloud.shape[0]):
        sphere_geometry = o3d.geometry.TriangleMesh.create_sphere(radius = gaussians.radius[i])
        sphere_geometry.compute_vertex_normals()
        sphere_geometry.paint_uniform_color(gaussians.color[i].cpu().detach().numpy())
        sphere_geometry.translate(gaussians.xyz[i].cpu().detach().numpy())
        sphere_geometries.append(sphere_geometry)
    
    o3d.visualization.draw_geometries(sphere_geometries)
    '''




    

    
