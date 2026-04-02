import os
from . import logger
import time
import cv2
import pycolmap
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict
from lib.transform import convert_euler_to_matrix, wgs84tocgcs2000
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class QueryLocalizer:
    def __init__(self, config=None):
        self.config = config
        # 初始化其他需要的成员变量
        self.dataset = config['render2loc']['datasets']
        self.render_camera = config['render2loc']['render_camera']
        self.query_camera =  config['render2loc']['query_camera']
        self.outputs = config['render2loc']['results']
        self.save_loc_path = Path(self.dataset) / self.outputs / (f"{iter}_estimated_pose.txt")
        

    def interpolate_depth(self, pos, depth):
        ids = torch.arange(0, pos.shape[0])
        if depth.ndim != 2:
            depth = depth[:,:,0]
        h, w = depth.size()
        
        i = pos[:, 0]
        j = pos[:, 1]

        # Valid corners, check whether it is out of range
        i_top_left = torch.floor(i).long()
        j_top_left = torch.floor(j).long()
        valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

        i_top_right = torch.floor(i).long()
        j_top_right = torch.ceil(j).long()
        valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

        i_bottom_left = torch.ceil(i).long()
        j_bottom_left = torch.floor(j).long()
        valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

        i_bottom_right = torch.ceil(i).long()
        j_bottom_right = torch.ceil(j).long()
        valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

        valid_corners = torch.min(
            torch.min(valid_top_left, valid_top_right),
            torch.min(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corners]
        j_top_left = j_top_left[valid_corners]

        i_top_right = i_top_right[valid_corners]
        j_top_right = j_top_right[valid_corners]

        i_bottom_left = i_bottom_left[valid_corners]
        j_bottom_left = j_bottom_left[valid_corners]

        i_bottom_right = i_bottom_right[valid_corners]
        j_bottom_right = j_bottom_right[valid_corners]


        # Valid depth
        valid_depth = torch.min(
            torch.min(
                depth[i_top_left, j_top_left] > 0,
                depth[i_top_right, j_top_right] > 0
            ),
            torch.min(
                depth[i_bottom_left, j_bottom_left] > 0,
                depth[i_bottom_right, j_bottom_right] > 0
            )
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]

        # vaild index
        ids = ids[valid_depth]
        
        i = i[ids]
        j = j[ids]
        dist_i_top_left = i - i_top_left.float()
        dist_j_top_left = j - j_top_left.float()
        w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
        w_top_right = (1 - dist_i_top_left) * dist_j_top_left
        w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
        w_bottom_right = dist_i_top_left * dist_j_top_left

        #depth is got from interpolation
        interpolated_depth = (
            w_top_left * depth[i_top_left, j_top_left] +
            w_top_right * depth[i_top_right, j_top_right] +
            w_bottom_left * depth[i_bottom_left, j_bottom_left] +
            w_bottom_right * depth[i_bottom_right, j_bottom_right]
        )

        pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

        return [interpolated_depth, pos, ids]


    def read_valid_depth(self, mkpts1r, depth=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        depth = torch.tensor(depth).to(device)
        mkpts1r = torch.tensor(mkpts1r).to(device)
        mkpts1r_a = torch.unsqueeze(mkpts1r[:,0],0)
        mkpts1r_b =  torch.unsqueeze(mkpts1r[:,1],0)
        mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0).to(device)

        depth, _, valid = self.interpolate_depth(mkpts1r_inter , depth)

        return depth.cpu(), valid
    def get_query_intrinsic(self, camera):
        """
        计算35mm等效焦距和内参矩阵。
        
        参数:
        image_width_px -- 图像的宽度（像素）
        image_height_px -- 图像的高度（像素）
        sensor_width_mm -- 相机传感器的宽度（毫米）
        sensor_height_mm -- 相机传感器的高度（毫米）
        
        返回:
        K -- 内参矩阵，形状为3x3
        """
        image_width_px, image_height_px, fx, fy, cx, cy = camera
        # 计算内参矩阵中的焦距和主点坐标
        
        # 构建内参矩阵 K
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        
        return K, image_width_px, image_height_px
    def get_query_intrinsic_single_focal(self, camera):
        """
        计算35mm等效焦距和内参矩阵。
        
        参数:
        image_width_px -- 图像的宽度（像素）
        image_height_px -- 图像的高度（像素）
        sensor_width_mm -- 相机传感器的宽度（毫米）
        sensor_height_mm -- 相机传感器的高度（毫米）
        
        返回:
        K -- 内参矩阵，形状为3x3
        """
        image_width_px, image_height_px, fmm, cx, cy = camera
        # 计算内参矩阵中的焦距和主点坐标
        fx, fy = fmm, fmm
        # 构建内参矩阵 K
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        
        return K, image_width_px, image_height_px    
    def get_intrinsic(self, camera):
        """
        计算35mm等效焦距和内参矩阵。
        
        参数:
        image_width_px -- 图像的宽度（像素）
        image_height_px -- 图像的高度（像素）
        sensor_width_mm -- 相机传感器的宽度（毫米）
        sensor_height_mm -- 相机传感器的高度（毫米）
        
        返回:
        K -- 内参矩阵，形状为3x3
        """
        image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera[0], camera[1], camera[2], camera[3], camera[4]
        # 计算焦距在x和y方向上的比率
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm
        
        # 计算内参矩阵中的焦距和主点坐标
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y
        cx = image_width_px / 2
        cy = image_height_px / 2
        
        # 构建内参矩阵 K
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        
        return K

    def get_pose(self, render_pose):
        translation = render_pose[0]
        euler_angles = render_pose[1].copy()
        euler_angles[0] = euler_angles[0] - 180
        R_c2w = convert_euler_to_matrix(euler_angles)
        t_c2w = wgs84tocgcs2000(translation)
        
        # Initialize a 4x4 identity matrix
        T = np.identity(4)
        T[0:3, 0:3] = R_c2w
        T[0:3, 3] = t_c2w
        
        return T
    def get_Points3D(self, depth, R, t, K, points):   # points[n,2]
        '''
        depth, R, t, K, points: toprch.tensor
        return Point3D [n,3]: numpy.array
        '''
        if points.shape[-1] != 3:
            points_2D = torch.cat([points, torch.ones_like(points[ :, [0]])], dim=-1)
            points_2D = points_2D.T  
        t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
        # import pdb; pdb.set_trace()
        Points_3D = R @ K @ (depth * points_2D) + t   
        return (Points_3D.T).numpy().astype(np.float64)    #[3,n]
    def localize_using_pnp(self, points3D, points2D, query_camera, width, height):
        points3D = [points3D[i] for i in range(points3D.shape[0])]
        
        # import ipdb; ipdb.set_trace();
        fx, fy, cx, cy = query_camera[0][0], query_camera[1][1], query_camera[0][2], query_camera[1][2]
        cfg = {
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "params": [fx, fy, cx, cy],
        }  
        # import pdb; pdb.set_trace()
        # k1, k2, p1, p2 = -0.194506961008644,0.238274634658959,0, 0  #Deng
        # k1, k2, p1, p2 = -0.17891555, 0.113521386, 0.0009203404, -0.000358393 #CC
        # cfg = {
        #     "model": "OPENCV",
        #     "width": width,
        #     "height": height,
        #     "params": [fx, fy, cx, cy, k1, k2, p1, p2],
        # } 
        #     -0.194506961008644,
        #     0.238274634658959,
        #     0,
        #     0,
        #     -0.195437360791233
        ret = pycolmap.absolute_pose_estimation(
            points2D,
            points3D,
            cfg,
            estimation_options=self.config.get('estimation', {}),
            refinement_options=self.config.get('refinement', {}),
        )
        # import pdb; pdb.set_trace()
        
        
        return ret  

    def estimate_query_pose(self, matches, render_pose, depth_mat, query_camera=None):
        """
        Main function to perform localization on query images.

        Args:
            config (dict): Configuration settings.
            data (dict): Data related to queries and renders.
            iter (int): Iteration number for naming output files.
            outputs (Path): Path to the output directory.
            con (dict, optional): Additional configuration for the localization process.

        Returns:
            Path: Path to the saved estimated pose file.
        """
        if query_camera is not None:
            self.query_camera = query_camera
        
        # Get render intrinsics and query intrinsics
        query_K, width, height = self.get_query_intrinsic(self.query_camera)
        # query_K, width, height = self.get_query_intrinsic_single_focal(self.query_camera)
        
        
        # render_K = torch.tensor(query_K).float()
        render_K = torch.tensor(self.get_intrinsic(self.render_camera)).float()
        K_c2w = render_K.inverse()
        
        
        # query_K = render_K
        # Get render pose
        render_T = torch.tensor(self.get_pose(render_pose)).float()
        
        # Get 2D-2D matches
        mkpts_q = matches[0]
        mkpts_r = matches[1]
        
        depth, valid = self.read_valid_depth(mkpts_r, depth = depth_mat)
        # Compute 3D points
        Points_3D = self.get_Points3D(
            depth,
            render_T[:3, :3],
            render_T[:3, 3],
            K_c2w,
            torch.tensor(mkpts_r),
        )
        logger.info('Starting localization...')
        # Perform PnP to find camera pose
        points2D = mkpts_q[valid].cpu().numpy()
        ret = self.localize_using_pnp(Points_3D, points2D, query_K, width, height)
        logger.info('Done!')
        return ret
        
 
    

# if __name__=="__main__":
#     main()


