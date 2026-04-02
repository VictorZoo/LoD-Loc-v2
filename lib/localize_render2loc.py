import os
import cv2
from . import logger
import time
import pycolmap
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Union
from lib.read_model import parse_image_lists, parse_db_intrinsic_list, parse_pose_list

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class QueryLocalizer:
    def __init__(self, config=None):
        self.config = config
        # 初始化其他需要的成员变量
        self.dataset = Path(self.config['render2loc']['datasets'])
        self.render_camera = self.dataset / self.config['render2loc']['render_camera']
        self.query_camera =  self.dataset / self.config['render2loc']['query_camera']
        self.outputs = self.dataset / self.config['render2loc']['results']
        
        self.K_q = parse_image_lists(self.query_camera, with_intrinsics=True, simple_name=False)
        # Parse the render camera file to get the render camera intrinsics
        self.K_r = parse_db_intrinsic_list(self.render_camera)

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

    # read depth
    def read_valid_depth(self, depth_exr,mkpts1r, depth_mat = None):
        if depth_mat is not None:
            depth = depth_mat
        else:
            depth = cv2.imread(str(depth_exr), cv2.IMREAD_UNCHANGED)
        
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
        image_width_px, image_height_px, f_mm, cx, cy = camera
        # 计算内参矩阵中的焦距和主点坐标
        fx = f_mm
        fy = f_mm
        
        # 构建内参矩阵 K
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        
        return K
    
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
        image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera
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

    def get_Points3D(self, depth, R, t, K, points):   # points[n,2]
        '''
        depth, R, t, K, points: toprch.tensor
        return Point3D [n,3]: numpy.array
        '''
        if points.shape[-1] != 3:
            points_2D = torch.cat([points, torch.ones_like(points[ :, [0]])], dim=-1)
            points_2D = points_2D.T  
        t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
        Points_3D = R @ K @ (depth * points_2D) + t   
        return (Points_3D.T).numpy().astype(np.float64)    #[3,n]
        
        

    def localize(self, points3D, points2D, query_camera):
        points3D = [points3D[i] for i in range(points3D.shape[0])]
        # import ipdb; ipdb.set_trace();
        fx, fy, cx, cy = query_camera.params
        cfg = {
            "model": "PINHOLE",
            "width": query_camera.width,
            "height": query_camera.height,
            "params": [fx, fy, cx, cy],
        }  
        
        ret = pycolmap.absolute_pose_estimation(
            points2D,
            points3D,
            cfg,
            estimation_options=self.config.get('estimation', {}),
            refinement_options=self.config.get('refinement', {}),
        )
        
        return ret  


    def main(
        self,
        data: Dict,
        iter: int,
        con: Dict = None,
        depth_mat = None,
        ):
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
        save_loc_path = self.outputs / (f"{iter}_estimated_pose.txt")
        # Unpack data
        matches = data["matches"]
        render_poses = data["render_pose"]
        
        # Progress bar setup
        pbar = tqdm(total=len(self.K_q), unit='pts')
        
        # Configuration setup with default and user-provided settings
        con = {"estimation": {"ransac": {"max_error": 12}}, **(con or {})}
        poses = {}

        logger.info('Starting localization...')
        t_start = time.time()

        with open(save_loc_path, 'w') as f:
            for qname, query_camera in tqdm(self.K_q): 
                match = matches.get(qname.split('/')[-1])

                imgr_name = match["imgr_name"]
                depth_exr_final = match["exrr_pth"]
                max_correct = match["correct"]
                mkpts_r = match["mkpts_r"]
                mkpts_q = match["mkpts_q"]
                
                if depth_exr_final is None or max_correct == 0:
                    print("No match found for", qname)
                    qvec, tvec = 0, 0
                    qvec = ' '.join(map(str, qvec))
                    tvec = ' '.join(map(str, tvec))
                    name = qname.split('/')[-1]
                    f.write(f'{name} {qvec} {tvec}\n') 
                    continue
                
                # Read depth and valid points
                depth, valid = self.read_valid_depth(depth_exr_final, mkpts_r, depth_mat = depth_mat)
                
                # Get render pose
                render_pose = torch.tensor(render_poses[imgr_name.split('.')[0]]).float()
                
                # Compute 3D points
                K_w2c = torch.tensor(self.K_r).float()
                K_c2w = K_w2c.inverse()
                Points_3D = self.get_Points3D(
                    depth,
                    render_pose[:3, :3],
                    render_pose[:3, 3],
                    K_c2w,
                    torch.tensor(mkpts_r),
                )
                
                # Perform PnP to find camera pose
                points2D = mkpts_q[valid].cpu().numpy()
                ret = self.localize(Points_3D, points2D, query_camera)
                
                if ret['success']:
                    poses[qname] = (ret['qvec'], ret['tvec'])
                    
                    # Write successful poses to file
                    qvec = ' '.join(map(str, ret['qvec']))
                    tvec = ' '.join(map(str, ret['tvec']))
                    name = qname.split('/')[-1]
                    f.write(f'{name} {qvec} {tvec}\n')
        
        t_end = time.time()
        logger.info(f'Localize uses {t_end - t_start} seconds.')
        logger.info(f'Localized {len(poses)} / {len(self.K_q)} images.')
        logger.info(f'Writing poses to {save_loc_path}...')

        # Additional logging or processing can be done here if needed

        logger.info('Done!')
        pbar.close()

        # Return the path to the saved poses file
        return save_loc_path



