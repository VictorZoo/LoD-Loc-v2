import sys
# sys.path.append("./3DTilesRender/build_obj_noTrans/")

# from ModelRenderScene import ModelRenderScene
import json
import numpy as np
from lib import (
    transform,
)
from lib.transform import qvec2rotmat, rotmat2qvec

class RealTime_render:
    def __init__(self, config_file='configs/config_local.json'):
        with open(config_file) as fp:
            self.config = json.load(fp)
        
        self.euler_angles = 0.  #相机角度
        self.translation = 0.   #位置  
        
    # def receive_from_web_server(self, config_file, config_web = None):
        
    #     if config_web is None:
    #         with open(config_file) as web_fp:
    #             config_web = json.load(web_fp)
    #     self.euler_angles = config_web['euler_angles']  #相机角度
    #     self.translation = config_web['translation']   #位置  

    # def update_render_pose(self, ret):
    #     q_w2c = ret['qvec']
    #     t_w2c = ret['tvec']
    #     qmat = qvec2rotmat(q_w2c)
    #     qmat = qmat.T
    #     q_c2w = rotmat2qvec(qmat)
    #     self.euler_angles = transform.convert_quaternion_to_euler(q_c2w)
    #     # Convert quaternion to rotation matrix and Transform the translation vector
    #     R_c2w = np.asmatrix(qvec2rotmat(q_w2c)).transpose()  
    #     t_c2w = np.array(-R_c2w.dot(t_w2c))  
    #     # Convert coordinates from CGCS2000 to WGS84.get_query_intrinsic
    #     self.translation = transform.cgcs2000towgs84(t_c2w)
    #     print("Euler angles in 'xyz' order (in degrees):", self.euler_angles[0], self.euler_angles[1], 360-self.euler_angles[2])
    #     print("Translation in WGS84:", self.translation)
    # def delay_to_load_map(self, config_web):
    #     self.euler_angles = config_web['euler_angles']  #相机角度
    #     self.translation = config_web['translation']   #位置  
    #     for i in range(500):
    #         self.renderer.update_pose(self.translation, self.euler_angles)
    # def rendering(self, config_web):
    #     self.image_id = config_web['image_id']
    #     # self.query_path = config_web['query_path']
    #     # self.query_image = cv2.imread(self.query_path, cv2.IMREAD_GRAYSCALE) # query image path

    #     self.euler_angles = config_web['euler_angles']  #相机角度
    #     self.translation = config_web['translation']   #位置  
        
    #     self.renderer.update_pose(self.translation, self.euler_angles)
    #     color_image = self.renderer.get_color_image()
    #     mask = np.all(color_image == [255,255,255], axis=-1).astype(np.uint8)
        
    #     # self.renderer.save_color_image(str(self.outputs/"normalFOV_images"/self.image_id))
    #     return mask

    def get_pose(self, render_pose):
        translation = render_pose[0]
        euler_angles = render_pose[1].copy()
        # import pdb;pdb.set_trace()
        euler_angles[0] = euler_angles[0]#180
        R_c2w = transform.convert_euler_to_matrix(euler_angles)
        t_c2w = transform.wgs84tocgcs2000(translation)
        
        # Initialize a 4x4 identity matrix
        T = np.identity(4)
        T[0:3, 0:3] = R_c2w
        T[0:3, 3] = t_c2w
        
        return T
    
    def get_pose_c2w(self, render_pose):
        translation = render_pose[0]
        euler_angles = render_pose[1].copy()
        euler_angles[0] = euler_angles[0]#180
        R_c2w = transform.convert_euler_to_matrix(euler_angles)
        # t_c2w = transform.wgs84tocgcs2000(translation)
        t_c2w = translation
        
        # Initialize a 4x4 identity matrix
        T = np.identity(4)
        T[0:3, 0:3] = R_c2w
        T[0:3, 3] = t_c2w
        
        return R_c2w, t_c2w
    
    def get_pose_w2c(self, render_pose):
        translation = render_pose[0]
        euler_angles = render_pose[1].copy()
        euler_angles[0] = euler_angles[0] - 180
        
        R_c2w = transform.convert_euler_to_matrix(euler_angles)
        # t_c2w = transform.wgs84tocgcs2000(translation)
        t_c2w = translation
        
        # Initialize a 4x4 identity matrix
        T = np.identity(4)
        T[0:3, 0:3] = R_c2w
        T[0:3, 3] = t_c2w
        T = np.linalg.inv(T)
        R_w2c, t_w2c = T[0:3, 0:3], T[0:3, 3]
        return R_w2c, t_w2c
    
    def get_pose_w2cToWGS84(self, w2c_pose):
        
        c2w_pose = np.linalg.inv(w2c_pose)  #c2w
        R_c2w = c2w_pose[0:3, 0:3]
        t_c2w = c2w_pose[0:3, 3]
        q_c2w = rotmat2qvec(R_c2w)
        translation = transform.cgcs2000towgs84_2(t_c2w)
        euler_angles = transform.convert_quaternion_to_euler(q_c2w)
        
        return translation, euler_angles
    
    def get_pose_w2cToWGS84_batch(self, w2c_pose):
        
        w2c_pose = np.array(w2c_pose)
        c2w_pose = np.linalg.inv(w2c_pose)  #c2w
        R_c2w = c2w_pose[:, 0:3, 0:3]
        t_c2w = c2w_pose[:, 0:3, 3]

        # q_c2w = rotmat2qvec(R_c2w)
        # translation = transform.cgcs2000towgs84_batch(t_c2w)
        translation = t_c2w
        euler_angles = transform.convert_quaternion_to_euler_batch(R_c2w)
        
        return translation, euler_angles
        