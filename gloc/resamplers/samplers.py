from typing import List
import torch
import numpy as np
from pyquaternion import Quaternion

from gloc.utils import rotmat2qvec, qvec2rotmat


class RandomConstantSampler():
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views, center_noise, angle_noise, old_t, old_R):
        qvecs = []
        tvecs = []
        poses = []

        for _ in range(n_views):
            new_tvec, new_qvec, new_T = self.sample(center_noise, angle_noise, old_t, old_R)

            qvecs.append(new_qvec)
            tvecs.append(new_tvec)
            poses.append(new_T)

        return tvecs, qvecs, poses

    @staticmethod
    def sample(center_noise, angle_noise, old_t, old_R, low_std_ax=1):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        r_axis = Quaternion.random().axis # sample random axis
        teta = angle_noise # sample random angle smaller than theta
        r_quat = Quaternion(axis=r_axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        old_center = - old_R.T @ old_t # get image center using original pose
        perturb_c = np.random.rand(2)
        perturb_low_ax = np.random.rand(1)*0.1
        perturb_c = np.insert(perturb_c, low_std_ax, perturb_low_ax)
        perturb_c /= np.linalg.norm(perturb_c) # normalize noise vector

        # move along the noise direction for a fixed magnitude
        new_center = old_center + perturb_c*center_noise  
        new_t = - new_R @ new_center # use the new pose to convert to translation vector

        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_T

class RandomGaussianSampler_origin():
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views, center_std, max_angle, old_t, old_R):
        qvecs = []
        tvecs = []
        poses = []
        
        for _ in range(n_views):
            new_tvec, new_qvec, new_T = self.sample(center_std, max_angle, old_t, old_R)

            qvecs.append(new_qvec)
            tvecs.append(new_tvec)
            poses.append(new_T)

        return tvecs, qvecs, poses

    @staticmethod
    def sample(center_std, max_angle, old_t, old_R):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        r_axis = Quaternion.random().axis # sample random axis
        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        r_quat = Quaternion(axis=r_axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        old_center = - old_R.T @ old_t # get image center using original pose
        perturb_c = torch.normal(0., center_std)
        new_center = old_center + np.array(perturb_c) # perturb it 
        new_t = - new_R @ new_center # use the new pose to convert to translation vector

        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_T   
    
class RandomGaussianSampler():
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views, center_std, max_angle, old_t, old_R, t_prior, R_prior):
        qvecs = []
        tvecs = []
        poses = []

        for i in range(n_views):
            # use first axis half the time, the other for the rest
            ax_i = i // ((n_views+1) // 2)
            # ax = self.rotate_axis[ax_i]  #改，只需要yaw
            # ax = self.rotate_axis[2]
            # new_tvec, new_qvec, new_T = self.sample(ax, center_std, max_angle, old_t, old_R)
            # axis_name = 'yaw'  #['yaw', 'pitch', 'roll']
            new_tvec, new_qvec, new_T = self.sample_degree_2(center_std, max_angle, old_t, old_R, t_prior, R_prior)
            
            qvecs.append(new_qvec)
            tvecs.append(new_tvec)
            poses.append(new_T)

        return tvecs, qvecs, poses

    @staticmethod
    def sample_degree_2(center_std, max_angle, old_t, old_R, t_prior, R_prior):
        # 进来都是W2C的

        euler_prior = cal_euler(t_prior, R_prior)

        # 利用角度采样 利用yaw轴
        teta = (np.random.rand() * 2 - 1) * max_angle
        new_R, _ = sample_rotation_yaw_roll_pitch(teta, old_t, old_R,euler_prior)

        # 采样 x, y, z
        old_center = - old_R.T @ old_t # get image center using original pose
        perturb_c = torch.normal(0., center_std)
        new_center = old_center + np.array(perturb_c) # perturb it 
        new_t = - new_R @ new_center # use the new pose to convert to translation vector

        new_qvec = rotmat2qvec(new_R)  # 将原始旋转矩阵转换为四元数

        # 构建新的位姿矩阵
        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_T
    
    @staticmethod
    def sample(center_std, max_angle, old_t, old_R):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        r_axis = Quaternion.random().axis # sample random axis
        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        r_quat = Quaternion(axis=r_axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        old_center = - old_R.T @ old_t # get image center using original pose
        perturb_c = torch.normal(0., center_std)
        new_center = old_center + np.array(perturb_c) # perturb it 
        new_t = - new_R @ new_center # use the new pose to convert to translation vector

        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_T


class RandomDoubleAxisSampler():
    rotate_axis = {
        'pitch': [1, 0, 0], # x, pitch
        'yaw':   [0, 1, 0]  # y, yaw
    }

    # # Blender中
    # rotate_axis = {
    #     'pitch': [1, 0, 0], # x, pitch
    #     'yaw':   [0, 0, 1]  # y, yaw
    # }
    
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views: int, center_std: torch.tensor, max_angle: torch.tensor, 
                     old_t: np.array, old_R: np.array):
        qvecs = []
        tvecs = []
        poses = []

        for _ in range(n_views):
            # apply yaw first
            ax = self.rotate_axis['yaw']            
            new_tvec, _, new_R, _ = self.sample(ax, center_std, float(max_angle[0]), old_t, old_R)

            # apply pitch then
            ax = self.rotate_axis['pitch']            
            new_tvec, new_qvec, _, new_T = self.sample(ax, center_std, float(max_angle[1]), new_tvec, new_R)

            qvecs.append(new_qvec)
            tvecs.append(new_tvec)
            poses.append(new_T)

        return tvecs, qvecs, poses

    @staticmethod
    def sample(axis, center_std: torch.tensor, max_angle: float,  old_t: np.array, old_R: np.array, rot_only: bool =False):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        r_quat = Quaternion(axis=axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        if not rot_only:
            old_center = - old_R.T @ old_t # get image center using original pose
            perturb_c = torch.normal(0., center_std)
            new_center = old_center + np.array(perturb_c) # perturb it 
            new_t = - new_R @ new_center # use the new pose to convert to translation vector
        else:
            new_t = - new_R @ old_center # use the new pose to convert to translation vector            
        
        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_R, new_T
    
from scipy.spatial.transform import Rotation as R
def sample_rotation_yaw_roll(axis_name, angle, old_t, old_R):
        T = np.identity(4)
        T[0:3,0:3] = old_R
        T[0:3,3] = old_t   # w2c
        T = np.linalg.inv(T)  #C2W
        transf = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1.],
        ])
        T = T @ transf
        R_gt = T[:3, :3]
        t_gt = T[:3, 3]
        ret_gt = R.from_matrix(R_gt)
        euler_gt = ret_gt.as_euler('xyz',degrees=True)  # Pitch Roll Yaw

        if axis_name == 'roll':
            euler_gt[1] = euler_gt[1] + angle
        elif axis_name == 'yaw':
            euler_gt[2] = euler_gt[2] + angle
        elif axis_name == 'pitch':
            euler_gt[0] = euler_gt[0] + angle
        
        ret_2 = R.from_euler('xyz', euler_gt, degrees=True)
        R_ = ret_2.as_matrix()
        T[0:3,0:3] = R_
        T[0:3,3] = t_gt
        T = T @ np.linalg.inv(transf)
        T = np.linalg.inv(T)

        rotation = T[0:3,0:3]
        traslation = T[0:3,3]
        # new_qvec = rotmat2qvec(rotation)

        return rotation, traslation

def cal_euler(old_t, old_R):
    T = np.identity(4)
    T[0:3,0:3] = old_R
    T[0:3,3] = old_t   # w2c
    T = np.linalg.inv(T)  #C2W
    transf = np.array([
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1.],
    ])
    T = T @ transf
    R_gt = T[:3, :3]
    t_gt = T[:3, 3]
    ret_gt = R.from_matrix(R_gt)
    euler_gt = ret_gt.as_euler('xyz',degrees=True)  # Pitch Roll Yaw
    return euler_gt

def sample_rotation_yaw_roll_pitch(angle, old_t, old_R, euler_prior):
    T = np.identity(4)
    T[0:3,0:3] = old_R
    T[0:3,3] = old_t   # w2c
    T = np.linalg.inv(T)  #C2W
    transf = np.array([
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1.],
    ])
    T = T @ transf
    R_gt = T[:3, :3]
    t_gt = T[:3, 3]
    ret_gt = R.from_matrix(R_gt)
    euler_gt = ret_gt.as_euler('xyz',degrees=True)  # Pitch Roll Yaw

    # if axis_name == 'roll':
    euler_gt[1] = euler_gt[1] + angle
    if abs(euler_prior[1]-euler_gt[1])>=1:
        euler_gt[1] = euler_gt[1] - angle
    # elif axis_name == 'yaw':
    euler_gt[2] = euler_gt[2] + angle
    # elif axis_name == 'pitch':
    euler_gt[0] = euler_gt[0] + angle
    if abs(euler_prior[0]-euler_gt[0])>=1:
        euler_gt[0] = euler_gt[0] - angle
    
    ret_2 = R.from_euler('xyz', euler_gt, degrees=True)
    R_ = ret_2.as_matrix()
    T[0:3,0:3] = R_
    T[0:3,3] = t_gt
    T = T @ np.linalg.inv(transf)
    T = np.linalg.inv(T)

    rotation = T[0:3,0:3]
    traslation = T[0:3,3]
    # new_qvec = rotmat2qvec(rotation)

    return rotation, traslation

class RandomSamplerByAxis():
    rotate_axis = [
        [1, 0, 0], # x,  Pitch
        [0, 1, 0], # y,   Roll
        [0, 0, 1] # z,    Yaw
    ]
    
    # #Blender 中
    # rotate_axis = [
    #     [1, 0, 0], # x, pitch
    #     [0, 0, 1] # y, yaw
    # ]
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views, center_std, max_angle, old_t, old_R):
        qvecs = []
        tvecs = []
        poses = []

        for i in range(n_views):
            # use first axis half the time, the other for the rest
            ax_i = i // ((n_views+1) // 2)
            # ax = self.rotate_axis[ax_i]  #改，只需要yaw
            ax = self.rotate_axis[2]
            # new_tvec, new_qvec, new_T = self.sample(ax, center_std, max_angle, old_t, old_R)
            axis_name = 'yaw'  #['yaw', 'pitch', 'roll']
            new_tvec, new_qvec, new_T = self.sample_degree(axis_name, center_std, max_angle, old_t, old_R)

            qvecs.append(new_qvec)
            tvecs.append(new_tvec)
            poses.append(new_T)

        return tvecs, qvecs, poses

    @staticmethod
    def sample(axis, center_std, max_angle, old_t, old_R):
        old_qvec = rotmat2qvec(old_R) # transform to qvec
        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        r_quat = Quaternion(axis=axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        old_center = - old_R.T @ old_t # get image center using original pose
        perturb_c = torch.normal(0., center_std)
        new_center = old_center + np.array(perturb_c) # perturb it 
        new_t = - new_R @ new_center # use the new pose to convert to translation vector

        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_T
    
    @staticmethod
    def sample_degree(axis_name, center_std, max_angle, old_t, old_R):
        # 进来都是W2C的

        # 利用角度采样 利用yaw轴
        teta = (np.random.rand() * 2 - 1) * max_angle
        new_R, _ = sample_rotation_yaw_roll(axis_name, teta, old_t, old_R)

        # 采样 x, y, z
        old_center = - old_R.T @ old_t # get image center using original pose
        perturb_c = torch.normal(0., center_std)
        new_center = old_center + np.array(perturb_c) # perturb it 
        new_t = - new_R @ new_center # use the new pose to convert to translation vector

        new_qvec = rotmat2qvec(new_R)  # 将原始旋转矩阵转换为四元数

        # 构建新的位姿矩阵
        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_T


class RandomAndDoubleAxisSampler():
    rotate_axis = {
        'pitch': [1, 0, 0], # x, pitch
        'yaw':   [0, 1, 0]  # y, yaw
    }

    # #Blender 渲染中
    # rotate_axis = {
    #     'pitch': [1, 0, 0], # x, pitch
    #     'roll':   [0, 1, 0],  # y, roll
    #     'yaw':   [0, 0, 1]  # z, yaw
    # }
    
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views: int, center_std: torch.tensor, max_angle: torch.tensor, 
                     old_t: np.array, old_R: np.array):
        qvecs = []
        tvecs = []
        poses = []

        for i in range(n_views):
            # use DoubleRotation half the time, Random the rest
            ax_i = i // ((n_views+1) // 2)
            if ax_i == 0:
                # double ax rotation
                
                # apply yaw first
                ax = self.rotate_axis['yaw']            
                new_tvec, _, new_R, _ = self.sample(ax, center_std, float(max_angle[0]), old_t, old_R)

                # apply pitch then
                ax = self.rotate_axis['pitch']            
                new_tvec, new_qvec, _, new_T = self.sample(ax, center_std, float(max_angle[1]), new_tvec, new_R)

                qvecs.append(new_qvec)
                tvecs.append(new_tvec)
                poses.append(new_T)
            else:
                # use random axis, with yaw magnitude
                new_tvec, new_qvec, _, new_T = self.sample(None, center_std, float(max_angle[0]), old_t, old_R)

                qvecs.append(new_qvec)
                tvecs.append(new_tvec)
                poses.append(new_T)                

        return tvecs, qvecs, poses

    @staticmethod
    def sample(axis, center_std: torch.tensor, max_angle: float,  old_t: np.array, old_R: np.array, rot_only: bool =False):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        if axis is None:
            # if no axis provided, use a random one
            axis = Quaternion.random().axis # sample random axis

        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        r_quat = Quaternion(axis=axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        if not rot_only:
            old_center = - old_R.T @ old_t # get image center using original pose
            perturb_c = torch.normal(0., center_std)
            new_center = old_center + np.array(perturb_c) # perturb it 
            new_t = - new_R @ new_center # use the new pose to convert to translation vector
        else:
            new_t = - new_R @ old_center # use the new pose to convert to translation vector            
        
        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_R, new_T
