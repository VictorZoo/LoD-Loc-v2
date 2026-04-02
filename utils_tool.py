import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from functools import reduce
import torch
transf = np.array([
                    [1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1.],
                ])
def visualize_descriptors(descriptors):
    """
    可视化高维描述子张量的每个 batch.

    参数:
    descriptors (numpy.ndarray): 形状为 [batch, dim, H, W] 的描述子张量.

    返回:
    None
    """
    # 检查输入形状
    assert len(descriptors.shape) == 4, "输入的描述子张量形状必须为 [batch, dim, H, W]"
    
    # 转换为 [batch, H, W, dim]
    descriptors = np.transpose(descriptors, (0, 2, 3, 1))
    
    # 使用 PCA 将 256 维降到 3 维
    pca = PCA(n_components=3)
    batch_size, H, W, dim = descriptors.shape
    
    # 将每个批次的每个特征向量降到 3 维
    reduced_descriptors = np.zeros((batch_size, H, W, 3))
    for i in range(batch_size):
        flattened = descriptors[i].reshape(-1, dim)
        reduced = pca.fit_transform(flattened)
        reduced_descriptors[i] = reduced.reshape(H, W, 3)
    
    # 将降维后的特征向量映射到 [0, 255] 的范围，并转换为 uint8 类型
    reduced_descriptors = (255 * (reduced_descriptors - np.min(reduced_descriptors)) / np.ptp(reduced_descriptors)).astype(np.uint8)
    
    # 可视化每一个 Batch
    for i in range(batch_size):
        plt.figure(figsize=(10, 5))
        plt.imshow(reduced_descriptors[i])
        plt.title(f'Batch {i+1}')
        plt.axis('off')
        breakpoint()
        plt.show()

def get_t_euler(pose_batch):
    pose_batch = np.linalg.inv(pose_batch)  #C2W
    # 应用变换矩阵到整个批量的pose
    pose_batch = pose_batch @ transf

    # 提取批量的旋转矩阵和位移向量
    initial_R_batch = pose_batch[:, :3, :3]
    initial_xyz_batch = pose_batch[:, :3, 3]

    # 将旋转矩阵转换为欧拉角
    ret_init_batch = R.from_matrix(initial_R_batch)
    initial_euler_batch = ret_init_batch.as_euler('xyz', degrees=True)

    # 提取平移和欧拉角信息
    t_batch = initial_xyz_batch
    euler_batch = initial_euler_batch

    return t_batch, euler_batch

def trans_eulerTo4x4(combined_samples):
    degree, xyz = combined_samples[...,:3], combined_samples[...,3:]
    n, _ = degree.shape
    ret_2 = R.from_euler('xyz', degree, degrees=True)
    R_ = ret_2.as_matrix()
    T = torch.eye(4)
    T = T.unsqueeze(0).expand(n, -1, -1).clone()
    T[:,0:3,0:3] = torch.from_numpy(R_)
    T[:,0:3,3] = xyz
    T = T @ torch.inverse(torch.tensor(transf).to(T.dtype))# C2W

    return T

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def get_r_name(q_name, r_i, beam_i):
    r_name = q_name+f'_{r_i}beam{beam_i}'
    return r_name
    
def sample_poses(initial_xyz, initial_R, Yawxyz_error_ranges, num_samples_Yawxyz, q_name, beam_i):
    render_qvecs = []
    render_ts = []
    r_names = []
    if (len(Yawxyz_error_ranges.shape)) == 2:
        Yawxyz_error_ranges = Yawxyz_error_ranges.unsqueeze(0).expand(initial_xyz.shape[0],-1,-1)
    
    # Yawxyz_error_ranges = Yawxyz_error_ranges.to(initial_xyz.dtype)
    device = initial_xyz.device
    initial_xyz = initial_xyz.cpu()
    initial_R = initial_R.cpu()
    b, _ = initial_xyz.shape
    num_sample = reduce(lambda x, y: x * y, num_samples_Yawxyz)
    stacked_tensor = torch.empty((0,num_sample,4,4))
    stacked_euler = torch.empty((0,num_sample,6))

    ret_init = R.from_matrix(initial_R)
    initial_euler = torch.from_numpy(ret_init.as_euler('xyz',degrees=True))
    pitch, roll, initial_yaw = torch.unbind(initial_euler, dim=-1)
    
    # 4自由度 For CrossLoc
    # pitch[0] = pose_GT[0]#改
    # roll[0] = pose_GT[1]

    # 在误差范围内均匀采样
    yaw_error_range, xyz_error_ranges = Yawxyz_error_ranges[:,0], Yawxyz_error_ranges[:,1:]
    num_samples_yaw, num_samples_xyz = num_samples_Yawxyz[0], num_samples_Yawxyz[1:]
    # 在batch循环采样，
    for j in range(b):
        xyz_samples = [
            torch.linspace(
                initial_xyz[j,i] + float(xyz_error_ranges[j,i,0]),
                initial_xyz[j,i] + float(xyz_error_ranges[j,i,1]),
                num_samples_xyz[i]
            ) for i in range(3)]
        yaw_samples = torch.linspace(initial_yaw[j] + float(yaw_error_range[j,0]), initial_yaw[j] + float(yaw_error_range[j,1]), num_samples_yaw)
        
        pitch = pitch.to(yaw_samples[0].dtype)
        roll = roll.to(yaw_samples[0].dtype)

        pitch_samples, roll_samples, yaw_samples, xyz_samples_0, xyz_samples_1, xyz_samples_2 = torch.meshgrid(pitch[j], roll[j], yaw_samples, *xyz_samples)

        combined_samples = torch.stack([pitch_samples, roll_samples, yaw_samples, xyz_samples_0, xyz_samples_1, xyz_samples_2], dim=-1)
        combined_samples = combined_samples.squeeze()

        T_ = trans_eulerTo4x4(combined_samples.view(-1, 6)) #C2W
        T_ = torch.inverse(T_) #W2C
        T_ = T_.unsqueeze(0)
        stacked_tensor = torch.cat([stacked_tensor, T_], dim=0) #W2C
        stacked_euler = torch.cat([stacked_euler, combined_samples.view(-1, 6).unsqueeze(0)], dim=0) #C2W

    stacked_tensor = np.array(stacked_tensor.to(device).squeeze())
    stacked_euler = np.array(stacked_euler.to(device).squeeze())

    rotation_matrices = stacked_tensor[:, :3, :3]
    translations = stacked_tensor[:, :3, 3]

    for r_i in range(stacked_euler.shape[0]):
        r_names.append(get_r_name(q_name, r_i, beam_i))
        qvec_new = rotmat2qvec(rotation_matrices[r_i])
        render_qvecs.append(qvec_new)
        render_ts.append(translations[r_i])
    
    return r_names, stacked_tensor, stacked_euler, render_ts, render_qvecs

