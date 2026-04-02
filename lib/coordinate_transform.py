import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from .transform import qvec2rotmat,rotmat2qvec
def get_hms_w2c():
    # 欧拉角转四元数
    yaw,pitch,roll = 170, 0, 0  #w2c
    x, y, z = 401786.87, 3131345.4, 19.368  #c2w
    euler = [yaw,pitch,roll]
    ret = R.from_euler('zxy',[float(euler[0]), 90-float(euler[1]), float(euler[2])],degrees=True)
    R_matrix = ret.as_matrix()
    qw, qx, qy, qz = rotmat2qvec(R_matrix)


    q = [qw, qx, qy, qz]  #w2c
    R1 = np.asmatrix(qvec2rotmat(q)) #w2c
    T = np.identity(4)
    T[0:3, 0:3] = R1
    T[0:3, 3] = -R1.dot(np.array([x, y, z]))#w2c
    return T

def get_plane2hms(w2a):
    # get air to world coordinates
    save_path = '/home/ubuntu/Documents/code/github/Render2loc/datasets/demo4/results/thermal/a2w.txt'
    with open(save_path, 'w') as file_w:
        with open(w2a, 'r') as f:
            for data in f.read().rstrip().split('\n'):
                data = data.split()
                name = os.path.basename(data[0])
                q, t = np.split(np.array(data[1:], float), [4])
                R = np.asmatrix(qvec2rotmat(q)).transpose()  #c2w
                
                T = np.identity(4)
                T[0:3,0:3] = R
                T[0:3,3] = -R.dot(t)   #!  a2w      
                
                Tw2h = get_hms_w2c()   # w2h
                # calculate a2h
                Ta2h = np.dot(T, Tw2h)
                
                # # w2c
                # T2[0:3,0:3] = T2[0:3,0:3].transpose() 
                # T2[0:3,3] = -T2[0:3,0:3].dot(T2[0:3,3]) 

                # euler_xyz[2] = -euler_xyz[2]

                # euler to matrix
                
                quat2 = rotmat2qvec(Ta2h[:3, :3])
                t2 = Ta2h[:3, 3]

                out_line = [str(data[0])] + list(quat2)+[t2[0]]+[t2[1]]+[t2[2]]
                out_line_str  = str(data[0])+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' '+str(out_line[4])+' '+str(out_line[5])+' '+str(out_line[6])+' '+str(out_line[7])+' \n'
                
                file_w.write(out_line_str)
            
            
            