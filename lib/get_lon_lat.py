import pyproj
import numpy as np
from scipy.spatial.transform import Rotation as R
from lib.transform import  qvec2rotmat,rotmat2qvec
import os
def decimal_to_dms(decimal):
    """
    Convert decimal degrees to degrees, minutes and seconds.
    
    Args:
        decimal (float): The decimal degrees value to convert.

    Returns:
        (int, int, float): A tuple containing degrees, minutes, and seconds.
    """
    # Convert decimal to degrees and the remaining fraction
    degrees = int(decimal)
    fraction = decimal - degrees
    
    # Convert the fraction to minutes
    minutes_full = fraction * 60
    minutes = int(minutes_full)
    # The remaining fraction becomes seconds
    seconds = (minutes_full - minutes) * 60
    
    return degrees, minutes, seconds

def dms_to_string(degrees, minutes, seconds, direction):
    """
    Format the degrees, minutes, and seconds into a DMS string.
    
    Args:
        degrees (int): The degrees part of the DMS.
        minutes (int): The minutes part of the DMS.
        seconds (float): The seconds part of the DMS.
        direction (str): The direction (N/S/E/W).

    Returns:
        str: A string representing the DMS in format "D°M'S".
    """
    # Format seconds to ensure it has three decimal places
    seconds = round(seconds, 3)
    # Create the DMS string
    dms_string = f"{degrees}°{minutes}'{seconds}\" {direction}"
    return dms_string
def convert_quaternion_to_euler(q_c2w):
    """
    Convert a quaternion to Euler angles in 'xyz' order, with angles in degrees.
    
    Parameters:
    qvec (numpy.ndarray): The quaternion vector as [x, y, z, w].
    R (numpy.matrix, optional): If provided, this rotation matrix will be used
                                 instead of calculating it from the quaternion.
    
    Returns:
    numpy.ndarray: The Euler angles in 'xyz' order in degrees.
    """
    # Convert the quaternion in COLMAP format [QW, QX, QY, QZ] 
    # to the correct order for scipy's Rotation object: [x, y, z, w]
    q_xyzw = [float(q_c2w[1]), float(q_c2w[2]), float(q_c2w[3]), float(q_c2w[0])] 
    
    # Create a Rotation object from the quaternion
    ret = R.from_quat(q_xyzw)
    
    # Convert the quaternion to Euler angles in 'xyz' order, with angles in degrees
    euler_xyz = ret.as_euler('xyz', degrees=True)    
    euler_xyz[0] = euler_xyz[0] + 180 
    # TODO
    return list(euler_xyz)
def convert_euler_to_matrix(euler_xyz):
    """
    Convert Euler angles in 'xyz' order to a rotation matrix.
    
    Parameters:
    euler_xyz (list or numpy.ndarray): The Euler angles in 'xyz' order in degrees.
    
    Returns:
    numpy.ndarray: The rotation matrix as a 3x3 numpy array.
    """
    # Convert the Euler angles from degrees to radians
    euler_xyz[0] = euler_xyz[0] - 180 
    euler_xyz_rad = np.radians(euler_xyz)
    
    # Create a Rotation object from the Euler angles
    rotation = R.from_euler('xyz', euler_xyz_rad)
    
    # Convert the rotation to a matrix
    rotation_matrix = rotation.as_matrix()
    
    return rotation_matrix
def get_pose(euler_angles, translation):
        euler_angles[0] = euler_angles[0] 
        R_c2w = convert_euler_to_matrix(euler_angles)
        t_c2w = wgs84tocgcs2000(translation)
        
        # Initialize a 4x4 identity matrix
        T = np.identity(4)
        T[0:3, 0:3] = R_c2w
        T[0:3, 3] = t_c2w
        print("after: ", T)
        
        return T
def parse_pose(path):
    """Parse pose list from a file and convert coordinates.
    
    Args:
        path (str): The file path containing the pose list.
    """
    data_output = []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = (data[0].split('/')[-1])
            q_w2c, t_w2c = np.split(np.array(data[1:], dtype=float), [4])
            
            qmat = qvec2rotmat(q_w2c)
            qmat = qmat.T
            q_c2w = rotmat2qvec(qmat)
            
            euler_angles = convert_quaternion_to_euler(q_c2w)            
            # Convert quaternion to rotation matrix and Transform the translation vector
            R_c2w = np.asmatrix(qvec2rotmat(q_w2c)).transpose()  
            t_c2w = np.array(-R_c2w.dot(t_w2c)) 
            print("before: ", R_c2w, t_c2w) 
            
            # Convert coordinates from CGCS2000 to WGS84.
            xyz = cgcs2000towgs84(t_c2w)
            print(name)
            get_pose(euler_angles, xyz)
            print("Euler angles in 'xyz' order (in degrees):", euler_angles)
            print("Translation in WGS84:", xyz)
            data_output.append(tuple([name]) + tuple(euler_angles)+ tuple(xyz))
            
    # 打开一个文件用于写入
    output_path = os.path.dirname(path)
    output_file = os.path.join(output_path, 'prior_pose_euler.txt')
    with open(output_file, 'w') as file:
        for item in data_output:
            # 将元组中的数值转换为字符串，并按照指定格式拼接
            line = ' '.join(map(str, item))
            # 写入文件，并在每行后添加换行符
            file.write(line + '\n')


def cgcs2000towgs84(c2w_t):
    """Convert coordinates from CGCS2000 to WGS84.
    
    Args:
        c2w_t (list): [x, y, z] in CGCS2000 format
    """
    x, y = c2w_t[0][0], c2w_t[0][1]
    
    wgs84 = pyproj.CRS('EPSG:4326')
    cgcs2000 = pyproj.CRS('EPSG:4547')  
    
    transformer = pyproj.Transformer.from_crs(cgcs2000, wgs84, always_xy=True)
    lon, lat = transformer.transform(x, y)
    height = c2w_t[0][2]
    return [lon, lat, height]

def wgs84tocgcs2000(trans):
    """Convert coordinates from WGS84 to CGCS2000.
    
    Args:
        trans (list): [lon, lat, height] in WGS84 format
    """
    lon, lat, height = trans  # Unpack the WGS84 coordinates
    
    wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 CRS definition
    cgcs2000 = pyproj.CRS('EPSG:4547')  # CGCS2000 CRS definition
    
    # Create a transformer from WGS84 to CGCS2000
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)
    
    # Perform the transformation
    x, y = transformer.transform(lon, lat)
    
    # Return the transformed coordinates as a list
    return [x, y, height]  # Keep the original height from WGS84        
        
def main(input_pose):
    parse_pose(input_pose)


if __name__ == "__main__":
    input_pose = "/home/ubuntu/Documents/code/Render2loc/datasets/demo8/sensors_prior/prior_pose.txt"
    
    main(input_pose)
    
    
    
    
    
    