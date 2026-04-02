from osgeo import gdal
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path
from lib.transform import qvec2rotmat,rotmat2qvec


def generate_query_with_seeds(dsm_filepath, save_filepath, query_sequence, dsm = False):
    """
    Preprocess query images by adding seeds for prior pose and adjusting height based on DSM value.
    1. add 15 seeds for prior pose(delta yaw: [0, 30°, -30°], [delta x, delta y]: [[0, 0], [0, 5], [0, -5], [5, 0], [-5, 0]]
    2. height is always inaccurate, set height = dsm value + 1.5
    Args:
        dsm_filepath (str): Path to the DSM file.
        save_filepath (str): Path to save the processed query sequence.
        query_sequence (str): Path to the original query sequence file.
        dev (str): Device type, 'phone' by default.
    
    Returns:
        None
    """
    
    # Define translation deltas for seeds
    delta = [[0, 0, 0], [0, 5, 0], [0, -5, 0], [5, 0, 0], [-5, 0, 0]]
    
    with open(query_sequence, 'r') as f:
        with open(save_filepath, 'w') as file_w:
            for data in f.read().rstrip().split('\n'):
                data = data.split()
                name_raw = data[0].split('/')[-1]
                
                # Split the data into the quaternion and translation parts
                q, t = np.split(np.array(data[1:], dtype=float), [4])
                
                # Convert quaternion to rotation matrix and adjust translation
                R = np.asmatrix(qvec2rotmat(q)).transpose()  # Camera-to-world
                t = np.array(-R.dot(t))
                
                # Set height based on DSM value plus a constant offset
                d = 1.5  # Height offset in meters

                index = 0
                # Generate XY seeds
                for i in range(len(delta)):
                    # Calculate new x, y with delta
                    x, y, _ = t[0] + delta[i]
                    
                    if dsm:
                        # Open the DSM file
                        dataset = gdal.Open(dsm_filepath)
                        
                        # Get geotransformation parameters
                        geotrans = dataset.GetGeoTransform()
                        originX = geotrans[0]
                        originY = geotrans[3]
                        pixelWidth = geotrans[1]
                        pixelHeight = geotrans[5]
                        
                        # Get the DSM band
                        band = dataset.GetRasterBand(1)
                        # Calculate pixel offsets and read DSM value
                        xOffset = int((x - originX) / pixelWidth)
                        yOffset = int((y - originY) / pixelHeight)
                        z = band.ReadAsArray(xOffset, yOffset, 1, 1)[0, 0] + d
                    else:
                        z = t[0][2]  # Use original height if not 'phone'
                    
                    # Prepare the camera-to-world translation
                    t_c2w = np.array([x, y, z])
                    
                    # Generate YAW seeds
                    qvec1, qvec2 = generate_yaw_seeds(q)
                    
                    # Write the processed data to the output file
                    for _, q_seed in enumerate([q, qvec1, qvec2]):
                        R = np.asmatrix(qvec2rotmat(q_seed))  # World-to-camera
                        t_w2c = np.array(-R.dot(t_c2w))
                        out_line_str = f"{name_raw[:-4]}_{index}.jpg " + \
                            ' '.join(map(str, q_seed)) + f" {t_w2c[0][0]} {t_w2c[0][1]} {t_w2c[0][2]} \n"
                        file_w.write(out_line_str)
                        index += 1
    
    print("Done with writing pose.txt")


def generate_yaw_seeds(q):
    """
    Generate two yaw-adjusted quaternions from a given quaternion by adding and subtracting 30 degrees.
    
    Args:
        qvec (list): A list representing the quaternion in the order [x, y, z, w].
    
    Returns:
        tuple: Two tuples, each containing a list that represents a quaternion.
    """
    # Convert the quaternion to the correct order for scipy's Rotation object: [w, x, y, z]
    qmat = qvec2rotmat(q)
    qmat = qmat.T
    qvec = rotmat2qvec(qmat)  #!w2c
    
    
    qv = [float(qvec[1]),float(qvec[2]), float(qvec[3]), float(qvec[0])]
    
    # Create a Rotation object from the quaternion
    ret = R.from_quat(qv)
    
    # Convert the quaternion to Euler angles in 'xyz' order, with angles in degrees
    euler_xyz = ret.as_euler('xyz', degrees=True)
    
    # Create two sets of Euler angles with yaw angles increased and decreased by 30 degrees
    euler_xyz_2 = euler_xyz.copy()
    euler_xyz[2] += 30
    euler_xyz_2[2] -= 30
    
    # Convert the modified Euler angles back to rotation matrices
    ret_1 = R.from_euler('xyz', euler_xyz, degrees=True)
    ret_2 = R.from_euler('xyz', euler_xyz_2, degrees=True)
    
    # Convert the rotation matrices to quaternions
    new_matrix1 = ret_1.as_matrix()
    new_matrix2 = ret_2.as_matrix()
    
    new_qvec1 = rotmat2qvec(new_matrix1.T)
    new_qvec2 = rotmat2qvec(new_matrix2.T)
    
    # Return the two new quaternions
    return new_qvec1, new_qvec2      
def yaw_seed(qvec):
    
    qv = [float(qvec[1]),float(qvec[2]), float(qvec[3]), float(qvec[0])]
    
    ret = R.from_quat(qv)
    euler_xyz = ret.as_euler('xyz', degrees=True)
    
    euler_xyz_2 = ret.as_euler('xyz', degrees=True)

    euler_xyz[2] = euler_xyz[2] + 30
    euler_xyz_2[2] = euler_xyz_2[2] - 30
    
    # # euler to matrix
    ret_1 = R.from_euler('xyz', euler_xyz, degrees=True)
    ret_2 = R.from_euler('xyz', euler_xyz_2, degrees=True)
    new_matrix1 = ret_1.as_matrix()
    new_matrix2 = ret_2.as_matrix()
    
    new_qvec1 = rotmat2qvec(new_matrix1.T)
    new_qvec2 = rotmat2qvec(new_matrix2.T)  
    return new_qvec1, new_qvec2

def main(dsm_file: str, seed_path: Path, prior_path: Path, dsm=False):
    """
    Main function to check if seeds exist and generate them if necessary.

    Args:
        dsm_file (str): Path to the DSM file used for height correction.
        seed_path (Path): Path to the directory where the seeds will be saved.
        prior_path (Path): Path to the directory where prior information is stored.
        dev (str): Optional device identifier. Defaults to an empty string if not provided.
        dsm (bool): Optional set height or a prior height , Defaults to prior height if not provided.
    Returns:
        None
    """
    # If the directory does not exist, generate the query with seeds
    generate_query_with_seeds(dsm_file, seed_path, prior_path, dsm)
