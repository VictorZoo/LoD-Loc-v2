import numpy as np
from .transform import qvec2rotmat
import os
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict
import pycolmap
logger = logging.getLogger(__name__)

def parse_intrinsic_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            
            data_line=line.split(' ')
            name = data_line[0].split('/')[-1]
            _,_,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]  
            
            K_w2c = np.array([ #!
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            images[name] = K_w2c
  
    return images
def parse_pose_list(path):
    """
    Parses a pose list file and converts the poses into 4x4 transformation matrices.

    Args:
        path (str): Path to the pose list file.

    Returns:
        dict: Dictionary of poses with names as keys and 4x4 transformation matrices as values.
    """
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            tokens = data.split()
            name = os.path.basename(tokens[0])
            # Split the data into the quaternion and translation parts
            q, t = np.split(np.array(tokens[1:], dtype=float), [4])
            
            # Convert the quaternion to a rotation matrix
            R = np.asmatrix(qvec2rotmat(q)).transpose()
            
            # Initialize a 4x4 identity matrix
            T = np.identity(4)
            
            # Set the rotation and translation components
            T[0:3, 0:3] = R
            T[0:3, 3] = -R.dot(t)
            
            # Store the transformation matrix in the dictionary with the name as the key
            poses[name] = T  # c2w (camera-to-world transformation)
    
    # Assert that the poses dictionary is not empty
    assert len(poses) > 0
    
    return poses

def parse_image_list(path, with_intrinsics=False, simple_name = True):
    images = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            if simple_name:
                name = name.split('/')[-1]
            if with_intrinsics:
                model, width, height, *params = data
                params = np.array(params, float)
                cam = pycolmap.Camera(model, int(width), int(height), params)
                images.append((name, cam))
            else:
                images.append(name)
    
    assert len(images) > 0
    logger.info(f'Imported {len(images)} images from {path.name}')
    return images
def parse_db_intrinsic_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            
            data_line=line.split(' ')
            name = data_line[0].split('/')[-1]
            _,_,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]  
            
            K_w2c = np.array([ 
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            images[name] = K_w2c
            break
  
    return K_w2c

def parse_image_lists(paths, with_intrinsics=False, simple_name = True):
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    
    assert len(files) > 0
    for lfile in files:
        images += parse_image_list(lfile, with_intrinsics=with_intrinsics, simple_name = simple_name)
    return images


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)


def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))


def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator='_')
