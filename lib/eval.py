import cv2
import logging
import numpy as np
import os
from lib.transform import qvec2rotmat
from collections import defaultdict
def evaluate(results, gt, only_localized=False):
    """
    Evaluate the accuracy of pose predictions by comparing them with ground truth poses.
    
    Args:
        results (str): Path to the file containing the predicted poses.
        gt (str): Path to the file containing the ground truth poses.
        only_localized (bool): Flag to skip unlocalized images. Defaults to False.
    
    Returns:
        str: A formatted string with the evaluation results.
    """
    predictions = {}
    test_names = []
    
    with open(results, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            tokens = data.split()
            name = tokens[0].split('/')[-1]
            q, t = np.split(np.array(tokens[1:], dtype=float), [4])

            # Convert quaternion to rotation matrix and store with translation
            predictions[name] = (qvec2rotmat(q), t)
            test_names.append(name)
            

    gts = {}
    with open(gt, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            tokens = data.split()
            name = tokens[0].split('/')[-1]
            q, t = np.split(np.array(tokens[1:], dtype=float), [4])

            gts[name] = (qvec2rotmat(q), t)
    
    errors_t = []
    errors_R = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.
        else:
            R_gt, t_gt = gts[name]
            R, t = predictions[name]
            
            # Calculate translation and rotation errors
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))

        errors_t.append(e_t)
        errors_R.append(e_R)
    
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    med_t = np.median(errors_t)
    std_t = np.std(errors_t)
    med_R = np.median(errors_R)
    std_R = np.std(errors_R)
    
    out = '\nMedian errors: {:.3f}m, {:.3f}deg\n'.format(med_t, med_R)
    out += 'Std errors: {:.3f}m, {:.3f}deg\n'.format(std_t, std_R)
    
    out += 'Percentage of test images localized within:'
    threshs_t = [1, 3, 5]
    threshs_R = [1.0, 3.0, 5.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += '\n\t{:.0f}cm, {:.0f}deg : {:.2f}%'.format(th_t * 100, th_R, ratio * 100)
    
    print(out)
    return out
    
def pose(gt_pose, render_pose):
    # gt path
  
    # # dji_day_gt.txt
    # dji_day_gt = config["dji_day_gt"]
    # # dji_night_gt.txt
    # dji_night_gt = config["dji_night_gt"]
    # # phone_day_gt.txt
    # phone_day_gt = config["phone_day_gt"]
    # phone_night_gt.txt
    # phone_night_gt = config["phone_night_gt"]


    # render1 pose.txt path
    # render1_pose = config["render1_pose"]
    # # render2 pose.txt path
    # render2_pose = config["render2_pose"]
    # # render3 pose.txt path
    # render3_pose = config["render3_pose"]

    
    # # ---------------------------------
    # print("=====superpoint superglue")
    # print ('Evaluation DJI Day render_1 localization')
    # evaluate(render1_pose, dji_day_gt)
    
    # print ('Evaluation DJI Day render_2 localization')
    # evaluate(render2_pose, dji_day_gt)
    
    # print ('Evaluation DJI Day render_3 localization')
    # evaluate(render3_pose, dji_day_gt)

    # # ---------------------------------
    # print("=====superpoint superglue")
    # print ('Evaluation DJI Night render_1 localization')
    # evaluate(render1_pose, dji_night_gt)
    
    # print ('Evaluation DJI Night render_2 localization')
    # evaluate(render2_pose, dji_night_gt)
    
    # print ('Evaluation DJI Nihgt render_3 localization')
    # evaluate(render3_pose, dji_night_gt)   
    # ---------------------------------
    # print("=====fixed xy")
    # print ('Evaluation Phone Day render_1 localization')
    # evaluate(render1_pose, phone_day_gt)
    
    # print ('Evaluation Phone Day render_2 localization')
    # evaluate(render2_pose, phone_day_gt)
    
    # print ('Evaluation Phone Day render_3 localization')
    # evaluate(render3_pose, phone_day_gt)
 
 
    # # ---------------------------------
    print("=====superpoint superglue")
    print ('Evaluation Phone Night render_1 localization')
    evaluate(render_pose, gt_pose)
    
    # print ('Evaluation Phone Night render_2 localization')
    # evaluate(render2_pose, phone_night_gt)
    
    # print ('Evaluation Phone Nihgt render_3 localization')
    # evaluate(render3_pose, phone_night_gt)     

def load_gt_w(path):
    d = {}
    with open(path, 'r') as file:
        for line in file.read().rstrip().split('\n'):
            name = line.split(' ')[0]
            if name[-5] is 'W':
                # key = line.split(' ')[0].split('_')[-2]
                value = list(map(float,line.rstrip().split(' ')[1:]))
                #x, y, z = transformer.transform(value[1], value[0], value[2])  # for RTK equipment
                d[name] = [value[0], value[1], value[2]]
    return d
    
    
def eval_absolute_XYZ(gt_list, pred_list, result):
    
    errors_t = [] 
    image_num = len(pred_list)
   
    for name in pred_list.keys():
        if name not in gt_list.keys():
            e_t = np.inf
            # e_R = 180.
            continue
        else:
            gt = gt_list[name]
            t_gt = np.array(gt[:2])
            pred = pred_list[name]
            t = np.array(pred[:2])
            e_t = np.linalg.norm(t_gt - t, axis=0)
        errors_t.append(e_t)

    errors_t = np.array(errors_t)
    med_t = np.median(errors_t)
    max_t = np.max(errors_t)
    min_t = np.min(errors_t)
    # import ipdb; ipdb.set_trace();
    out = f'\nTarget Localization results'
    out = f'\nTest image nums: {image_num}'
    out += f'\nMedian errors: {med_t:.3f}m'
    out += f'\nMax errors: {max_t:.3f}m'
    out += f'\nMin errors: {min_t:.3f}m'
    # print(out)
    out += '\nPercentage of test images localized within:'
    threshs_t = [1, 3, 5, 10]
    for th_t in threshs_t:
        ratio = np.mean((errors_t < th_t))
        out += f'\n\t{th_t:.0f}m : {ratio*100:.2f}%'
    
    with open(result,'w') as f:
            f.writelines(out)
    f.close()
    # logger.info(out)   

def position(gt_position, estimate_position, result):
    gt_position_list = load_gt_w(gt_position)
    estimate_position_list = load_gt_w(estimate_position)
    
    eval_absolute_XYZ(gt_position_list, estimate_position_list, result)
       
    
    


