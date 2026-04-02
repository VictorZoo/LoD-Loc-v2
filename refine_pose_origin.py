# type: ignore
import logging
import os
import sys
import shutil
import torch
from os.path import join
from tqdm import tqdm
import numpy as np
from torch.utils.data.dataset import Subset
import torchvision.transforms as T

import commons
from parse_args import parse_args
from path_configs import get_path_conf
from gloc import extraction
from gloc import initialization
from gloc import rendering
from gloc.models import get_ref_model
from gloc.rendering import get_renderer
from gloc.datasets import get_dataset, find_candidates_paths, get_transform, RenderedImagesDataset, ImListDataset
from gloc.utils import utils, visualization
from gloc.resamplers import get_protocol
from configs import get_config
import subprocess
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

from RealTime_render import RealTime_render

blender_path = '/home/ubuntu/blender-3.3.1-linux-x64/blender' #(Your Blender Path)
rgb_blend_path = 'UAVD4L-LoD-1-3-rgb_revise.blend'
python_rgb_path = 'db_rgb_focal_mcloc.py'

model_overrides = {
    "model": {
        "name": 'Contrast',
        "lamb_val": [0.8,0.8,0.8],
        "num_sample_val": [[8, 10, 10, 30],[8, 10, 10, 30],[8, 10, 10, 30]]
    },
}

data_cfg_demo = OmegaConf.load("Contrast.yaml")

default_cfg_single = OmegaConf.create({"data": data_cfg_demo})


def run_command(cmd):
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        process.wait()


def main(args):
    commons.make_deterministic(args.seed)
    commons.setup_logging(args.save_dir, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    
    paths_conf = get_path_conf(args.colmap_res, args.mesh)
    temp_dir = join(paths_conf['temp'], args.exp_name)
    if not os.path.exists(temp_dir):           
        os.makedirs(temp_dir) #revise

    exp_config = get_config(args.name)
    scores = None
    
    render2loc = RealTime_render(args.render_config)
    # render2loc = None
    
    for i in range(len(exp_config)):
        ref_args = exp_config[i]
        args.__dict__.update(ref_args)
        scores_temp, render_dir = refinement_loop(args, render2loc)
        
        scores = utils.update_scores(scores, scores_temp)
        args.pose_prior = join(render_dir, 'est_poses.txt')
    
    visualization.plot_scores(scores, args.save_dir)

    ### cleaning up...
    logging.info(f'Moving rendering from temp dir {temp_dir} to {args.save_dir}')
    shutil.move(join(temp_dir, 'renderings'), args.save_dir, copy_function=shutil.move)
    shutil.rmtree(temp_dir)
    logging.info('Terminating without errors!')
    
    
def refinement_loop(args, render2loc):
    DS = args.name
    res = args.res

    paths_conf = get_path_conf(args.colmap_res, args.mesh)
    transform = get_transform(args, paths_conf[DS]['colmap'])
    pose_dataset = get_dataset(DS, paths_conf[DS], transform)
    temp_dir = join(paths_conf['temp'], args.exp_name)            
    
    first_step, all_pred_t, all_pred_R, scores = initialization.init_refinement(args, pose_dataset)
    ######### START REFINEMENT LOOP
    N_steps = args.steps
    N_per_beam = args.N // args.beams
    n_beams = args.beams
    N_views = args.N
    
    cfg = {}
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    default = default_cfg_single

    default = OmegaConf.merge(default, dict(model_overrides))
    cfg = OmegaConf.merge(default, cfg)

    args.cfg = cfg
    fine_model = get_ref_model(args)

    logging.info('Recomputing query features with refinement model...')
    queries_subset = Subset(pose_dataset, pose_dataset.q_frames_idxs)
    
    # q_descriptors = extraction.get_query_features(fine_model, queries_subset, args.feat_level)
    q_descriptors = extraction.get_query_mask(queries_subset)

    resampler = get_protocol(args, N_per_beam, args.protocol)
    # 改
    renderer = get_renderer(args, paths_conf)
    
    max_step = utils.get_n_steps(pose_dataset.num_queries(), N_views, N_steps, args.renderer, args.hard_stop)
    for step in range(first_step, N_steps):
        if (step - first_step) == max_step:
            logging.info('Stopping due to Open3D bug')
            break
        
        resampler.init_step(step)
        center_std, angle_delta = resampler.scaler.get_noise()
        logging.info(f'[||] Starting iteration n.{step+1}/{N_steps} [||]')
        logging.info(f'Perturbing poses with Theta {angle_delta} and center STD {center_std}. Resolution {res}')

        if (first_step == step) and (args.resume_step is not None):
            render_dir = args.resume_step
            logging.info(f'Resuming from step dir {render_dir}...')
        else:
            perturb_str = resampler.get_pertubr_str(step, res)
            render_dir = perturb_step(render2loc, perturb_str, pose_dataset, renderer, resampler, all_pred_t, all_pred_R, temp_dir, n_beams)
            # renderer.end_epoch(render_dir)
        
        (all_pred_t, all_pred_R, 
            all_errors_t, all_errors_R) = rank_candidates(fine_model, pose_dataset, render_dir, pose_dataset.transform, 
                                                          q_descriptors, N_per_beam, n_beams, chunk_limit=args.chunk_size)
        result_str, results = utils.eval_poses_top_n(all_errors_t, all_errors_R, descr=f'step {step}')
        logging.info(result_str)

        scores['steps'].append(results)
        torch.save(scores, join(args.save_dir, 'scores.pth'))

        # if args.clean_logs:
        #     from clean_logs import main as cl_logs
        #     logging.info('Removing rendering files...')
        #     cl_logs(temp_dir, only_step=step)

    return scores, render_dir


def perturb_step(render2loc, perturb_str, pose_dataset, renderer, resampler, pred_t, pred_R, basepath, n_beams=1):    
    out_dir = os.path.join(basepath, 'renderings', perturb_str)
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)
    logging.info(f'Generating renders in {out_dir}')
    
    # rend_model = renderer.load_model()
    r_names_per_beam = {}        
    for q_idx in tqdm(range(len(pose_dataset.q_frames_idxs)), ncols=100):
        
        idx = pose_dataset.q_frames_idxs[q_idx]
        q_name = pose_dataset.get_basename(idx)
        q_key_name = os.path.splitext(pose_dataset.images[idx].name)[0]
        r_names_per_beam[q_idx] = {}

        K, w, h = pose_dataset.get_intrinsics(q_key_name)

        r_dir = os.path.join(out_dir, q_name)
        if not os.path.exists(r_dir):
            os.makedirs(r_dir)
        # import time
        # time_s = time.time()



        for beam_i in range(n_beams):
            beam_dir = join(r_dir, f'beam_{beam_i}')
            if not os.path.exists(beam_dir):
                os.makedirs(beam_dir)
            pred_t_beam = pred_t[q_idx, beam_i]
            pred_R_beam = pred_R[q_idx, beam_i]
            
            r_names, render_ts, render_qvecs, calibr_pose, poses = resampler.resample(K, q_name, pred_t_beam, pred_R_beam, q_idx=q_idx, beam_i=beam_i)
            # breakpoint()
            # x_, euler_ = get_t_euler(np.array(poses))
            r_names_per_beam[q_idx][beam_i] = r_names
            # poses have to be logged in 'beam_dir', but rendered in 'r_dir', so that
            # they can be rendered all together in 'deferred' mode, thus being more efficient
            
            rendering.log_poses(beam_dir, r_names, render_ts, render_qvecs, K, w, h, args.renderer)

            if renderer.supports_deferred_rendering:
                to_render_dir = r_dir
            else:
                to_render_dir = beam_dir
            
            # translation_, euler_angles_ = render2loc.get_pose_w2cToWGS84_batch(poses)
            # pose_dict_seed = {f"{r_name}.png": [euler_angles_[i, 0], euler_angles_[i, 1], euler_angles_[i, 2], 
            #                    translation_[i, 0], translation_[i ,1], translation_[i, 2]] 
            #                     for i, r_name in enumerate(r_names)}
            
            # for counter, name in enumerate(pose_dict_seed.keys()):
            #     for i in range(5):
            #         render2loc.translation = [pose_dict_seed[name][3], pose_dict_seed[name][4], pose_dict_seed[name][5]] 
            #         render2loc.euler_angles = [pose_dict_seed[name][0], pose_dict_seed[name][1], pose_dict_seed[name][2]]  #不需要负号
            #         # import pdb;pdb.set_trace()         
            #         render2loc.renderer.update_pose(render2loc.translation, render2loc.euler_angles)
            #     # breakpoint()
            #     render2loc.renderer.save_color_image(str(to_render_dir+'/'+name))
            #     # color_image = render2loc.renderer.get_color_image()
            #     # mask = np.all(color_image == [255,255,255], axis=-1).astype(np.uint8)
            #     # assert counter == eval(name.split('.')[0])
                
            #     # render_feats[counter] = mask

            
            # **** Blender  ****
            cmd = '{} -b {} -P {} -- {} {} {}'.format(blender_path, 
                                                    rgb_blend_path, 
                                                    python_rgb_path, 
                                                    to_render_dir+'/rendered_intrinsics.txt',
                                                    to_render_dir+'/rendered_views.txt', 
                                                    to_render_dir+'_blender',
                                        )
            # breakpoint()
            # os.system(cmd)
            run_command(cmd)

        # breakpoint()
    logging.info('Moving each renders into their beams folder')
    if renderer.supports_deferred_rendering:
        for q_idx in range(len(pose_dataset.q_frames_idxs)):
            idx = pose_dataset.q_frames_idxs[q_idx]
            q_name = pose_dataset.get_basename(idx)
            
            r_dir = join(out_dir, q_name)
            rendering.split_to_beam_folder(r_dir, n_beams, r_names_per_beam[q_idx])

    return out_dir


def rank_candidates(fine_model, pose_dataset, render_dir, transform, q_descriptors, N_per_beam, n_beams, chunk_limit=1100):
    all_pred_t   = np.empty((len(pose_dataset.q_frames_idxs), n_beams, N_per_beam, 3))
    all_pred_R   = np.empty((len(pose_dataset.q_frames_idxs), n_beams, N_per_beam, 3, 3))
    all_errors_t = np.empty((len(pose_dataset.q_frames_idxs), n_beams, N_per_beam))
    all_errors_R = np.empty((len(pose_dataset.q_frames_idxs), n_beams, N_per_beam))
    all_scores   = np.empty((len(pose_dataset.q_frames_idxs), n_beams, N_per_beam))
    
    logging.info(f'Extracting candidates paths')
    candidates_pathlist, query_res = find_candidates_paths(pose_dataset, n_beams, render_dir)

    logging.info(f'Found {len(candidates_pathlist)} images for {pose_dataset.n_q} queries, now extracting features altogether')
    same_res_transform = T.Compose(transform.transforms.copy())
    same_res_transform.transforms[1] = T.Resize((query_res), antialias=True)
    imlist_ds = ImListDataset(candidates_pathlist, same_res_transform)

    chunk_start_q_idx, chunk_end_q_idx, chunks = extraction.split_renders_into_chunks(
        pose_dataset.n_q, len(imlist_ds), n_beams, N_per_beam, chunk_limit
    )
   
    # dim = extraction.get_feat_dim(fine_model, query_res, args.feat_level)
    dim = (480, 720)
    # visualize_descriptors(q_descriptors[1])
    # breakpoint()
    # logging.info(f'Query splits: {chunk_start_q_idx}, {chunk_end_q_idx}')
    # logging.info(f'Chunk splits: {[c for c in chunks]}')
    for ic, chunk in enumerate(chunks):
        q_idx_start = chunk_start_q_idx[ic]
        q_idx_end = chunk_end_q_idx[ic]

        logging.info(f'Chunk n.{ic}')
        logging.info(f'Query from {q_idx_start} to {q_idx_end}')
        logging.info(f'Images from {chunk[0]} to {chunk[-1]}')
        # breakpoint()
        chunk_ds = Subset(imlist_ds, chunk)    
        # breakpoint()
        # descriptors = extraction.get_candidates_features(fine_model, chunk_ds, dim, args.feat_level, 36)
        descriptors = extraction.get_candidates_mask(chunk_ds, dim, 36)
        
        # visualize_descriptors(descriptors)
        logging.info(f'Extracted shape {descriptors.shape}, now computing predictions')
        for q_idx in tqdm(range(q_idx_start, q_idx_end), ncols=100):
            q_name = pose_dataset.get_basename(pose_dataset.q_frames_idxs[q_idx])        
            query_dir = os.path.join(render_dir, q_name)
            q_feats = q_descriptors[q_idx]

            for beam_i in range(n_beams):
                beam_dir = join(query_dir, f'beam_{beam_i}')
                rd = RenderedImagesDataset(beam_dir, verbose=False)        
                start_idx = (q_idx-q_idx_start)*n_beams*N_per_beam + beam_i*N_per_beam
                end_idx = start_idx + N_per_beam
                r_db_descriptors = descriptors[start_idx:end_idx]
                # predictions, scores = fine_model.rank_candidates(q_feats, r_db_descriptors, get_scores=True) 
                predictions, scores = fine_model.rank_candidates_mask(q_feats, r_db_descriptors, get_scores=True)
                # predictions, scores = fine_model.rank_candidates_multimask(q_feats, r_db_descriptors, get_scores=True)
                true_t, true_R, pred_t, pred_R = utils.get_pose_from_preds_w_truth(q_idx, pose_dataset, rd, predictions, N_per_beam)
                
                errors_t, errors_R = utils.get_errors_from_preds(true_t, true_R, pred_t, pred_R, N_per_beam)
                
                all_pred_t[q_idx, beam_i] = pred_t
                all_pred_R[q_idx, beam_i] = pred_R
                all_errors_t[q_idx, beam_i] = errors_t
                all_errors_R[q_idx, beam_i] = errors_R
                all_scores[q_idx, beam_i] = scores[:N_per_beam]

                # save scores within each beam so renders can be deleted afterwards
                torch.save((predictions, scores), join(beam_dir, 'scores.pth'))

        del q_feats, r_db_descriptors, descriptors

    # sort predictions/errors according the score across beams
    # only needed to log and eval poses, the optimization is beam-independent
    flat_pred_R, flat_pred_t, flat_preds, all_errors_t, all_errors_R = utils.sort_preds_across_beams(all_scores, all_pred_t, all_pred_R, all_errors_t, all_errors_R)
    # log pose estimate
    if flat_preds.shape[-1] > 6:
        # if there are at least 6 preds per query
        logging.info(f'Generating pose file...')
        utils.log_pose_estimate(render_dir, pose_dataset, flat_pred_R, flat_pred_t, flat_preds=flat_preds)

    return all_pred_t, all_pred_R, all_errors_t, all_errors_R


if __name__ == '__main__':
    args = parse_args()
    main(args)
