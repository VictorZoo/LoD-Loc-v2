import os
import numpy as np 
import torch
import cv2
from .plotting import make_matching_figure
import matplotlib.cm as cm
from tqdm import tqdm
from pathlib import Path
import immatch
import yaml

class ImageMatcher:
    def __init__(self, config_file = ''):
        self.config_file = config_file
        
        self.matcher = self.init()
    def init(self):
        with open(self.config_file, 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['airloc']
            if 'ckpt' in args:
                args['ckpt'] = os.path.join('..', args['ckpt'])
            class_name = args['class']

        # Init model
        model = immatch.__dict__[class_name](args)
        matcher = lambda im1, im2: model.match_pairs(im1, im2)
        return matcher
    def show_match(self, img0_raw, img1_raw, mkpts0, mkpts1,mconf, save_path):
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()
        mconf = mconf.cpu().numpy()
        color = cm.jet(mconf)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, path = save_path)
        return fig
        
    def main(self, data, save_loc_path: Path):
        """
        Main function to process image pairs for matching and save the results.

        Args:
            data (dict): Dictionary containing image pairs and other data.
            matcher (callable): A function that takes two image paths and returns matches.
            save_loc_path (Path): Path to the directory where the results will be saved.
            iter (int): Iteration number used for naming output files.

        Returns:
            fig: The figure object resulting from the visualization (if generated).
        """
        pairs = data["pairs"]
        pbar = tqdm(total=len(pairs), unit='pts')
        match_res = {}
        
        for imgq_pth, render_list in pairs.items():
            imgq_pth = str(imgq_pth)
            max_correct = 0
            imgq_name = os.path.basename(imgq_pth)
            
            for render_pth in render_list:
                imgr_pth, exrr_pth = render_pth
                imgr_name = os.path.basename(imgr_pth)
                imgr_pth = str(imgr_pth)
                
                
                # Perform matching and filter matches using RANSAC
                matches, _, _, mconf = self.matcher(imgq_pth, imgr_pth)
                F, mask = cv2.findFundamentalMat(matches[:,:2], matches[:,2:], cv2.FM_RANSAC,3, 0.99)
                
                # Filter matches based on the mask
                index = np.where(mask == 1)
                new_matches = matches[index[0]]
                mconf = torch.tensor(mconf[index[0]]).float()
                mkpts0q = torch.tensor(new_matches[:, :2]).float()
                mkpts1r = torch.tensor(new_matches[:, 2:]).float()
                correct = torch.tensor(mkpts0q.shape[0])
                
                # Update final match results if more correct matches are found
                if correct > max_correct:
                    max_correct = correct
                    imgr_name_final, imgr_pth_final, exrr_pth_final = imgr_name, imgr_pth, exrr_pth
                    mkpts0q_final, mkpts1r_final, mconf_final = mkpts0q, mkpts1r, mconf
            
            # Visualize and save the matches if there are any correct matches
            if max_correct > 0:
                if not os.path.exists(save_loc_path / 'matches'):
                    os.makedirs(save_loc_path / 'matches/')
                match_vis_path = save_loc_path / 'matches' / (imgq_name.split('.')[0] + '.png')
                imgq_raw = cv2.imread(str(imgq_pth), cv2.IMREAD_GRAYSCALE)
                imgr_raw = cv2.imread(str(imgr_pth_final), cv2.IMREAD_GRAYSCALE)
                fig = self.show_match(imgq_raw, imgr_raw, mkpts0q_final, mkpts1r_final, mconf_final, match_vis_path)
            
            # Update progress bar
            pbar.update(1)
            
            # Store match results in the dictionary
            match_res[imgq_name] = {
                "imgr_name": imgr_name_final,
                "exrr_pth": exrr_pth_final,
                "mkpts_r": mkpts1r_final,
                "mkpts_q": mkpts0q_final,
                "correct": max_correct
            }
        
        # Update the input data with the new match results
        data["matches"] = match_res
        pbar.close()
        
        return fig




         

        