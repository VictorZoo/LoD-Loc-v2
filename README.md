<p align="center">
  <h1 align="center"><ins>LoD-Loc v2</ins>:<br>Aerial Visual Localization over Low Level-of-Detail City Models using Explicit Silhouette Alignment</h1>
  <p align="center">
    <h>Juelin&nbsp;Zhu</h>
    ·
    <h>Shuaibang&nbsp;Peng</h>
    ·
    <h>Long&nbsp;Wang</h>
    ·
    <h>Hanlin&nbsp;Tan</h>
    ·
    <h>Yu&nbsp;Liu</h>
    ·
    <h>Maojun&nbsp; Zhang</h>
    ·
    <h>Shen&nbsp; Yan</h>
  </p>
  <h2 align="center">ICCV 2025</h2>

  <h3 align="center">
    <a href="https://pppppsb.github.io/LoD-Locv2/">Project Page</a>
    | <a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_LoD-Loc_v2_Aerial_Visual_Localization_over_Low_Level-of-Detail_City_Models_ICCV_2025_paper.pdf">Paper</a> 
    | <a href="https://pppppsb.github.io/LoD-Locv2/">Demo</a>
  </h3>
  <div align="center"></div>
</p>
<p align="center">
    <a href="utils/Intro_new1_01.png"><img src="utils/Intro_new1_01.png" alt="teaser" width="100%"></a>
    <br>
    <em>LoD-Loc v2 tackles visual localization w.r.t a scene represented as LoD1.0 3D maps. Given a query image and its pose prior, the method utilizes the wireframe of LoD models to recover the camera pose.</em>
</p>


This repository is an implementation of the paper "LoD-Loc v2: Aerial Visual Localization over Low Level-of-Detail City Models using Explicit Silhouette Alignment".

## Important Things

We highly appreciate the research community's interest in the LoD-Loc v2 project. Please note that the **OSG rendering technique** described in the paper involves project intellectual property constraints. Consequently, we have implemented a **Blender-based** rendering pipeline as a substitute in this codebase.

## Segmentation

We adopt the segmentation paradigm of [SAM2-Unet](https://github.com/WZH0120/SAM2-UNet) for our semantic segmentation task. For detailed pipeline specifications, please refer to the original SAM2-Unet paper.


## Test
```bash
CUDA_VISIBLE_DEVICES="0" \
python ./script/refine_pose_origin.py \
--render_config "./config/config_RealTime_render_1.json" \
--sampler "rand_yaw_or_pitch" \
--name "inTraj" \
--pose_prior "/home/ubuntu/code/mcloc_poseref/data/UAVD4L-LoD/inTraj/inPlace_gps_newAll.txt"
```

## Acknowledgement
LoD-Loc v2 takes the [MC-Loc](https://github.com/ga1i13o/mcloc_poseref) as its code backbone. Thanks to Gabriele Trivigno for the opening source of his excellent work and his PyTorch implementation.


## BibTex citation

Please consider citing our work if you use any code from this repo or ideas presented in the paper:
```
@InProceedings{Zhu_2025_ICCV,
    author    = {Zhu, Juelin and Peng, Shuaibang and Wang, Long and Tan, Hanlin and Liu, Yu and Zhang, Maojun and Yan, Shen},
    title     = {LoD-Loc v2: Aerial Visual Localization over Low Level-of-Detail City Models using Explicit Silhouette Alignment},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {26610-26621}
}
```
