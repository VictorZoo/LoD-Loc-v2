CUDA_VISIBLE_DEVICES="0" \
python refine_pose_origin.py \
--render_config "./config/config_RealTime_render_1.json" \
--sampler "rand_yaw_or_pitch" \
--name "inTraj" \
--pose_prior "/home/ubuntu/code/mcloc_poseref/data/UAVD4L-LoD/inTraj/inPlace_gps_newAll.txt"

CUDA_VISIBLE_DEVICES="0" \
python refine_pose_origin.py \
--render_config "./config/config_RealTime_render_swiss.json" \
--sampler "rand_yaw_or_pitch" \
--name "Swiss_out" \
--pose_prior "/home/ubuntu/code/mcloc_poseref/data/UAVD4L-LoD/Swiss_in/GPS_pose_new.txt"

CUDA_VISIBLE_DEVICES="0" \
python refine_pose_origin.py \
--render_config "./config/config_RealTime_render_swiss.json" \
--sampler "rand_yaw_or_pitch" \
--name "Swiss_in" \
--pose_prior "/home/ubuntu/code/mcloc_poseref/data/UAVD4L-LoD/Swiss_in/GPS_pose_new.txt"

CUDA_VISIBLE_DEVICES="0" \
python refine_pose_origin.py \
--render_config "./config/config_RealTime_render_swiss.json" \
--sampler "rand_yaw_or_pitch" \
--name "Swiss_out" \
--pose_prior "/home/ubuntu/code/mcloc_poseref/data/UAVD4L-LoD/Swiss_in/GPS_pose_new.txt"

