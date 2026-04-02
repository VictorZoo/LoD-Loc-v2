import json
from osm_render_class import DataGenerator      # ← 文件名按你保存的为准
import torch

# ———————————————————————————————————
# ① 常量字典（与脚本保持一致）
CONSTANTS = {
    "MAX_LAYOUT_HEIGHT": 640,
    "BLD_INS_LABEL_MIN": 10,
    "N_MAX_SAMPLES": 6,
    "PATCH_SIZE": 1536,
    "ZOOM_LEVEL": 18,
}

CLASSES = {
    "NULL": 0,
    "ROAD": 1,
    "BLD_FACADE": 2,
    "GREEN_LANDS": 3,
    "CONSTRUCTION": 4,
    "COAST_ZONES": 5,
    "OTHERS": 6,
    "BLD_ROOF": 7,
}

HEIGHTS = {
    "ROAD": 4,
    "GREEN_LANDS": 8,
    "CONSTRUCTION": 10,
    "COAST_ZONES": 0,
    "ROOF": 1,
}

# ———————————————————————————————————
# ② 路径配置（自行替换）
GE_PROJ_DIR = "/home/ubuntu/code/city-dreamer/data/ges/US-NewYork-1113rdAve-R313-A429"   # 内含 metadata.json  & frames
OSM_DIR     = "/home/ubuntu/code/city-dreamer/data/osm/US-NewYork"            # 内含 seg_map.npy / height_field.npy / metadata.json
OUT_DIR     = "/home/ubuntu/code/city-dreamer/data/test_output"              # 结果保存目录

# ———————————————————————————————————
# ③ 实例化数据生成器（focal 将自动从 metadata.json 的 vfov 计算）
dg = DataGenerator(
    ge_project_dir=GE_PROJ_DIR,
    osm_dir=OSM_DIR,
    output_dir=OUT_DIR,
    constants=CONSTANTS,
    classes = CLASSES,
    heights = HEIGHTS,
)

# ———————————————————————————————————
# ④ 渲染第 0 帧，可选 pitch_offset 让相机再俯/仰几个度
out_seg = dg.render_frame(
    frame_idx=0,
    pitch_offset=-10.0           # 改成正值 = 低头，负值 = 抬头
)

# ———————————————————————————————————
# ⑤ 把首层语义标签存成 PNG（快速查看效果）
seg_maps = dg.get_color_img(out_seg)
for idx, sg in enumerate(seg_maps):
    sg.save(f"{OUT_DIR}/frame0_sem.png")  #jpg
# dg.save_first_hit_png(out_seg, f"{OUT_DIR}/frame0_sem.png")

print("Done → outputs/frame0_sem.png")
