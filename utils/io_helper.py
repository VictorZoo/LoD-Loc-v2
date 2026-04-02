# -*- coding: utf-8 -*-
#
# @File:   osm_helper.py
# @Author: Haozhe Xie
# @Date:   2023-03-21 16:16:06
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-28 18:35:22
# @Email:  root@haozhexie.com

import numpy as np
import os
import json
from typing import Dict
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def _load_seg_map(osm_dir: str) -> np.ndarray:
    """Load semantic/instance map saved as .npy (H×W, uint16/uint8)."""
    output_seg_map_file_path = os.path.join(osm_dir, "seg.png")
    seg_map = np.array(Image.open(output_seg_map_file_path).convert("P"))
    return seg_map


def _load_height_field(osm_dir: str) -> np.ndarray:
    """Load height field saved as .npy (H×W, uint16)."""
    output_hf_file_path = os.path.join(osm_dir, "hf.png")
    height_field = np.array(Image.open(output_hf_file_path))
    return height_field


def _load_metadata(osm_dir: str) -> Dict:
    with open(os.path.join(osm_dir, "metadata.json")) as f:
        return json.load(f)

