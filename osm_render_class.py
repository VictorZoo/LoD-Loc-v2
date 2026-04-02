import os
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

import extensions.voxlib as voxlib
from extensions.extrude_tensor import TensorExtruder
from utils.io_helper import _load_seg_map, _load_height_field, _load_metadata
from utils.osm_helper import lnglat2xy, get_img_patch
from utils.camera_helper import viewdir_to_yaw_pitch, yaw_pitch_to_viewdir
from utils.helpers import get_seg_map, get_color_img
import logging
import cv2
# ――――――――――――――――――――――――――――――――――――――――――――
# Helper I/O utilities ---------------------------------------------------------
# These three helper loaders are included here to avoid an extra utils.io_helper
# dependency. Modify them as you wish (e.g. to read PNG instead of NPY).
# ――――――――――――――――――――――――――――――――――――――――――――



# ――――――――――――――――――――――――――――――――――――――――――――
# Data‑Generator class ---------------------------------------------------------
# ――――――――――――――――――――――――――――――――――――――――――――

class DataGenerator:
    """Generate voxel‑raycasting views given an OSM patch + GE Studio project."""

    def __init__(
        self,
        ge_project_dir: str,
        osm_dir: str,
        output_dir: str,
        constants: Dict,
        classes: Dict,
        heights: Dict,
    ) -> None:
        """Construct the generator and pre‑compute static resources.

        Args
        -----
        ge_project_dir : directory with the GE Studio project (.json / .esp / metadata.json)
        osm_dir        : directory containing `seg_map.npy`, `height_field.npy`, `metadata.json`
        output_dir     : where to dump rendered outputs
        constants      : dict with keys  MAX_LAYOUT_HEIGHT, BLD_INS_LABEL_MIN, N_MAX_SAMPLES
        """
        self.ge_project_dir = ge_project_dir
        self.osm_dir = osm_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.constants = constants
        self.classes = classes
        self.heights = heights

        # ── Load GE camera data
        self.ge_camera_poses = self._load_ge_camera_poses()
        self.camera_focal = self._compute_focal(self.ge_camera_poses)

        # ── Load OSM rasters
        self.seg_map = _load_seg_map(self.osm_dir)
        self.height_field = _load_height_field(self.osm_dir)
        self.metadata = _load_metadata(self.osm_dir)

        # ── Build 3‑D semantic volume once
        self.ins_seg_map, self.building_stats = self.get_instance_seg_map(self.seg_map, contours=None)
        self.part_seg_map, self.part_hf = self._crop_generator(self.ge_camera_poses)
        self.tensor_extruder = TensorExtruder(self.constants["MAX_LAYOUT_HEIGHT"])
        self.seg_volume = self._get_seg_volume(self.part_seg_map, self.part_hf, self.tensor_extruder)

    # ─────────────────────── get_instance_seg_map ────────────────────────
    def get_instance_seg_map(self, seg_map, contours=None, use_contours=False):
        if use_contours:
            _, labels, stats, _ = cv2.connectedComponentsWithStats(
                (1 - contours).astype(np.uint8), connectivity=4
            )
        else:
            _, labels, stats, _ = cv2.connectedComponentsWithStats(
                (seg_map == self.classes["BLD_FACADE"]).astype(np.uint8), connectivity=4
            )

        # Remove non-building instance masks
        labels[seg_map != self.classes["BLD_FACADE"]] = 0
        # Building instance mask
        building_mask = labels != 0

        # Make building instance IDs are even numbers and start from 10
        # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
        labels = (labels + self.constants["BLD_INS_LABEL_MIN"]) * 2

        seg_map[seg_map == self.classes["BLD_FACADE"]] = 0
        seg_map = seg_map * (1 - building_mask) + labels * building_mask
        assert np.max(labels) < 2147483648
        return seg_map.astype(np.int32), stats[:, :4]

    # ─────────────────────── Crop helpers ────────────────────────────────
    def _crop_generator(self, ge_camera_poses):
        cx, cy = lnglat2xy(
            ge_camera_poses["center"]["coordinate"]["longitude"],
            ge_camera_poses["center"]["coordinate"]["latitude"],
            self.metadata["resolution"],
            self.constants['ZOOM_LEVEL'],
            dtype=float,
        )
        cx -= self.metadata["bounds"]["xmin"]
        cy -= self.metadata["bounds"]["ymin"]
        tr_cx, tr_cy = int(cx + 0.5), int(cy + 0.5)
        part_hf = get_img_patch(
                self.height_field,
                tr_cx,
                tr_cy,
                self.constants['PATCH_SIZE'],
            ).astype(np.int32)
        part_hf += ge_camera_poses["elevation"]
        part_seg_map = get_img_patch(
                self.ins_seg_map,
                tr_cx,
                tr_cy,
                self.constants['PATCH_SIZE'],
            ).astype(np.int32)
        # buildings = np.unique(part_seg_map[part_seg_map > self.constants["BLD_INS_LABEL_MIN"]])
        # part_building_stats = {}
        # for bid in buildings:
        #     _bid = bid // 2 - self.constants["BLD_INS_LABEL_MIN"]
        #     _stats = self.building_stats[_bid].copy().astype(np.float32)
        #     # NOTE: assert building_stats.shape[1] == 4, represents x, y, w, h of the components.
        #     # Convert x and y to dx and dy, where dx and dy denote the offsets to the center.
        #     _stats[0] = _stats[0] - tr_cx + _stats[2] / 2
        #     _stats[1] = _stats[1] - tr_cy + _stats[3] / 2
        #     part_building_stats[bid] = _stats
        return part_seg_map, part_hf

    # ───────────────────────────────────────────────────── internal helpers ──
    def _load_ge_camera_poses(self) -> Dict:
        ge_proj_name = self.ge_project_dir.split('/')[-1]
        camera_setting_file = os.path.join(self.ge_project_dir, "%s.json" % ge_proj_name)
        ge_project_file = os.path.join(self.ge_project_dir, "%s.esp" % ge_proj_name)

        if not os.path.exists(camera_setting_file) or not os.path.exists(ge_project_file):
            return None

        camera_settings = None
        with open(camera_setting_file) as f:
            camera_settings = json.load(f)

        camera_target = None
        with open(ge_project_file) as f:
            ge_proj_settings = json.load(f)
            scene = ge_proj_settings["scenes"][0]["attributes"]
            camera_group = next(
                _attr["attributes"] for _attr in scene if _attr["type"] == "cameraGroup"
            )
            camera_taget_effect = next(
                _attr["attributes"]
                for _attr in camera_group
                if _attr["type"] == "cameraTargetEffect"
            )
            camera_target = next(
                _attr["attributes"]
                for _attr in camera_taget_effect
                if _attr["type"] == "poi"
            )

        camera_poses = {
            "elevation": 0,
            "vfov": camera_settings["cameraFrames"][0]["fovVertical"],
            "width": camera_settings["width"],
            "height": camera_settings["height"],
            "center": {
                "coordinate": {
                    "longitude": next(
                        _attr["value"]["relative"]
                        for _attr in camera_target
                        if _attr["type"] == "longitudePOI"
                    )
                    * 360
                    - 180,
                    "latitude": next(
                        _attr["value"]["relative"]
                        for _attr in camera_target
                        if _attr["type"] == "latitudePOI"
                    )
                    * 180
                    - 90,
                    "altitude": next(
                        _attr["value"]["relative"]
                        for _attr in camera_target
                        if _attr["type"] == "altitudePOI"
                    )
                    + 1,
                }
            },
            "poses": [],
        }
        # NOTE: The conversion from latitudePOI to latitude is unclear.
        #       Fixed with the collected metadata.
        extra_metadata_file = os.path.join(self.ge_project_dir, "metadata.json")
        if not os.path.exists(extra_metadata_file):
            logging.error(
                "The project %s is missing the extra metadata file, "
                "which could result in a misalignment between the footage and segmentation maps."
                % ge_proj_name
            )
        else:
            with open(extra_metadata_file) as f:
                extra_metadata = json.load(f)

            camera_poses["elevation"] = extra_metadata["elevation"]
            camera_poses["center"]["coordinate"]["latitude"] = extra_metadata["clat"]

        # NOTE: All Google Earth renderings are centered around an altitude of 1.
        if camera_poses["center"]["coordinate"]["altitude"] != 1:
            logging.warning("The altitude of the camera center is not 1.")
            return None

        for cf in camera_settings["cameraFrames"]:
            camera_poses["poses"].append({"coordinate": cf["coordinate"]})
        return camera_poses

    # ────────────────── focal length computation ──
    @staticmethod
    def _compute_focal(cam_meta: Dict, default_vfov: float = 40.0) -> float:
        """Compute pixel focal length from GE metadata.

        Google Earth Studio 在 metadata.json 中保存了每个工程的垂直视场角
        `vfov`（单位: 度）。如果该字段缺失，就回落到 `default_vfov`。

            focal = H / (2 · tan(vfov / 2))
        """
        height = cam_meta["height"]
        vfov   = cam_meta.get("vfov", default_vfov)  # ← 读取真实 vfov
        return height / (2.0 * np.tan(np.deg2rad(vfov)))

    # ───────────────────────────────────────────────
    def _get_seg_volume(self, part_seg_map, part_hf, tensor_extruder=None):
        if tensor_extruder is None:
            tensor_extruder = TensorExtruder(self.constants["MAX_LAYOUT_HEIGHT"])

        seg_volume = tensor_extruder(
            torch.from_numpy(part_seg_map[None, None, ...]).cuda(),
            torch.from_numpy(part_hf[None, None, ...]).cuda(),
        ).squeeze()
        logging.debug("The shape of SegVolume: %s" % (seg_volume.size(),))
        # Change the top-level voxel of the "Building Facade" to "Building Roof"
        roof_seg_map = part_seg_map.copy()
        non_roof_msk = part_seg_map <= self.constants["BLD_INS_LABEL_MIN"]
        # Assume the ID of a facade instance is 2k, the corresponding roof instance is 2k - 1.
        roof_seg_map = roof_seg_map - 1
        roof_seg_map[non_roof_msk] = 0
        # for rh in range(1, HEIGHTS["ROOF"] + 1):
        #     seg_volume = seg_volume.scatter_(
        #         dim=2,
        #         index=torch.from_numpy(part_hf[..., None] + rh).long().cuda(),
        #         src=torch.from_numpy(roof_seg_map[..., None]).cuda(),
        #     )

        return seg_volume

    # ───────────────────────────────────────────────────── core public API ──
    def render_frame(self, frame_idx: int, pitch_offset: float = 0.0):
        """Render one GE frame (index in camera_poses["poses"])."""
        gcp = self.ge_camera_poses["poses"][frame_idx]
        center = self.ge_camera_poses["center"]["coordinate"]

        # Convert lng/lat ➜ xy (global OSM coord)
        cx, cy = lnglat2xy(center["longitude"], center["latitude"],
                            self.metadata["resolution"], self.constants["ZOOM_LEVEL"], dtype=float)

        x, y = lnglat2xy(gcp["coordinate"]["longitude"], gcp["coordinate"]["latitude"],
                                    self.metadata["resolution"], self.constants["ZOOM_LEVEL"], dtype=float)
        gcp_cx, gcp_cy = x - self.metadata["bounds"]["xmin"], y - self.metadata["bounds"]["ymin"]

        # Patch / volume centers
        cx -= self.metadata["bounds"]["xmin"]
        cy -= self.metadata["bounds"]["ymin"]
        tr_cx, tr_cy = int(cx + 0.5), int(cy + 0.5)
        vol_cx, vol_cy = ((self.constants['PATCH_SIZE'] - 1) // 2, (self.constants['PATCH_SIZE'] - 1) // 2)

        # cam_origin in voxel space
        cam_origin = torch.tensor([
            gcp_cy - tr_cy + vol_cy,
            gcp_cx - tr_cx + vol_cx,
            gcp["coordinate"]["altitude"],
        ], dtype=torch.float32, device=self.seg_volume.device)

        # viewdir
        viewdir = torch.tensor([
            cy - gcp_cy,
            cx - gcp_cx,
            -gcp["coordinate"]["altitude"],
        ], dtype=torch.float32, device=self.seg_volume.device)

        if pitch_offset != 0.0:
            yaw_deg, pitch_deg = viewdir_to_yaw_pitch(viewdir.cpu().numpy())
            viewdir_np = yaw_pitch_to_viewdir(yaw_deg, pitch_deg + pitch_offset).astype(np.float32)
            viewdir = torch.from_numpy(viewdir_np).cuda()

        voxel_id, depth2, raydirs = voxlib.ray_voxel_intersection_perspective(
            self.seg_volume,
            cam_origin,
            viewdir,
            torch.tensor([0, 0, 1], dtype=torch.float32, device=self.seg_volume.device),
            self.camera_focal * 2.06,
            [
                (self.ge_camera_poses["height"] - 1) / 2.0,
                (self.ge_camera_poses["width"] - 1) / 2.0,
            ],
            [self.ge_camera_poses["height"], self.ge_camera_poses["width"]],
            self.constants["N_MAX_SAMPLES"],
        )
        out_seg = voxel_id.squeeze()[..., 0].cpu().numpy()
        out_seg[out_seg >= self.constants["BLD_INS_LABEL_MIN"]] = self.classes["BLD_FACADE"]

        return out_seg

    def render_all_frames(self, pitch_offset: float = 0.0):
        """Render every frame in ge_camera_poses."""
        results = []
        for idx in tqdm(range(len(self.ge_camera_poses["poses"]))):
            results.append(self.render_frame(idx, pitch_offset))
        return results

    # ---------------------------------------------------------------------
    # Convenience save helpers (optional):
    def save_first_hit_png(self, voxel_id: torch.Tensor, path: str) -> None:
        """Save first‑layer semantic map as PNG for quick inspection."""
        from PIL import Image
        first_hit = voxel_id[..., 0].cpu().numpy().astype(np.uint8)
        Image.fromarray(first_hit).save(path)
    
    def get_color_img(self, out_seg):
        seg_maps = []
        seg_map = get_seg_map(
                out_seg
            )
        seg_maps.append(
            get_color_img(
                seg_map
            )
        )
        return seg_maps
