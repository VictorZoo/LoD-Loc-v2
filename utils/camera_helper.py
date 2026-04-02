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

def viewdir_to_yaw_pitch(viewdir):
    """
    Args:
        viewdir: list or np.ndarray of shape (3,) or (N, 3)
    Returns:
        yaw: float or np.ndarray
        pitch: float or np.ndarray
    """
    viewdir = np.atleast_2d(viewdir)
    normed = viewdir / np.linalg.norm(viewdir, axis=1, keepdims=True)

    vx, vy, vz = normed[:, 0], normed[:, 1], normed[:, 2]
    yaw = np.degrees(np.arctan2(vy, vx))
    pitch = np.degrees(np.arcsin(-vz))

    if viewdir.shape[0] == 1:
        return yaw[0], pitch[0]
    else:
        return yaw, pitch



def yaw_pitch_to_viewdir(yaw, pitch):
    """
    Args:
        yaw: float or np.ndarray in degrees
        pitch: float or np.ndarray in degrees
    Returns:
        viewdir: np.ndarray of shape (3,) or (N, 3)
    """
    yaw = np.atleast_1d(yaw)
    pitch = np.atleast_1d(pitch)

    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    vx = np.cos(pitch_rad) * np.cos(yaw_rad)
    vy = np.cos(pitch_rad) * np.sin(yaw_rad)
    vz = -np.sin(pitch_rad)

    viewdirs = np.stack([vx, vy, vz], axis=1)
    
    if viewdirs.shape[0] == 1:
        return viewdirs[0]
    else:
        return viewdirs

def recover_viewdir(unit_viewdir, altitude):
    unit_viewdir = np.array(unit_viewdir)
    vz = unit_viewdir[2]
    norm = altitude / -vz
    return unit_viewdir * norm
