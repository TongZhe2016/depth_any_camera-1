import os
import numpy as np
import re
import yaml
import sys
import glob
import json
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from dac.utils.colmap_loader import read_intrinsics_binary

import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

RESO = 1/4

"""
    Grid search method to prepare fisheye grid for Scannet++ dataset.
    Each scene has a different camera, so the lookup table is scene-specific.
"""

def cam2image(pcd):
    x = pcd[:, 0] / pcd[:, 2]
    y = pcd[:, 1] / pcd[:, 2]
    z = pcd[:, 2]
    
    # TODO: the OPEVCV scaling is applied on theta atan(r)?
    r = torch.sqrt(x**2 + y**2)
    theta = torch.arctan(r)
    theta_d = theta * (1 + k1*theta*theta + k2*theta**4 + k3*theta**6 + k4*theta**8)
    
    x = theta_d * x / (r+1e-9)
    y = theta_d * y / (r+1e-9)
    

    """
        Projection to image coordinates using intrinsic parameters
    """
    x = fx * x + cx
    y = fy * y + cy

    return x, y, z

def chunk(grid):
    x = grid[0, :]
    y = grid[1, :]

    x = (x - cx) / fx
    y = (y - cy) / fy
    dist = torch.sqrt(x*x + y*y)
    
    indx = torch.abs(map_dist[:, 1:] - dist[None, :]).argmin(dim=0)    
    x = x * map_dist[indx, 0] / map_dist[indx, 1]
    y = y * map_dist[indx, 0] / map_dist[indx, 1]
    
    # z has closed form solution sqrt(1 - z^2) / z = sqrt(x^2 + y^2)
    z = 1 / torch.sqrt(1 + x**2 + y**2)

    xy = torch.stack((x, y))
    xy *= z
    return xy


if __name__=="__main__":
    zipnerf_data_path = 'datasets/zipnerf/fisheye'
    # target_scene_names = ['alameda', 'berlin', 'london', 'nyc']  # if not empty, only process these scenes
    
    scene_dirs = sorted(glob.glob(zipnerf_data_path + '/*'))
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        # if len(target_scene_names) > 0 and scene_name not in target_scene_names:
        #     continue
        
        print(f"Processing {scene_name}")
        cameras_intrinsic_file = os.path.join(scene_dir, 'sparse/0/cameras.bin')
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        intr = cam_intrinsics[1]
        height = intr.height 
        width = intr.width
        
        fx = intr.params[0] * RESO
        fy = intr.params[1] * RESO
        cx = intr.params[2] * RESO
        cy = intr.params[3] * RESO
        k1 = intr.params[4]
        k2 = intr.params[5]
        k3 = intr.params[6]
        k4 = intr.params[7]
        print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}, k1: {k1}, k2: {k2}, k3: {k3}, k4: {k4}")
            
        # use half resolution to save memory and speed up
        H = int(height * RESO)
        W = int(width * RESO)
        # [H, W]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        # [H*W]
        u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32) + 0.5
        # [3, H*W]
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)
        grid = torch.from_numpy(pixels)    
        map_dist = []
        z_dist = []
        # ATTENTION, the range of ro2 = (x/z)^2 + (y/z)^2 can go beyond 1.0 for fisheye cameras, set it properly
        for ro in np.linspace(0.0, 25, 500000):
            theta = np.arctan(ro)
            theta_d = theta * (1 + k1*theta*theta + k2*theta**4 + k3*theta**6 + k4*theta**8)
            map_dist.append([ro, theta_d])
            
        map_dist = np.array(map_dist).astype(np.float32)
        # print(map_dist)
        map_dist = torch.from_numpy(map_dist).cuda()

        xys = []
        for i in tqdm(range(H)):
            xy = chunk(grid[:, i*W:(i+1)*W].cuda())
            xys.append(xy.permute(1, 0))
            # if i % 10 == 0:
            #     print(i)
        xys = torch.cat(xys, dim=0)

        z = torch.sqrt(1. - torch.norm(xys, dim=1, p=2) ** 2)
        isnan = z.isnan()
        z[isnan] = 1.
        pcd = torch.cat((xys, z[:, None], isnan[:, None]), dim=1)
        print("saving grid")
        np.save(os.path.join(scene_dir, 'sparse/0/grid_fisheye.npy'), pcd.detach().cpu().numpy().reshape(H, W, 4))

        """
            Treating each ray as a point on an unit sphere, apply forward distortion and project to compute the approximation error using the lookup table
        """
        
        # import ipdb; ipdb.set_trace()
        # show error map
        x, y, d = cam2image(pcd[:, :3])

        error = (x - grid[0].cuda()) ** 2 + (y - grid[1].cuda()) ** 2

        error_map = error.reshape(H, W).detach().cpu().numpy()
        error_map = np.clip(error_map, 0, 30)
        plt.imshow(error_map)
        print(f'max error: {error_map.max()}')
        # plt.show()
        plt.savefig(os.path.join(scene_dir, 'sparse/0/error_map.png'))

