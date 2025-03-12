#!/usr/bin/env python
"""
Depth-Any-Camera demo script for inference different types of camera data on a single perspective trained model.
Model: DAC-Indoor
Test data source: Scannet++, Matterport3D, NYU
"""

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import cv2
import torch
import torch.cuda as tcuda
from PIL import Image
import glob
import torchvision.transforms.functional as TF

from dac.models.idisc_erp import IDiscERP
from dac.models.idisc import IDisc
from dac.models.idisc_equi import IDiscEqui
from dac.models.cnn_depth import CNNDepth
from dac.utils.visualization import save_val_imgs_v3
from dac.utils.erp_geometry import erp_patch_to_cam_fast, cam_to_erp_patch_fast
from dac.utils.colmap_loader import read_intrinsics_binary
from dac.dataloders.dataset import resize_for_input


def run_one_sample(model, model_name, device, sample, cano_sz, save_img_dir, grid_fisheye, args: argparse.Namespace):
    #######################################################################
    ############# data prepare (A simple version dataloader) ##############
    #######################################################################
    
    image = np.asarray(
        Image.open(sample["image_filename"])
    )
    image_name = os.path.basename(sample["image_filename"])
    
    fwd_sz=sample["fwd_sz"]
    out_sz=sample["out_sz"]
    
    phi = np.array(0).astype(np.float32)
    roll = np.array(0).astype(np.float32)
    theta = 0

    image = image.astype(np.float32) / 255.0
    depth_gt = np.ones_like(image)
    mask_valid_depth = depth_gt > 0.01
            
    # Automatically calculate the erp crop size
    crop_width = int(cano_sz[0] * sample["crop_wFoV"] / 180)
    crop_height = int(crop_width * fwd_sz[0] / fwd_sz[1])
    
    # convert to ERP
    image, depth_gt, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
        image, depth_gt, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
        crop_height, crop_width, cano_sz[0], cano_sz[0]*2, sample["cam_params"], roll, scale_fac=None
    )
    lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
    long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])
            
    # resizing process to fwd_sz.
    image, depth_gt, pad, pred_scale_factor, attn_mask = resize_for_input((image * 255.).astype(np.uint8), depth_gt, fwd_sz, None, [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=erp_mask)

    # convert to tensor batch
    normalization_stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    image = TF.normalize(TF.to_tensor(image), **normalization_stats)
    gt = TF.to_tensor(depth_gt)
    mask = TF.to_tensor((depth_gt > 0.01).astype(np.uint8))
    attn_mask = TF.to_tensor((attn_mask>0).astype(np.float32)) # the non-empty region after ERP conversion
    batch = {
        "image": image.unsqueeze(0),
        "gt": gt.unsqueeze(0),
        "mask": mask.unsqueeze(0),
        "attn_mask": attn_mask.unsqueeze(0),
        "lat_range": lat_range.unsqueeze(0),
        "long_range": long_range.unsqueeze(0),
        "info": {
            "pred_scale_factor": pred_scale_factor,
        },
    }
    
    #######################################################################
    ########################### model inference ###########################
    #######################################################################

    gt, mask, attn_mask, lat_range, long_range = batch["gt"].to(device), batch["mask"].to(device), batch["attn_mask"].to(device), batch["lat_range"].to(device), batch["long_range"].to(device)
    with torch.no_grad():
        if model_name == "IDiscERP":
            preds, _, _ = model(batch["image"].to(device), lat_range, long_range)
        else:
            preds, _, _ = model(batch["image"].to(device))
    preds *= pred_scale_factor
    
    #######################################################################
    ##################  Visualization and Output results  #################
    #######################################################################

    if "attn_mask" in batch.keys():
        attn_mask = batch["attn_mask"][0]
    else:
        attn_mask = None

    # default indoor visulization parameters
    vis_depth_max = 10.0

    ##########  Convert the ERP result back to camera space for visualization (No need for original ERP image)  ##########
    """
        Currently work perfect with phi = 0. For larger phi, corners may have artifacts.
    """
    # set output size the same aspact ratio as raw image (no need to be same as fw_size)
    cam_params={"dataset":"zipnerf"} # when grid table is available, no need for intrinsic parameters

    # scale the full erp_size depth scaling factor is equivalent to resizing data (given same aspect ratio)
    erp_h = cano_sz[0]
    erp_h = erp_h * batch["info"]["pred_scale_factor"]
    if "f_align_factor" in batch["info"]:
        erp_h = erp_h / batch["info"]["f_align_factor"][0].detach().cpu().numpy()
    img_out, depth_out, valid_mask, active_mask, depth_out_gt = erp_patch_to_cam_fast(
        batch["image"][0], preds[0].detach().cpu(), attn_mask, 0., 0., out_h=out_sz[0], out_w=out_sz[1], erp_h=erp_h, erp_w=erp_h*2, cam_params=cam_params, 
        fisheye_grid2ray=grid_fisheye, depth_erp_gt=batch["gt"][0].detach().cpu())
    
    # Save depth_out as uint16 image
    depth_out_uint16 = (depth_out.squeeze().numpy() * args.depth_scale).astype(np.uint16)
    depth_out_filename = os.path.join(save_img_dir, f"{image_name[:-4]}.png")
    cv2.imwrite(depth_out_filename, depth_out_uint16)
    print(f"Saved depth image to {depth_out_filename}")

    if args.vis:
        vis_out = save_val_imgs_v3(
            0,
            depth_out,
            img_out,
            f"{image_name[:-4]}_vis.jpg",
            save_img_dir,
            active_mask=active_mask,
            depth_max=vis_depth_max,
            )
        return vis_out
    else:
        return None
    
    
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    parser.add_argument("--config-file", type=str, default="checkpoints/dac_swinl_indoor.json")
    parser.add_argument("--model-file", type=str, default="checkpoints/dac_swinl_indoor.pt")
    parser.add_argument("--data-dir", type=str, default="datasets/zipnerf/fisheye/alameda")
    parser.add_argument("--resolution", "-r", type=int, default=4)
    parser.add_argument("--depth-scale", type=int, default=1000) # same as scannet++ original data process
    parser.add_argument("--vis", type=int, default=1)

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = eval(config["model_name"]).build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()
    cano_sz=config["cano_sz"] # the ERP size model was trained on

    # laod the camera intrinsics of the scene
    cameras_intrinsic_file = os.path.join(args.data_dir, "sparse/0/cameras.bin")
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    intr = cam_intrinsics[1]
    
    save_img_dir = os.path.join(args.data_dir, "depth_dac")
    os.makedirs(save_img_dir, exist_ok=True)
    
    # load the ray lookup table for fisheye cameras, for wrapping the depth back to align original image
    try:
        grid_fisheye = np.load(os.path.join(args.data_dir, "sparse/0/grid_fisheye.npy"))
    except:
        print("No fisheye grid found, please prepare the fisheye grid for the dataset first: splits/zipnerf/create_fisheye_grid_zipnerf.py")
    out_sz = (int(intr.height / args.resolution), int(intr.width / args.resolution))
    grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_sz[1], out_sz[0]), interpolation=cv2.INTER_NEAREST)
    grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_sz[1], out_sz[0]))
    grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
    
    # Prepare the sample for the model
    sample = {
            "dataset_name": "zipnerf",
            "crop_wFoV": 180, # degree decided by origianl data fov + some buffer
            "fwd_sz": [500, 750], # the patch size input to the model
            "out_sz": out_sz, # the final result saving size
            "cam_params": {
                "dataset":"zipnerf",
                "fl_x": intr.params[0],
                "fl_y": intr.params[1],
                "cx": intr.params[2],
                "cy": intr.params[3],
                "k1": intr.params[4],
                "k2": intr.params[5],
                "k3": intr.params[6],
                "k4": intr.params[7],
                "camera_model": "OPENCV_FISHEYE",
            }
        }
    
    # Process all the images included in the scene
    image_files = sorted(glob.glob(os.path.join(args.data_dir, "images", "*.JPG")))
    images_out = []
    for image_file in image_files:
        sample["image_filename"] = image_file
        vis_out = run_one_sample(model, config["model_name"], device, sample, cano_sz, save_img_dir, grid_fisheye, args)
        if vis_out is not None and len(images_out) < 300:
            images_out.append(Image.fromarray(vis_out))
    if args.vis:
        images_out[0].save(os.path.join(save_img_dir, "output_vis.gif"), save_all=True, append_images=images_out[1:], duration=100, loop=0)
    print("Demo finished")