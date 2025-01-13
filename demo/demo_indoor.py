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

from dac.models.idisc_erp import IDiscERP
from dac.models.idisc import IDisc
from dac.models.idisc_equi import IDiscEqui
from dac.models.cnn_depth import CNNDepth
from dac.utils.visualization import save_file_ply, save_val_imgs_v2
from dac.utils.unproj_pcd import reconstruct_pcd, reconstruct_pcd_erp
from dac.utils.erp_geometry import erp_patch_to_cam_fast
from dac.dataloders.dataset import resize_for_input
from dac.utils.erp_geometry import cam_to_erp_patch_fast
import torchvision.transforms.functional as TF

##################################################################################################################
############## samples for the demo of scannet++(fisheye), matterport3d(ERP), and nyu(perspective) ###############
##################################################################################################################

SAMPLE_1 = {
    "dataset_name": "scannetpp",
    "image_filename": "demo/input/scannetpp_rgb.jpg",
    "annotation_filename_depth": "demo/input/scannetpp_depth.png",
    "depth_scale": 1000.0,
    "fishey_grid": "demo/input/scannetpp_grid_fisheye.npy",
    "crop_height": 938, # this are decided by the dataset FOV and Trained ERP Size (Check each dataloader)
    "crop_width": 1400, # this are decided by the dataset FOV and Trained ERP Size (Check each dataloader)
    "fwd_sz": (500, 750), # the patch size input to the model
    "min_depth": 0.01,
    "max_depth": 40.0,
    "erp": False,
    "cam_params": {
        'dataset':'scannetpp',
        "fl_x": 789.9080967683176,
        "fl_y": 791.5566599926353,
        "cx": 879.203786509326,
        "cy": 584.7893145555763,
        "k1": -0.029473047856246333,
        "k2": -0.005769803970428537,
        "k3": -0.002148236771485755,
        "k4": 0.00014840568362061509,
        "w": 1752,
        "h": 1168,
        "camera_model": "OPENCV_FISHEYE",
    },
    "camera_intrinsics": torch.tensor(
        [
            [789.9080967683176, 0.0, 879.203786509326],
            [0.0, 791.5566599926353, 584.7893145555763],
            [0.0, 0.0, 1.0],
        ]
    )
}

SAMPLE_2 = {
    "dataset_name": "matterport3d",
    "image_filename": "demo/input/matterport3d_rgb.png",
    "annotation_filename_depth": "demo/input/matterport3d_depth.exr",
    "depth_scale": 1.0,
    "fishey_grid": None,
    "fwd_sz": (512, 1024), # the patch size input to the model
    "min_depth": 0.01,
    "max_depth": 80.0,
    "erp": True,
    "cam_params": {
        'dataset':'matterport3d',
        "w": 1024,
        "h": 512,
    },
    "camera_intrinsics": torch.tensor(
        [
            [1 / np.tan(np.pi/512), 0.000000e00, 256.],
            [0.000000e00, 1 / np.tan(np.pi/512), 512.],
            [0.000000e00, 0.000000e00, 1.000000e00],
        ]
    ) # a virtual camera intrinsics (not used in DAC, but used in evaluate perspective model on ERP)
}

SAMPLE_3 = {
    "dataset_name": "nyu",
    "image_filename": "demo/input/nyu_rgb.jpg",
    "annotation_filename_depth": "demo/input/nyu_depth.png",
    "depth_scale": 1000,
    "fishey_grid": None,
    "crop_height": 480, # this are decided by the dataset FOV and Trained ERP Size (Check each dataloader)
    "crop_width": 640, # this are decided by the dataset FOV and Trained ERP Size (Check each dataloader)
    "fwd_sz": (480, 640), # the patch size input to the model
    "min_depth": 0.01,
    "max_depth": 80.0,
    "erp": False,
    "cam_params": {
        'dataset': 'nyu',
        'fx': 5.1885790117450188e+02,
        'fy': 5.1946961112127485e+02,
        'cx': 3.2558244941119034e+02,
        'cy': 2.5373616633400465e+02,
        # 'k1': 2.0796615318809061e-01,
        # 'k2': -5.8613825163911781e-01,
        # 'p1': 7.2231363135888329e-04,
        # 'p2': 1.0479627195765181e-03,
        # 'k3': 4.9856986684705107e-01,
        'wFOV': 1.105,
        'hFOV': 0.8663,
        "w": 640,
        "h": 480,
    },
    "camera_intrinsics": torch.tensor(
        [
            [5.1885790117450188e02, 0, 3.2558244941119034e02],
            [0, 5.1946961112127485e02, 2.5373616633400465e02],
            [0, 0, 1],
        ]
    )
}

def demo_one_sample(model, device, sample, cano_sz, args: argparse.Namespace):
    #######################################################################
    ############# data prepare (A simple version dataloader) ##############
    #######################################################################
    
    image = np.asarray(
        Image.open(sample["image_filename"])
    )
    depth = (
        np.asarray(
            cv2.imread(sample["annotation_filename_depth"], cv2.IMREAD_ANYDEPTH)
        ).astype(np.float32)
        / sample["depth_scale"]
    )
    
    dataset_name = sample["dataset_name"]
    fwd_sz=sample["fwd_sz"]
    min_depth = sample["min_depth"]
    max_depth = sample["max_depth"]
    
    if not sample["erp"]:
        # convert depth from zbuffer to euclid 
        if dataset_name in ['nyu', 'kitti']:
            x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
            depth = depth * np.sqrt((x - sample["camera_intrinsics"][0, 2].item())**2 + (y - sample["camera_intrinsics"][1, 2].item())**2 + sample["camera_intrinsics"][0, 0].item()**2) / sample["camera_intrinsics"][0, 0].item()
            depth = depth.astype(np.float32)
        elif dataset_name == 'scannetpp': # Very critical for scannet++ fisheye. Skip kitti360 because we prepared the depth already in euclid.
            # For fisheye, converting back to euclid with undistorted ray direction via the ray lookup table for efficiency
            fisheye_grid = np.load(sample["fishey_grid"])
            fisheye_grid_z = cv2.resize(fisheye_grid[:, :, 2], (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
            depth = depth / fisheye_grid_z
        depth = depth.astype(np.float32)
        phi = np.array(0).astype(np.float32)
        roll = np.array(0).astype(np.float32)
        theta = 0

        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > min_depth
                
        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            sample["crop_height"], sample["crop_width"], cano_sz[0], cano_sz[0]*2, sample["cam_params"], roll, scale_fac=None
        )
        lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])
            
        # resizing process to fwd_sz.
        image, depth, pad, pred_scale_factor, attn_mask = resize_for_input((image * 255.).astype(np.uint8), depth, fwd_sz, sample["camera_intrinsics"], [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=erp_mask)
    else:
        attn_mask = np.ones_like(depth)
        lat_range = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32)
        long_range = torch.tensor([-np.pi, np.pi], dtype=torch.float32)
        
        # resizing process to fwd_sz.
        to_cano_ratio = cano_sz[0] / image.shape[0]
        image, depth, pad, pred_scale_factor = resize_for_input(image, depth, fwd_sz, sample["camera_intrinsics"], cano_sz, to_cano_ratio)


    # convert to tensor batch
    normalization_stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    image = TF.normalize(TF.to_tensor(image), **normalization_stats)
    gt = TF.to_tensor(depth)
    mask = TF.to_tensor((np.logical_and(depth > min_depth, depth < max_depth)).astype(np.uint8))
    attn_mask = TF.to_tensor((attn_mask>0).astype(np.float32)) # the non-empty region after ERP conversion
    batch = {
        "image": image.unsqueeze(0),
        "gt": gt.unsqueeze(0),
        "mask": mask.unsqueeze(0),
        "attn_mask": attn_mask.unsqueeze(0),
        "lat_range": lat_range.unsqueeze(0),
        "long_range": long_range.unsqueeze(0),
        "info": {
            "camera_intrinsics": sample["camera_intrinsics"],
            "pred_scale_factor": pred_scale_factor,
        },
    }
    
    #######################################################################
    ########################### model inference ###########################
    #######################################################################

    gt, mask, attn_mask, lat_range, long_range = batch["gt"].to(device), batch["mask"].to(device), batch['attn_mask'].to(device), batch["lat_range"].to(device), batch["long_range"].to(device)
    with torch.no_grad():
        preds, losses, _ = model(batch["image"].to(device), lat_range, long_range, gt, mask, attn_mask)
    preds *= pred_scale_factor
    
    #######################################################################
    ###########################  Visualization  ###########################
    #######################################################################
    save_img_dir = os.path.join(args.out_dir)
    os.makedirs(save_img_dir, exist_ok=True)
    if 'attn_mask' in batch.keys():
        attn_mask = batch['attn_mask'][0]
    else:
        attn_mask = None

    # adjust vis_depth_max for outdoor datasets
    if dataset_name == 'kitti360':
        vis_depth_max = 40.0
        vis_arel_max = 0.3
    elif dataset_name == 'kitti':
        vis_depth_max = 80.0
        vis_arel_max = 0.3
    else:
        # default indoor visulization parameters
        vis_depth_max = 10.0
        vis_arel_max = 0.5

    rgb = save_val_imgs_v2(
        0,
        preds[0],
        batch["gt"][0],
        batch["image"][0],
        f'{dataset_name}_output.jpg',
        save_img_dir,
        active_mask=attn_mask,
        valid_depth_mask=batch["mask"][0],
        depth_max=vis_depth_max,
        arel_max=vis_arel_max
    )
    
    ##########  Additioanal step for erp mode: converting the testing erp image back to original space for visualization  ##########
    pred_depth = preds[0, 0].detach().cpu().numpy()
    intrinsics = batch['info']['camera_intrinsics'][0].detach().cpu().numpy()
    if not sample['erp']:                    
        if dataset_name == 'kitti360':
            out_h = 700
            out_w = 700
            grid_fisheye = np.load(sample["fishey_grid"])
            grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
            grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
            cam_params={'dataset':'kitti360'}
        elif dataset_name == 'scannetpp':
            """
                Currently work perfect with phi = 0. For larger phi, corners may have artifacts.
            """
            grid_fisheye = np.load(sample["fishey_grid"])
            # set output size the same aspact ratio as raw image (no need to be same as fw_size)
            out_h = 500
            out_w = 750
            grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
            grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
            cam_params={'dataset':'scannetpp'} # when grid table is available, no need for intrinsic parameters
        else:
            # set output size the same aspact ratio as raw image (no need to be same as fw_size)
            out_h = sample["cam_params"]['h']
            out_w = sample["cam_params"]['w']
            grid_fisheye = None
            # cam_params={'dataset':'nyu', 'fx':intrinsics[0, 0]* s_ratio, 'fy':intrinsics[1, 1]* s_ratio, 'cx':intrinsics[0, 2]* s_ratio, 'cy':intrinsics[1, 2]* s_ratio}
            cam_params = sample["cam_params"]
            
        # convert the ERP result back to camera space for visualization (No need for original ERP image)
        if dataset_name not in ['matterport3d', 'gibson_v2']:    
            # scale the full erp_size depth scaling factor is equivalent to resizing data (given same aspect ratio)
            erp_h = cano_sz[0]
            erp_h = erp_h * batch['info']['pred_scale_factor']
            if 'f_align_factor' in batch['info']:
                erp_h = erp_h / batch['info']['f_align_factor'][0].detach().cpu().numpy()
            img_out, depth_out, valid_mask, active_mask, depth_out_gt = erp_patch_to_cam_fast(
                batch["image"][0], preds[0].detach().cpu(), attn_mask, 0., 0., out_h=out_h, out_w=out_w, erp_h=erp_h, erp_w=erp_h*2, cam_params=cam_params, 
                fisheye_grid2ray=grid_fisheye, depth_erp_gt=batch["gt"][0].detach().cpu())
            rgb = save_val_imgs_v2(
                0,
                depth_out,
                depth_out_gt,
                img_out,
                f'{dataset_name}_output_remap.jpg',
                save_img_dir,
                active_mask=active_mask,
                depth_max=vis_depth_max,
                arel_max=vis_arel_max
                )
        if args.save_pcd:
            pcd = reconstruct_pcd_erp(pred_depth, mask=(batch['attn_mask'][0][0]==0).numpy(), lat_range=batch['lat_range'][0], long_range=batch['long_range'][0])         
    else:
        if args.save_pcd:
            pcd = reconstruct_pcd(pred_depth, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        #     # the case raw fisheye image is loaded
        #     pcd = reconstruct_pcd_fisheye(pred_depth, grid_fisheye, mask=mask_fisheye)

    if args.save_pcd:
        save_pcd_dir = os.path.join(args.out_dir)
        os.makedirs(os.path.join(save_pcd_dir), exist_ok=True)
        pc_file = os.path.join(save_pcd_dir, f'{dataset_name}_pcd.ply')
        pcd = pcd.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)
        non_zero_indices = pcd[:, -1] > 0
        pcd = pcd[non_zero_indices]
        rgb = rgb[non_zero_indices]
        save_file_ply(pcd, rgb, pc_file)


def main(config: Dict[str, Any], args: argparse.Namespace):
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = eval(args.model_name).build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()
    
    samples = [SAMPLE_1, SAMPLE_2, SAMPLE_3]
    cano_sz=config["data"]["cano_sz"] # the ERP size model was trained on
    
    for i, sample in enumerate(samples):
        print(f"demo for sample {i}: {sample['dataset_name']}")
        demo_one_sample(model, device, sample, cano_sz, args)
    print("Demo finished")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    parser.add_argument("--config-file", type=str, default="configs/test/dac_swinl_indoor_test_scannetpp.json")
    parser.add_argument("--model-name", type=str, default="IDiscERP", help="Model name: IDiscERP, IDisc, IDiscEqui, CNNDepth")
    parser.add_argument("--model-file", type=str, default="checkpoints/dac_swinl_indoor.pt")
    parser.add_argument("--out-dir", type=str, default='demo/output')
    parser.add_argument("--save-pcd", action="store_true")

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    main(config, args)