import argparse
import datetime
import os
import random

import cv2
import einops
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch_lightning import seed_everything
from tqdm import tqdm

import config


def main(args):
    image_files = os.listdir(args.data_dir)
    if args.debug:
        image_files = image_files[:1]

    cur_ann_output_dir = os.path.join(args.output_dir, args.ann)
    seg_ann_output_dir = os.path.join(args.output_dir, "seg")

    for img_file in tqdm(image_files, desc=f"Processing images ann {args.ann}"):
        img_basename = os.path.splitext(img_file)[0]
        img_path = os.path.join(args.data_dir, img_file)
        input_img = cv2.imread(img_path)

        detected_map = cv2.imread(
            os.path.join(seg_ann_output_dir, f"{img_basename}_detected.png"), cv2.IMREAD_GRAYSCALE
        )
        depth_threshold = 150
        detected_mask = ((detected_map > depth_threshold) * 255).astype(np.uint8)

        for i in range(args.num_samples):
            diffusion_img = cv2.imread(os.path.join(cur_ann_output_dir, f"{img_basename}_{i}.png"))
            input_img = cv2.resize(input_img, (diffusion_img.shape[1], diffusion_img.shape[0]))
            detected_mask = cv2.resize(
                detected_mask, (diffusion_img.shape[1], diffusion_img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            detected_mask = cv2.cvtColor(detected_mask, cv2.COLOR_GRAY2BGR)

            # Blend
            blended_img = (input_img * (1 - detected_mask / 255) + diffusion_img * (detected_mask / 255)).astype(
                np.uint8
            )
            cv2.imwrite(os.path.join(cur_ann_output_dir, f"{img_basename}_{i}_blended.png"), blended_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A collection of util tools for AzureML")

    parser.add_argument("--ann", type=str, default="depth")
    parser.add_argument("--data_dir", type=str, default="../data/water_bottle_all_renamed/single", required=True)
    parser.add_argument("--output_dir", type=str, default="./water_bottle_output/", required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.ann == "all":
        for ann in tqdm(["depth", "normal", "canny", "seg", "lineart"]):
            args.ann = ann
            main(args)
    else:
        main(args)