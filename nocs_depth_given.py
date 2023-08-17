import argparse
import os
import random

import cv2
import einops
import numpy as np
import torch

from pytorch_lightning import seed_everything
from tqdm import tqdm

import config

from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *


def normalize_depth(depth):
    depth_norm = np.copy(depth)
    depth_norm = depth_norm / 1000.0
    vmin = np.percentile(depth_norm, 2)
    vmax = np.percentile(depth_norm, 85)

    depth_norm -= vmin
    depth_norm /= vmax - vmin
    depth_norm = 1.0 - depth_norm
    depth_image = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)

    return depth_image


@torch.no_grad()
def one_image_batch(
    model,
    ddim_sampler,
    input_image,
    depth_image,
    detect_resolution,
    image_resolution,
    low_threshold,
    high_threshold,
    num_samples,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    prompt,
    a_prompt,
    n_prompt,
    config,
):
    input_image = HWC3(input_image)
    detected_map = HWC3(depth_image)

    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)],
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
    }
    shape = (4, H // 8, W // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = (
        [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    )
    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
    )

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    )

    results = [x_samples[i] for i in range(num_samples)]

    return detected_map, results


def save_samples(input_img, output, output_dir, img_basename):
    detected_map, results = output

    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{img_basename}.png"), input_img)
    cv2.imwrite(os.path.join(output_dir, f"{img_basename}_detected.png"), detected_map)
    for i, result in enumerate(results):
        img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{i}.png"), img)
        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{i}_detected.png"), detected_map)


def main(args):
    model_name = "control_v11f1p_sd15_depth"
    model = create_model(f"./models/{model_name}.yaml").cpu()
    model.load_state_dict(load_state_dict("./models/v1-5-pruned.ckpt", location="cuda"), strict=False)
    model.load_state_dict(load_state_dict(f"./models/{model_name}.pth", location="cuda"), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    os.makedirs(args.output_dir, exist_ok=True)

    img_path = os.path.join(args.img_path)
    img_basename = os.path.basename(img_path).split(".")[0]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    depth_path = os.path.join(args.depth_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  # 16bit, millimeters
    depth_image = normalize_depth(depth)
    output = one_image_batch(
        model=model,
        ddim_sampler=ddim_sampler,
        input_image=img,
        depth_image=depth_image,
        detect_resolution=512,
        image_resolution=512,
        low_threshold=100,
        high_threshold=200,
        num_samples=args.num_samples,
        ddim_steps=20,
        guess_mode=False,
        strength=1.0,
        scale=9.0,
        seed=12345,
        eta=1.0,
        prompt="a bottle on table with water in it and label on it",
        a_prompt="best quality",
        n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
        config=config,
    )
    save_samples(img, output, args.output_dir, img_basename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A parser for NOCS")

    parser.add_argument("--img_path", type=str, default="", required=True)
    parser.add_argument("--depth_path", type=str, default="", required=True)
    parser.add_argument("--output_dir", type=str, default="./nocs_output/", required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
