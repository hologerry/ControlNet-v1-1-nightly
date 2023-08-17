import argparse
import os
import random

import cv2
import einops
import numpy as np
import torch

from PIL import Image
from pytorch_lightning import seed_everything
from scipy.ndimage import binary_dilation

import config

from annotator.util import HWC3
from cldm.ddim_hacked_blended import DDIMSampler
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


def read_image(img_path: str, dest_size=(512, 512)):
    image = Image.open(img_path).convert("RGB")
    image = image.resize(dest_size, Image.NEAREST)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).cuda()

    image = image * 2.0 - 1.0

    return image


def read_mask(mask_path: str, dilation_radius: int = 0, dest_size=(64, 64), img_size=(512, 512)):
    org_mask = Image.open(mask_path).convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask) / 255

    if dilation_radius > 0:
        k_size = 1 + 2 * dilation_radius
        masks_array = [binary_dilation(mask, structure=np.ones((k_size, k_size)))]
    else:
        masks_array = [mask]
    masks_array = np.array(masks_array).astype(np.float32)
    masks_array = masks_array[:, np.newaxis, :]
    masks_array = torch.from_numpy(masks_array).cuda()

    org_mask = org_mask.resize(img_size, Image.NEAREST)
    org_mask = np.array(org_mask).astype(np.float32) / 255.0
    org_mask = org_mask[None, None]
    org_mask[org_mask < 0.5] = 0
    org_mask[org_mask >= 0.5] = 1
    org_mask = torch.from_numpy(org_mask).cuda()

    return masks_array, org_mask


@torch.no_grad()
def one_image_batch(
    model,
    ddim_sampler,
    init_image,
    depth,
    mask,
    org_mask,
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
    skip_steps,
    percentage_of_pixel_blending,
):
    init_image_batch = torch.cat([init_image for _ in range(num_samples)], dim=0)
    mask_batch = torch.cat([mask for _ in range(num_samples)], dim=0)
    org_mask_batch = torch.cat([org_mask for _ in range(num_samples)], dim=0)
    H, W = init_image.shape[2:]

    detected_map = HWC3(depth)
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
        mask=mask_batch,
        org_mask=org_mask_batch,
        init_image=init_image_batch,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
        skip_steps=skip_steps,
        percentage_of_pixel_blending=percentage_of_pixel_blending,
    )

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    )

    results = [x_samples[i] for i in range(num_samples)]

    return results


def save_samples(init_image, depth, mask, org_mask, output, output_dir, img_basename):
    results = output

    image = init_image[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5
    image = image.clip(0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask_image = mask[0].permute(1, 2, 0).cpu().numpy() * 255
    mask_image = mask_image.clip(0, 255).astype(np.uint8)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

    org_mask_image = org_mask[0].permute(1, 2, 0).cpu().numpy() * 255
    org_mask_image = org_mask_image.clip(0, 255).astype(np.uint8)
    org_mask_image = cv2.cvtColor(org_mask_image, cv2.COLOR_GRAY2BGR)

    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(os.path.join(output_dir, f"{img_basename}.png"), image)
    cv2.imwrite(os.path.join(output_dir, f"{img_basename}_norm_depth.png"), depth)
    cv2.imwrite(os.path.join(output_dir, f"{img_basename}_mask.png"), mask_image)
    cv2.imwrite(os.path.join(output_dir, f"{img_basename}_org_mask.png"), org_mask_image)

    for i, result in enumerate(results):
        img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{i}.png"), img)


def main(args):
    if args.sd == "21":
        sd_model_name = "v2-1_512-ema-pruned.ckpt"
        model_name = "control_v11p_sd21_zoedepth" if args.zoe else "control_v11p_sd21_depth"
        config_name = f"{model_name}.yaml"
        model_name += ".safetensors"
    elif args.sd == "15":
        sd_model_name = "v1-5-pruned.ckpt"
        model_name = "control_v11f1p_sd15_depth.pth"
        config_name = "control_v11f1p_sd15_depth.yaml"
    else:
        raise NotImplementedError
    model = create_model(f"./models/{config_name}").cpu()
    model.load_state_dict(load_state_dict(f"./models/{sd_model_name}", location="cuda"), strict=False)
    model.load_state_dict(
        load_state_dict(f"./models/{model_name}", location="cuda", add_prefix="control_model"),
        strict=False,
    )

    model = model.cuda()

    ddim_sampler = DDIMSampler(model)

    os.makedirs(args.output_dir, exist_ok=True)

    img_basename = os.path.basename(args.img_path).split(".")[0]

    init_image = read_image(args.img_path)
    mask, org_mask = read_mask(args.mask_path, args.dilation_radius)

    depth_path = os.path.join(args.depth_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  # 16bit, millimeters
    depth_image = normalize_depth(depth)

    output = one_image_batch(
        model=model,
        ddim_sampler=ddim_sampler,
        init_image=init_image,
        depth=depth_image,
        mask=mask,
        org_mask=org_mask,
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
        prompt=args.prompt,
        a_prompt="best quality",
        n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
        config=config,
        skip_steps=0,
        percentage_of_pixel_blending=args.percentage_of_pixel_blending,
    )
    save_samples(init_image, depth_image, mask, org_mask, output, args.output_dir, img_basename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A parser for NOCS")

    parser.add_argument("--img_path", type=str, default="", required=True)
    parser.add_argument("--depth_path", type=str, default="", required=True)
    parser.add_argument("--mask_path", type=str, default="", required=True)
    parser.add_argument("--output_dir", type=str, default="./nocs_output/", required=True)

    parser.add_argument("--sd", type=str, default="15")
    parser.add_argument("--zoe", action="store_true", default=False)

    parser.add_argument("--prompt", type=str, default="", required=True)

    parser.add_argument("--dilation_radius", type=int, default=0)
    parser.add_argument("--percentage_of_pixel_blending", type=float, default=0.0)

    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
