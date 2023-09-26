import argparse
import json
import os
import random

import cv2
import einops
import numpy as np
import torch

from PIL import Image
from pytorch_lightning import seed_everything
from scipy.ndimage import binary_dilation
from tqdm import tqdm

import config

from annotator.util import HWC3, resize_image
from annotator.zoe import ZoeDetector
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *


def read_image(img_path: str, dest_size=(512, 512)):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    if w > dest_size[0] or h > dest_size[1]:
        image = cv2.resize(image, dest_size, interpolation=cv2.INTER_AREA)
    elif w < dest_size[0] or h < dest_size[1]:
        image = cv2.resize(image, dest_size, interpolation=cv2.INTER_CUBIC)

    return image


@torch.no_grad()
def one_image_batch(
    model,
    ddim_sampler,
    image,
    detected_map,
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
    H, W, C = image.shape
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

    return results


def save_samples(init_image, detected_map, num_prompts, prompt_idx, results, output_dir, img_basename):
    # the img_basename contains the subfolder name
    true_dir = os.path.dirname(os.path.join(output_dir, img_basename))
    os.makedirs(true_dir, exist_ok=True)
    if prompt_idx == 0:
        # save the input only once
        # image = init_image[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5
        # image = image.clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(init_image, cv2.COLOR_RGB2BGR)

        detected = cv2.cvtColor(detected_map, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_color.png"), image)
        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_det_depth.png"), detected)

    for i, result in enumerate(results):
        img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_prompt{prompt_idx}_{i}.png"), img)


def file_ok(file_path):
    if not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) == 0:
        return False
    # img = cv2.imread(file_path)
    # if img is None:
    #     return False
    return True


def current_sample_ok(num_prompts, num_samples, output_dir, img_basename):
    image_path = os.path.join(output_dir, f"{img_basename}_color.png")
    if not file_ok(image_path):
        print(f"image_path {image_path} not ok")
        return False
    # depth_path = os.path.join(output_dir, f"{img_basename}_norm_depth.png")
    # if not file_ok(depth_path):
    #     print(f"depth_path {depth_path} not ok")
    #     return False
    # mask_image_path = os.path.join(output_dir, f"{img_basename}_mask.png")
    # if not file_ok(mask_image_path):
    #     print(f"mask_image_path {mask_image_path} not ok")
    #     return False
    # org_mask_image_path = os.path.join(output_dir, f"{img_basename}_org_mask.png")
    # if not file_ok(org_mask_image_path):
    #     print(f"org_mask_image_path {org_mask_image_path} not ok")
    #     return False
    for prompt_idx in range(num_prompts):
        for i in range(num_samples):
            res_img_path = os.path.join(output_dir, f"{img_basename}_prompt{prompt_idx}_{i}.png")
            if not file_ok(res_img_path):
                print(f"res_img_path {res_img_path} not ok")
                return False
    print(f"current_sample_ok {img_basename} good")
    return True


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

    preprocessor = ZoeDetector()

    prompts = [
        "a mug on table",
        "a glass mug on table",
        "a plastic mug on table",
        "a transparent mug on table",
        # "a mug with label and cap, contains water, on table",
        # "a glass mug with label and cap, contains water, on table",
        # "a plastic mug with label and cap, contains water, on table",
        # "a transparent mug with label and cap, contains water, on table",
    ]
    num_prompts = len(prompts)

    data_root_path = "../bot_render_output"
    splits = ["train", "test"]

    all_pair_filenames_json = os.path.join(data_root_path, f"bot_render_mug_pair_filenames.json")
    with open(all_pair_filenames_json, "r") as f:
        data_dict = json.load(f)

    if args.debug:
        # splits = ["val"]
        splits = ["test"]

    for split in splits:
        cur_split_path = os.path.join(data_root_path, f"{split}_pair")
        cur_split_output_path = os.path.join(data_root_path, f"{split}_bc_mug_depth_seed{args.seed}")
        cur_split_pairs = data_dict[split]

        cur_job_pairs = cur_split_pairs[args.part_idx :: args.part_num]
        if args.sub_job_num > 0:
            cur_job_pairs = cur_job_pairs[args.sub_job_idx :: args.sub_job_num]

        desc_str = f"Job {args.job_idx} part [{args.part_idx}/{args.part_num}] Processing {split}"
        if args.sub_job_num > 0:
            desc_str += f" sub job [{args.sub_job_idx}/{args.sub_job_num}]"
        for pair in tqdm(cur_job_pairs, desc=desc_str):
            # filename already contains the subfolder name
            color_filename = pair["color_filename"]
            # mask_filename = pair["mask_filename"]

            # depth_filename = pair["syn_depth_filename"]
            out_base_filename = color_filename.replace("_color", "").replace(".png", "")
            if current_sample_ok(num_prompts, args.num_samples, cur_split_output_path, out_base_filename):
                continue
            color_path = os.path.join(cur_split_path, color_filename)
            # mask_path = os.path.join(cur_split_path, mask_filename)
            # depth_path = os.path.join(cur_split_path, depth_filename)
            if not os.path.exists(color_path) or os.path.getsize(color_path) == 0:
                print(f"pair color_path {color_path} not ok")
                continue
            # if not os.path.exists(mask_path) or os.path.getsize(mask_path) == 0:
            #     print(f"pair mask_path {mask_path} not ok")
            #     continue
            # if not os.path.exists(depth_path) or os.path.getsize(depth_path) == 0:
            #     print(f"pair depth_path {depth_path} not ok")
            #     continue

            init_image = read_image(color_path)

            input_image = HWC3(init_image)
            detected_map = preprocessor(resize_image(input_image, 512))
            detected_map = HWC3(detected_map)

            img = resize_image(input_image, 512)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            # mask, org_mask = read_mask(mask_path, args.dilation_radius)

            # depth_path = os.path.join(depth_path)
            # depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  # 16bit, millimeters
            # depth_image = normalize_depth(depth)

            for prompt_idx in range(len(prompts)):
                cur_prompt = prompts[prompt_idx]
                results = one_image_batch(
                    model=model,
                    ddim_sampler=ddim_sampler,
                    image=img,
                    detected_map=detected_map,
                    num_samples=args.num_samples,
                    ddim_steps=20,
                    guess_mode=False,
                    strength=1.0,
                    scale=9.0,
                    seed=args.seed,
                    eta=1.0,
                    prompt=cur_prompt,
                    a_prompt="best quality",
                    n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
                    config=config,
                )
                save_samples(
                    init_image,
                    detected_map,
                    num_prompts,
                    prompt_idx,
                    results,
                    cur_split_output_path,
                    out_base_filename,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A parser for DREDS")

    parser.add_argument("--sd", type=str, default="15")
    parser.add_argument("--zoe", action="store_true", default=False)

    # parser.add_argument("--dilation_radius", type=int, default=1)
    # parser.add_argument("--percentage_of_pixel_blending", type=float, default=0.0)

    parser.add_argument("--num_samples", type=int, default=4)

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--job_num", type=int, default=1)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=8)
    parser.add_argument("--part_idx", type=int, default=0)
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--sub_job_idx", type=int, default=-1)
    parser.add_argument("--sub_job_num", type=int, default=-1)

    args = parser.parse_args()

    assert args.job_idx < args.job_num
    assert args.gpu_idx < args.gpu_num
    args.part_num = args.job_num * args.gpu_num
    args.part_idx = args.job_idx * args.gpu_num + args.gpu_idx

    main(args)
