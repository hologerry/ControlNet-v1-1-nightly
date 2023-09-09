import os

from argparse import ArgumentParser
from multiprocessing import Pool

import torch


def print_cmd(cmd):
    print(cmd)
    os.system(cmd)


def one_job_process(args):
    available_gpus = torch.cuda.device_count()
    cmds = []
    for gpu_idx in range(args.gpu_num):
        cmd = f"export CUDA_VISIBLE_DEVICES={gpu_idx%available_gpus} && python bot_render_depth_bottle.py"
        cmd += f" --seed {args.seed}"
        # cmd += f" --dilation_radius {args.dilation_radius}"
        cmd += f" --job_idx {args.job_idx} --job_num {args.job_num} --gpu_idx {gpu_idx} --gpu_num {args.gpu_num}"
        if args.sub_job_num > 0:
            cmd += f" --sub_job_idx {args.sub_job_idx} --sub_job_num {args.sub_job_num}"

        cmds.append(cmd)

    pool = Pool(args.gpu_num)
    pool.map(print_cmd, cmds)
    pool.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    # parser.add_argument("--dilation_radius", type=int, default=1)

    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--job_num", type=int, default=1)
    parser.add_argument("--gpu_num", type=int, default=8)
    parser.add_argument("--sub_job_idx", type=int, default=-1)
    parser.add_argument("--sub_job_num", type=int, default=-1)

    args = parser.parse_args()

    print(f"Job {args.job_idx} started")
    one_job_process(args)
