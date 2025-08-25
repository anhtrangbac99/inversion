# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm


def main():
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    opt['input_folder'] = './data/GOPRO/test/input'
    opt['save_folder'] = './data/GOPRO/test/blur_crops'
    opt['crop_size'] = 512
    opt['step'] = 256
    opt['thresh_size'] = 0
    extract_subimages(opt)

    opt['input_folder'] = './data/GOPRO/test/target'
    opt['save_folder'] = './data/GOPRO/test/sharp_crops'
    opt['crop_size'] = 512
    opt['step'] = 256
    opt['thresh_size'] = 0
    extract_subimages(opt)



def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        # sys.exit(1)

    img_list = [os.path.join(input_folder,i) for i in os.listdir(input_folder)]

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    crop_size = opt["crop_size"]
    step = opt["step"]
    thresh_size = opt["thresh_size"]
    img_name, _ = osp.splitext(osp.basename(path))  # ignore original ext
    extension = ".png"  # force PNG output

    # Strip DIV2K-like suffixes
    for tag in ("x2", "x3", "x4", "x8"):
        img_name = img_name.replace(tag, "")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return f"Read fail: {path}"

    # handle too-small images
    h, w = img.shape[:2]
    if min(h, w) < crop_size:
        if opt.get("pad_small", False):
            img = maybe_pad(img, crop_size)
            h, w = img.shape[:2]
        elif opt.get("skip_small", True):
            return f"Skip small: {img_name} ({w}x{h})"

    # sliding window coords
    h_space = np.arange(0, max(1, h - crop_size + 1), step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, max(1, w - crop_size + 1), step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    params = [cv2.IMWRITE_PNG_COMPRESSION, opt.get("compression_level", 3)]

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped = img[int(x):int(x)+crop_size, int(y):int(y)+crop_size, ...]
            if cropped.size == 0 or cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
                continue
            cropped = np.ascontiguousarray(cropped)
            cv2.imwrite(
                osp.join(opt["save_folder"], f"{img_name}_s{index:03d}{extension}"),
                cropped,
                params
            )
    return f"Processed {img_name} ({index} crops)"



if __name__ == '__main__':
    main()