"""
PTC Whole Slide
Contains all functions for processing.

Reference source:
https://github.com/BMIRDS/deepslide/blob/master/code/utils_processing.py

Modify is_purple to is_white
Delete BALANCING CLASS DISTRIBUTION (using different weights in the model instead)

"""

import functools
import itertools
import math
import time
from multiprocessing import (Process, Queue, RawArray)
from pathlib import Path
from shutil import copyfile
from typing import (Callable, Dict, List, Tuple)

import numpy as np
from PIL import Image
from imageio import (imsave, imread)
from skimage.measure import block_reduce

from utils import (get_all_image_paths, get_image_names, get_image_paths,
                   get_subfolder_paths)

Image.MAX_IMAGE_PIXELS = None

###########################################
#         FITERLING WHITE AREAS           #
###########################################
def is_white(crop: np.ndarray, white_threshold: int,
             scale_size: int) -> bool:
    """
    Determines if a given portion of an image is white.
    Args:
        crop: Portion of the image to check for being white.
        white_threshold: Number of white points for region to be considered white.
        scale_size: Scalar to use for reducing image to check for white.
    Returns:
        A boolean representing whether the image is white or not.
    """
    block_size = (crop.shape[0] // scale_size,
                  crop.shape[1] // scale_size, 1)
    pooled = block_reduce(image=crop, block_size=block_size, func=np.average)

    # Calculate boolean arrays for determining if portion is white.
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    cond1 = r > 200
    cond2 = b > 200
    cond3 = g > 200
    cond4 = abs(r-b) < 6
    cond5 = abs(g-b) < 6
    cond6 = abs(r-g) < 6

    # Find the indexes of pooled satisfying all 3 conditions.
    pooled = pooled[cond1 & cond2 & cond3 & cond4 & cond5 & cond6]
    num_white = pooled.shape[0]

    return num_white < white_threshold

###########################################
#         PRODUCING PATCHES               #
###########################################
def produce_patches(input_folder: Path, output_folder: Path,
                    inverse_overlap_factor: float, by_folder: bool,
                    num_workers: int, patch_size: int, white_threshold: int,
                    scale_size: int, image_ext: str,
                    type_histopath: bool) -> None:
    """
    Produce the patches from the WSI in parallel.
    Args:
        input_folder: Folder containing the WSI.
        output_folder: Folder to save the patches to.
        inverse_overlap_factor: Overlap factor used in patch creation.
        by_folder: Whether to generate the patches by folder or by image.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        white_threshold: Number of white points for region to be considered white.
        scale_size: Scalar to use for reducing image to check for white.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for white histopathology images and filter whitespace.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    image_locs = get_all_image_paths(
        master_folder=input_folder) if by_folder else get_image_names(
            folder=input_folder)
    outputted_patches = 0

    print(f"\ngetting small crops from {len(image_locs)} "
          f"images in {input_folder} "
          f"with inverse overlap factor {inverse_overlap_factor:.2f} "
          f"outputting in {output_folder}")

    start_time = time.time()

    for image_loc in image_locs:
        image = imread(
            uri=(image_loc if by_folder else input_folder.joinpath(image_loc)), pilmode="RGB")


        img = RawArray(
            typecode_or_type=np.ctypeslib.as_ctypes_type(dtype=image.dtype),
            size_or_initializer=image.size)
        img_np = np.frombuffer(buffer=img,
                               dtype=image.dtype).reshape(image.shape)
        np.copyto(dst=img_np, src=image)

        # Number of x starting points.
        x_steps = int((image.shape[0] - patch_size) / patch_size *
                      inverse_overlap_factor) + 1
        # Number of y starting points.
        y_steps = int((image.shape[1] - patch_size) / patch_size *
                      inverse_overlap_factor) + 1
        # Step size, same for x and y.
        step_size = int(patch_size / inverse_overlap_factor)

        # Create the queues for passing data back and forth.
        in_queue = Queue()
        out_queue = Queue(maxsize=-1)

        # Create the processes for multiprocessing.
        processes = [
            Process(target=find_patch_mp,
                    args=(functools.partial(
                        find_patch,
                        output_folder=output_folder,
                        image=img_np,
                        by_folder=by_folder,
                        image_loc=image_loc,
                        white_threshold=white_threshold,
                        scale_size=scale_size,
                        image_ext=image_ext,
                        type_histopath=type_histopath,
                        patch_size=patch_size), in_queue, out_queue))
            for __ in range(num_workers)
        ]
        for p in processes:
            p.daemon = True
            p.start()

        # Put the (x, y) coordinates in the input queue.
        for xy in itertools.product(range(0, x_steps * step_size, step_size),
                                    range(0, y_steps * step_size, step_size)):
            in_queue.put(obj=xy)

        # Store num_workers None values so the processes exit when not enough jobs left.
        for __ in range(num_workers):
            in_queue.put(obj=None)

        num_patches = sum([out_queue.get() for __ in range(x_steps * y_steps)])

        # Join the processes as they finish.
        for p in processes:
            p.join(timeout=1)

        if by_folder:
            print(f"{image_loc}: num outputted windows: {num_patches}")
        else:
            outputted_patches += num_patches

    if not by_folder:
        print(
            f"finished patches from {input_folder} "
            f"with inverse overlap factor {inverse_overlap_factor:.2f} in {time.time() - start_time:.2f} seconds "
            f"outputting in {output_folder} "
            f"for {outputted_patches} patches")

def find_patch_mp(func: Callable[[Tuple[int, int]], int], in_queue: Queue,
                  out_queue: Queue) -> None:
    """
    Find the patches from the WSI using multiprocessing.
    Helper function to ensure values are sent to each process
    correctly.
    Args:
        func: Function to call in multiprocessing.
        in_queue: Queue containing input data.
        out_queue: Queue to put output in.
    """
    while True:
        xy = in_queue.get()
        if xy is None:
            break
        out_queue.put(obj=func(xy))

def find_patch(xy_start: Tuple[int, int], output_folder: Path,
               image: np.ndarray, by_folder: bool, image_loc: Path,
               patch_size: int, image_ext: str, type_histopath: bool,
               white_threshold: int, scale_size: int) -> int:
    """
    Find the patches for a WSI.
    Args:
        output_folder: Folder to save the patches to.
        image: WSI to extract patches from.
        xy_start: Starting coordinates of the patch.
        by_folder: Whether to generate the patches by folder or by image.
        image_loc: Location of the image to use for creating output filename.
        patch_size: Size of the patches extracted from the WSI.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for white histopathology images and filter whitespace.
        white_threshold: Number of white points for region to be considered white.
        scale_size: Scalar to use for reducing image to check for white.
    Returns:
        The number 1 if the image was saved successfully and a 0 otherwise.
        Used to determine the number of patches produced per WSI.
    """
    x_start, y_start = xy_start

    patch = image[x_start:x_start + patch_size, y_start:y_start +
                  patch_size, :]
    # Sometimes the images are RGBA instead of RGB. Only keep RGB channels.
    patch = patch[..., [0, 1, 2]]

    if by_folder:
        output_subsubfolder = output_folder.joinpath(
            Path(image_loc.name).with_suffix(""))
        output_subsubfolder = output_subsubfolder.joinpath(
            output_subsubfolder.name)
        output_subsubfolder.mkdir(parents=True, exist_ok=True)
        output_path = output_subsubfolder.joinpath(
            f"{str(x_start).zfill(5)};{str(y_start).zfill(5)}.{image_ext}")
    else:
        output_path = output_folder.joinpath(
            f"{image_loc.stem}_{x_start}_{y_start}.{image_ext}")
        print(output_path)

    if type_histopath:
        if is_white(crop=patch,
                     white_threshold=white_threshold,
                     scale_size=scale_size):
            imsave(uri=output_path, im=patch)
        else:
            return 0
    else:
        imsave(uri=output_path, im=patch)
    return 1

###########################################
#         GENERATING TRAINING DATA        #
###########################################

def get_folder_size_and_num_images(folder: Path) -> Tuple[float, int]:
    """
    Finds the number and size of images in a folder path.
    Used to decide how much to slide windows.
    Args:
        folder: Folder containing images.
    Returns:
        A tuple containing the total size of the images and the number of images.
    """
    image_paths = get_image_paths(folder=folder)

    file_size = 0
    for image_path in image_paths:
        file_size += image_path.stat().st_size

    file_size_mb = file_size / 1e6
    return file_size_mb, len(image_paths)



def gen_train_patches(input_folder: Path, output_folder: Path,
                      num_workers: int,
                      inverse_overlap_factor: float,
                      patch_size: int, white_threshold: int,
                      scale_size: int, image_ext: str,
                      type_histopath: bool) -> None:
    """
    Generates all patches for subfolders in the training set.
    Args:
        input_folder: Folder containing the subfolders containing WSI.
        output_folder: Folder to save the patches to.
        num_train_per_class: The desired number of training patches per class.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        white_threshold: Number of white points for region to be considered white.
        scale_size: Scalar to use for reducing image to check for white.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for white histopathology images and filter whitespace.
    """
    # Find the subfolders and how much patches should overlap for each.
    subfolders = get_subfolder_paths(folder=input_folder)
    print(f"{subfolders} subfolders found from {input_folder}")


    # Produce the patches.
    for input_subfolder in subfolders:
        produce_patches(input_folder=input_subfolder,
                        output_folder=output_folder.joinpath(
                            input_subfolder.name),
                        inverse_overlap_factor=inverse_overlap_factor,
                        by_folder=False,
                        num_workers=num_workers,
                        patch_size=patch_size,
                        white_threshold=white_threshold,
                        scale_size=scale_size,
                        image_ext=image_ext,
                        type_histopath=type_histopath)

    print("\nfinished all folders\n")


def gen_val_patches(input_folder: Path, output_folder: Path,
                    overlap_factor: float, num_workers: int, patch_size: int,
                    white_threshold: int, scale_size: int,
                    image_ext: str, type_histopath: bool) -> None:
    """
    Generates all patches for subfolders in the validation set.
    Args:
        input_folder: Folder containing the subfolders containing WSI.
        output_folder: Folder to save the patches to.
        overlap_factor: The amount of overlap between patches.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        white_threshold: Number of white points for region to be considered white.
        scale_size: Scalar to use for reducing image to check for white.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for white histopathology images and filter whitespace.
    """
    # Find the subfolders and how much patches should overlap for each.
    subfolders = get_subfolder_paths(folder=input_folder)
    print(f"{len(subfolders)} subfolders found from {input_folder}")

    # Produce the patches.
    for input_subfolder in subfolders:
        produce_patches(input_folder=input_subfolder,
                        output_folder=output_folder.joinpath(
                            input_subfolder.name),
                        inverse_overlap_factor=overlap_factor,
                        by_folder=False,
                        num_workers=num_workers,
                        patch_size=patch_size,
                        white_threshold=white_threshold,
                        scale_size=scale_size,
                        image_ext=image_ext,
                        type_histopath=type_histopath)

    print("\nfinished all folders\n")
