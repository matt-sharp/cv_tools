from pathlib import Path
import os.path as osp

from tqdm import tqdm
import cv2
import numpy as np


def scandir(dir_path, suffix=None, recursive=False):

    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must ne a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path,
                                        suffix=suffix,
                                        recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
def cal_dir_stat(root, num_channels=3, max_value=None):
    root = Path(root).expanduser().resolve()
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)

    img_list = list(scandir(root, suffix=('.jpg', '.png', '.tif')))
    for img_path in tqdm(img_list):
        img_path = osp.join(root, img_path)
        # image in M*N*num_channels shape, channel in BGR order
        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype('float32')
        # im = im / 255.0
        if max_value is not None:
            im = im / max_value
        pixel_num += (im.size / num_channels)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='training set directory')
    parser.add_argument('--channels',
                        '-c',
                        type=int,
                        default=3,
                        help='number of channels')

    args = parser.parse_args()

    mean, std = cal_dir_stat(args.root, args.channels)
    print(f"mean:{mean}\nstd: {std}")


if __name__ == "__main__":
    main()
