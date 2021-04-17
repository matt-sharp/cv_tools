# 对成对的遥感大幅影像及标注进行同步裁剪
from pathlib import Path

from tqdm import tqdm
import cv2
import rasterio as rio
import tifffile
from torch.nn.modules.utils import _pair

#from torchseg.utils import scandir

ROTATE_MAP = {
    0: None,
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}

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

class CropBase:

    IMAGE_SUFFIX = ('.tif', '.img', '.tiff')
    MASK_SUFFIX = ('.tif', '.tiff')

    def __init__(self,
                 image_dir,
                 mask_dir,
                 out_dir,
                 patch_size=512,
                 slide_step=512,
                 rot_aug=True):
        self.patch_size = _pair(patch_size)
        self.slide_step = _pair(slide_step)

        self.rot_aug = rot_aug

        self.image_file = [
            Path(image_dir) / img
            for img in scandir(image_dir, suffix=self.IMAGE_SUFFIX)
        ]
        self.mask_file = [
            Path(mask_dir) / mask
            for mask in scandir(mask_dir, suffix=self.MASK_SUFFIX)
        ]
        assert len(self.image_file) == len(self.mask_file)

        # create output dir
        out_dir = Path(out_dir).expanduser().resolve()
        self.image_out_dir = out_dir / 'image'
        self.mask_out_dir = out_dir / 'label'
        self.image_out_dir.mkdir(exist_ok=True, parents=True)
        self.mask_out_dir.mkdir(exist_ok=True, parents=True)

    def crop_image(self):
        images = self.image_file
        masks = self.mask_file
        pbar = tqdm(zip(images, masks), total=len(images))
        for img_file, mask_file in pbar:
            assert img_file.stem == mask_file.stem

            file_stem = img_file.stem
            img_suffix = img_file.suffix
            mask_suffix = mask_file.suffix

            pbar.set_description(f'{file_stem}')

            img = rio.open(img_file)
            mask = rio.open(mask_file)
            height, width = img.height, img.width

            left, top = 0, 0
            # left_top_xy = []  # left-top corner coordinates (xmin, ymin)
            while left < width:
                if left + self.patch_size[0] >= width:
                    left = max(width - self.patch_size[0], 0)
                top = 0
                while top < height:
                    if top + self.patch_size[1] >= height:
                        top = max(height - self.patch_size[1], 0)
                    # right = min(left + self.patch_size[0], width - 1)
                    # bottom = min(top + self.patch_size[1], height - 1)
                    win = rio.windows.Window(left, top, self.patch_size[0],
                                             self.patch_size[1])
                    # transpose to channel_last format
                    patch_img = img.read(window=win).transpose(1, 2, 0)
                    patch_mask = mask.read(window=win).transpose(1, 2, 0)

                    # filter out background demainated patch
                    one_channel_img = patch_img.sum(axis=2)
                    if (one_channel_img == 0).sum() == one_channel_img.size:
                        if top + self.patch_size[1] >= height:
                            break
                        else:
                            top += self.slide_step[1]
                        continue

                    if self.rot_aug:
                        # augmentation
                        for a in (0, 90, 180, 270):  # angle
                            out_file = f'{file_stem}__{left}__{top}__{a}{img_suffix}'
                            mode = ROTATE_MAP[a]
                            if mode is None:
                                rimg = patch_img.copy()
                                rmask = patch_mask.copy()
                            else:
                                rimg = cv2.rotate(patch_img, mode)
                                rmask = cv2.rotate(patch_mask, mode)

                            tifffile.imsave(str(self.image_out_dir / out_file),
                                            rimg)
                            tifffile.imsave(str(self.mask_out_dir / out_file),
                                            rmask)
                            # Horizal flip
                            out_file = f'{file_stem}__{left}__{top}__{a}_hflip{img_suffix}'
                            rimg = cv2.flip(rimg, 1)
                            rmask = cv2.flip(rmask, 1)
                            tifffile.imsave(str(self.image_out_dir / out_file),
                                            rimg)
                            tifffile.imsave(str(self.mask_out_dir / out_file),
                                            rmask)
                    else:
                        out_file = f'{file_stem}__{left}__{top}{img_suffix}'
                        tifffile.imsave(str(self.image_out_dir / out_file),
                                        patch_img)
                        tifffile.imsave(str(self.mask_out_dir / out_file),
                                        patch_mask)

                    # left_top_xy.append((left, top))
                    if top + self.patch_size[1] >= height:
                        break
                    else:
                        top += self.slide_step[1]

                if left + self.patch_size[0] >= width:
                    break
                else:
                    left += self.slide_step[0]

            img.close()
            mask.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='raw image directory')
    parser.add_argument('--mask_dir', type=str, help='raw label directory')
    parser.add_argument('--out_dir',
                        type=str,
                        help='output directory of cropped image and label')
    parser.add_argument('--patch_size',
                        '-p',
                        type=int,
                        default=512,
                        help='patch image size')
    parser.add_argument('--slide_step',
                        '-s',
                        type=int,
                        default=512,
                        help='slide step when cropping image')
    parser.add_argument(
        '--rot_aug',
        '-r',
        action='store_true',
        help='whether perform rotation and flip augmentation during cropping')

    args = parser.parse_args()

    base = CropBase(image_dir=args.image_dir,
                    mask_dir=args.mask_dir,
                    out_dir=args.out_dir,
                    patch_size=args.patch_size,
                    slide_step=args.slide_step,
                    rot_aug=args.rot_aug)

    base.crop_image()


if __name__ == "__main__":
    main()
