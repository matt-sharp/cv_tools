from collections import OrderedDict
import os
import os.path as osp
from pathlib import Path
import time

from tqdm import tqdm
import numpy as np
# import cv2
from osgeo import gdal

import torch
import torch.nn as nn
import torchvision.transforms as T

from torchseg import utils
from torchseg.utils import Config, get_root_logger, collect_env, scandir
from torchseg.datasets import build_dataset, build_dataloader
from torchseg.models import build_model
from torchseg.optimizer import build_optimizer
from torchseg.runner import Runner, init_dist, load_checkpoint, get_dist_info
from torchseg.apis import single_gpu_test
from torchseg.datasets.rs_dataset import RSInferenceDataset
from torchseg.apis.test import extract_label


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='TorchSeg inference')

    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image_file', help='remote sensing image file')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--patch_size',
                        type=int,
                        default=512,
                        help='patch size')
    parser.add_argument('--slide_step',
                        type=int,
                        default=512,
                        help='slide step')
    parser.add_argument('--pad_size', type=int, default=256)
    parser.add_argument('--to_rgb',
                        action='store_true',
                        help='whether convert image to RGB(uint8)')
    parser.add_argument('--tta', action='store_true', help='TTA flag')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    print(args)

    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    num_classes = len(model.CLASSES)

    if not distributed:
        model = nn.DataParallel(model.cuda(), device_ids=[0])
    else:
        model = nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
    model.eval()
    rank, _ = get_dist_info()

    def to_tensor(img):
        return torch.from_numpy(img.astype('f4').transpose(2, 0, 1))

    tsf = T.Compose([
        T.Lambda(to_tensor),
        T.Normalize(**cfg.img_norm_cfg),
    ])
    patch_size = args.patch_size
    slide_step = args.slide_step
    pad_size = args.pad_size

    suffixs = ('.tif', '.tiff', '.img')
    if osp.isfile(args.image_file) and args.image_file.endswith(suffixs):
        image_list = [args.image_file]
    else:
        image_list = [
            osp.join(args.image_file, f)
            for f in scandir(args.image_file, suffix=suffixs)
        ]

    for img_file in image_list:
        dataset = RSInferenceDataset(img_file,
                                     patch_size=patch_size,
                                     slide_step=slide_step,
                                     pad=pad_size,
                                     to_rgb=args.to_rgb,
                                     transform=tsf)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # out_labels = np.zeros((dataset.height, dataset.width), dtype=np.uint8)
        if rank == 0:
            basename = Path(img_file).stem
            out_file = osp.join(args.out_dir, f'{basename}_classify.tif')
            driver = gdal.GetDriverByName('GTiff')
            src_ds = gdal.Open(img_file)
            out_raster = driver.Create(out_file, dataset.width, dataset.height,
                                       1, gdal.GDT_Byte)
            gt = src_ds.GetGeoTransform()
            if gt is not None:
                out_raster.SetGeoTransform(gt)
            out_raster.SetProjection(src_ds.GetProjection())
            src_ds = None

            out_band = out_raster.GetRasterBand(1)
        pbar = tqdm(data_loader)
        for img, offset in pbar:
            pbar.set_description(Path(img_file).name)
            x_offset, y_offset = offset
            img = img.cuda()
            with torch.no_grad():
                result1 = model(img)
                result1 = result1['out']

                if args.tta:
                    result2 = model(torch.flip(img, [-1]))  # horizontal flip
                    result2 = torch.flip(result2['out'], [-1])

                    result3 = model(torch.flip(img, [-2]))  # vertical flip
                    result3 = torch.flip(result3['out'], [-2])

                    result4 = model(torch.flip(img, [-1, -2]))  # diagonal flip
                    result4 = torch.flip(result4['out'], [-1, -2])

                    pred = torch.mean(result1 + result2 + result3 + result4,
                                      dim=1,
                                      keepdim=True)
                else:
                    pred = result1

                # pred = result['out']  # nB,nC,H,W
                pred_label = extract_label(
                    pred[..., pad_size:-pad_size, pad_size:-pad_size],
                    num_classes).cpu().numpy().squeeze()
                # out_labels[y_offset:y_offset + slide_step[1], x_offset:x_offset +
                #            slide_step[0]] = pred_label
                if rank == 0:
                    out_band.WriteArray(pred_label.astype(np.uint8),
                                        xoff=x_offset.item(),
                                        yoff=y_offset.item())

        # basename = Path(args.image_file).stem
        # out_file = osp.join(args.out_dir, f'{basename}_mask.png')
        # colored_pred = get_mask_pallete(cfg.data_type)[out_labels.astype(np.uint8)]
        # cv2.imwrite(out_file, colored_pred[..., ::-1])
        if rank == 0:
            out_raster = None


if __name__ == "__main__":
    main()
