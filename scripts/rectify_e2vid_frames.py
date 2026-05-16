#!/usr/bin/env python3
"""Apply EVB_CIRS stereo rectification to a folder of E2VID-reconstructed frames.

The rectification maps were precomputed once via
`EMatch/datasets/EVB_CIRS/calibration/StereoCalibration` (cv2.stereoRectify +
cv2.initUndistortRectifyMap) and saved to
`EMatch/data/EVB_CIRS/calibration/rectify_maps_{left,right}.npz`. Each .npz
contains `map_x` and `map_y` arrays of shape (720, 1280) float32 — direct
inputs to cv2.remap.

Usage:
    python scripts/rectify_e2vid_frames.py \\
        --in  output/1_0m_out_left/reconstruction \\
        --map ../EMatch/data/EVB_CIRS/calibration/rectify_maps_left.npz \\
        --out output/1_0m_out_left_rect
"""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in',  dest='in_dir',  required=True,
                    help='Folder with frame_*.png + timestamps.txt')
    ap.add_argument('--map', dest='map_npz', required=True,
                    help='rectify_maps_{left,right}.npz')
    ap.add_argument('--out', dest='out_dir', required=True,
                    help='Output folder (will be created)')
    ap.add_argument('--interp', default='linear', choices=['linear', 'cubic', 'nearest'])
    args = ap.parse_args()

    if not os.path.isdir(args.in_dir):
        sys.exit('Input folder not found: ' + args.in_dir)

    maps = np.load(args.map_npz)
    if 'map_x' not in maps.files or 'map_y' not in maps.files:
        sys.exit('Expected map_x and map_y in {} (got {})'
                 .format(args.map_npz, maps.files))
    map_x = maps['map_x']
    map_y = maps['map_y']
    if map_x.shape != map_y.shape:
        sys.exit('map_x {} and map_y {} shape mismatch'.format(map_x.shape, map_y.shape))
    map_h, map_w = map_x.shape
    print('[map]  loaded {}  ({} x {})'.format(args.map_npz, map_w, map_h))

    interp = {'linear': cv2.INTER_LINEAR,
              'cubic':  cv2.INTER_CUBIC,
              'nearest': cv2.INTER_NEAREST}[args.interp]

    os.makedirs(args.out_dir, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(args.in_dir, 'frame_*.png')))
    if not frames:
        sys.exit('No frame_*.png in ' + args.in_dir)
    print('[in]   {} frames'.format(len(frames)))

    for k, fp in enumerate(frames):
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit('Failed to read ' + fp)
        if img.shape[:2] != (map_h, map_w):
            sys.exit('Frame {} is {} but rectify map expects {}x{}'
                     .format(fp, img.shape, map_w, map_h))
        rect = cv2.remap(img, map_x, map_y, interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(fp)), rect)
        if (k + 1) % 50 == 0 or k + 1 == len(frames):
            print('  rectified {}/{}'.format(k + 1, len(frames)))

    src_ts = os.path.join(args.in_dir, 'timestamps.txt')
    if os.path.isfile(src_ts):
        shutil.copy2(src_ts, os.path.join(args.out_dir, 'timestamps.txt'))
        print('[copy] timestamps.txt')
    else:
        print('[warn] no timestamps.txt to copy')

    print('[done] -> ' + args.out_dir)


if __name__ == '__main__':
    main()
