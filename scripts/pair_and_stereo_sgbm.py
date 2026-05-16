#!/usr/bin/env python3
"""Pair rectified L/R E2VID frames by timestamp and run OpenCV StereoSGBM.

Inputs are two folders produced by `rectify_e2vid_frames.py` (each must contain
frame_*.png + timestamps.txt). Outputs land in --out:

    disp_<idx>.npy   float32 disparity in pixels (NaN where invalid)
    disp_<idx>.png   uint16, value = disp * 256 (DSEC convention; reload with
                     `cv2.imread(..., cv2.IMREAD_UNCHANGED).astype(float)/256.0`)
    viz_<idx>.png    color-mapped uint8 visualization for quick inspection
    pairs.csv        idx, l_path, r_path, ts_l, ts_r, dt_ms

For 1280x720 / fx≈1694 / baseline≈0.175 m, expected disparity at 1.0 m is ~296 px,
so the default --max_disp is 320 (multiple of 16). Bump higher for closer scenes.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

# Local import (script lives in rpg_e2vid/scripts/, so add this dir to path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pairing import pair_by_timestamp  # noqa: E402


def build_sgbm(min_disp: int, num_disp: int, block_size: int,
               uniqueness_ratio: int, speckle_window: int, speckle_range: int,
               disp12_max_diff: int):
    if num_disp % 16 != 0 or num_disp <= 0:
        raise ValueError('--num_disp must be a positive multiple of 16 (got {})'.format(num_disp))
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError('--block_size must be odd and >=3 (got {})'.format(block_size))
    cn = 1  # grayscale
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * cn * block_size * block_size,
        P2=32 * cn * block_size * block_size,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window,
        speckleRange=speckle_range,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def disp_to_viz(disp: np.ndarray, viz_max: float) -> np.ndarray:
    """Color-map |disparity| for a quick eyeball check.

    Uses the absolute value so the same colormap range works for both standard
    (positive) and reversed (negative) disparity sign conventions.
    """
    vis = np.abs(disp.copy())
    vis[~np.isfinite(vis)] = 0
    vis = np.clip(vis, 0, viz_max)
    vis_u8 = (vis * (255.0 / max(viz_max, 1e-6))).astype(np.uint8)
    return cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--left',  required=True, help='Folder of rectified left frames')
    ap.add_argument('--right', required=True, help='Folder of rectified right frames')
    ap.add_argument('--out',   required=True, help='Output folder (created)')
    ap.add_argument('--min_disp',     type=int,   default=0,
                    help='SGBM minDisparity. Use a negative value (e.g. -512) to search '
                         'reversed-sign disparities — needed for the EVB_CIRS toe-in rig '
                         'where rectified-right features sit at higher x than rectified-left. '
                         'Default 0 (standard convention).')
    ap.add_argument('--max_disp',     type=int,   default=320,
                    help='Alias for --num_disp (kept for backward compat). '
                         'SGBM numDisparities (multiple of 16). Default 320.')
    ap.add_argument('--num_disp',     type=int,   default=None,
                    help='SGBM numDisparities (multiple of 16). If unset, falls back to --max_disp.')
    ap.add_argument('--block_size',   type=int,   default=5,  help='Odd, >=3. Default 5.')
    ap.add_argument('--uniqueness_ratio', type=int, default=10,
                    help='SGBM uniquenessRatio. Lower = more matches kept (noisier). Default 10.')
    ap.add_argument('--speckle_window', type=int, default=100,
                    help='SGBM speckleWindowSize. 0 disables speckle filter. Default 100.')
    ap.add_argument('--speckle_range', type=int, default=2,
                    help='SGBM speckleRange. Default 2.')
    ap.add_argument('--disp12_max_diff', type=int, default=1,
                    help='SGBM L/R consistency check. -1 disables. Default 1.')
    ap.add_argument('--pre_blur', type=float, default=0.0,
                    help='Optional Gaussian sigma applied to both frames before SGBM '
                         '(useful to suppress E2VID texture noise). 0 disables.')
    ap.add_argument('--max_dt_ms',    type=float, default=10.0,
                    help='Max |Δt| between paired L and R frames, in milliseconds.')
    ap.add_argument('--skip_warmup_s', type=float, default=0.0,
                    help='Drop left frames whose ts < this (FireNet warm-up).')
    ap.add_argument('--limit', type=int, default=0,
                    help='If >0, only process the first N pairs (for quick checks).')
    args = ap.parse_args()

    pairs = pair_by_timestamp(args.left, args.right,
                              max_dt_ms=args.max_dt_ms,
                              skip_warmup_s=args.skip_warmup_s)
    if not pairs:
        sys.exit('No L/R pairs within {} ms tolerance.'.format(args.max_dt_ms))
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print('[pair] {} pairs (max |dt| = {:.2f} ms)'
          .format(len(pairs), max(abs(p.dt_ms) for p in pairs)))

    os.makedirs(args.out, exist_ok=True)
    num_disp = args.num_disp if args.num_disp is not None else args.max_disp
    sgbm = build_sgbm(args.min_disp, num_disp, args.block_size,
                      args.uniqueness_ratio, args.speckle_window,
                      args.speckle_range, args.disp12_max_diff)
    print('[sgbm] search range = [{}, {}) px'.format(args.min_disp, args.min_disp + num_disp))

    csv_path = os.path.join(args.out, 'pairs.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['idx', 'l_path', 'r_path', 'ts_l', 'ts_r', 'dt_ms',
                     'mean_disp_valid', 'pct_valid'])

        for k, p in enumerate(pairs):
            L = cv2.imread(p.l_path, cv2.IMREAD_GRAYSCALE)
            R = cv2.imread(p.r_path, cv2.IMREAD_GRAYSCALE)
            if L is None or R is None:
                sys.exit('Failed to read pair: {} | {}'.format(p.l_path, p.r_path))
            if L.shape != R.shape:
                sys.exit('Shape mismatch on pair {}: L {} vs R {}'
                         .format(k, L.shape, R.shape))

            if args.pre_blur > 0:
                L = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=args.pre_blur)
                R = cv2.GaussianBlur(R, ksize=(0, 0), sigmaX=args.pre_blur)

            # SGBM returns int16 disparity * 16; pixels below the search range
            # (i.e., invalid) take the value (minDisparity - 1) * 16.
            raw = sgbm.compute(L, R)
            disp = raw.astype(np.float32) / 16.0
            invalid = disp < args.min_disp     # below search range = invalid
            disp[invalid] = np.nan

            base = 'disp_{:06d}'.format(k)
            np.save(os.path.join(args.out, base + '.npy'), disp)

            # 16-bit PNG: encode |disp| × 256 (DSEC-compatible for positive
            # disparities; for negative-sign rigs, magnitude is preserved here
            # — sign lives in the .npy).
            disp_u16 = np.zeros_like(disp, dtype=np.uint16)
            valid = np.isfinite(disp)
            disp_u16[valid] = np.clip(np.abs(disp[valid]) * 256.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(args.out, base + '.png'), disp_u16)

            viz_max = max(abs(args.min_disp), abs(args.min_disp + num_disp))
            cv2.imwrite(os.path.join(args.out, 'viz_{:06d}.png'.format(k)),
                        disp_to_viz(disp, viz_max))

            valid_disp = disp[valid]
            mean_disp = float(valid_disp.mean()) if valid_disp.size else float('nan')
            pct_valid = float(valid.mean() * 100.0)
            wr.writerow([k, p.l_path, p.r_path,
                         '{:.6f}'.format(p.ts_l), '{:.6f}'.format(p.ts_r),
                         '{:.3f}'.format(p.dt_ms),
                         '{:.2f}'.format(mean_disp),
                         '{:.1f}'.format(pct_valid)])

            if (k + 1) % 20 == 0 or k + 1 == len(pairs):
                print('  [{:>4}/{}] ts={:.3f}s  dt={:+.2f}ms  '
                      'mean_disp={:.1f}px  valid={:.1f}%'
                      .format(k + 1, len(pairs), p.ts_l, p.dt_ms,
                              mean_disp, pct_valid))

    print('[done] disparities -> {}'.format(args.out))
    print('       summary CSV -> {}'.format(csv_path))


if __name__ == '__main__':
    main()
