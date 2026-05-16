#!/usr/bin/env python3
"""Pair rectified L/R E2VID frames by timestamp and run RAFT-Stereo.

Mirrors the output layout of `pair_and_stereo_sgbm.py` so the two backends are
drop-in comparable:
    disp_<idx>.npy   float32 disparity in pixels (NaN where invalid)
    disp_<idx>.png   uint16, value = disp * 256  (DSEC convention)
    viz_<idx>.png    color-mapped uint8 visualization
    pairs.csv        idx, l_path, r_path, ts_l, ts_r, dt_ms, mean_disp, pct_valid

Run inside the `raftstereo` conda env. The grayscale E2VID frames are
broadcast to 3 channels before being fed to the network. Frames are padded to
the next multiple of 32 by RAFT-Stereo's InputPadder, then unpadded after
inference.

Example:
    conda activate raftstereo
    python scripts/pair_and_stereo_raft.py \\
        --left  output/1_0m_out_left_rect \\
        --right output/1_0m_out_right_rect \\
        --out   output/1_0m_out_disp_raft \\
        --raft_repo /home/vsjay/ifros_hop_ws/RAFT-Stereo \\
        --ckpt /home/vsjay/ifros_hop_ws/RAFT-Stereo/models/raftstereo-middlebury.pth \\
        --skip_warmup_s 1.0
"""

import argparse
import csv
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

# Local pairing helper (same dir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pairing import pair_by_timestamp  # noqa: E402


def build_model(raft_repo: str, ckpt: str, args: SimpleNamespace, device: str):
    """Mirror the loading pattern from RAFT-Stereo/demo.py."""
    core_dir = os.path.join(raft_repo, 'core')
    if not os.path.isdir(core_dir):
        sys.exit('RAFT-Stereo core dir not found: ' + core_dir)
    # raft_stereo.py uses `from core.update import ...`, so the *repo root*
    # must be on sys.path; the `core/` dir itself is also added so the demo's
    # `from raft_stereo import RAFTStereo` resolves.
    sys.path.insert(0, raft_repo)
    sys.path.insert(0, core_dir)
    from raft_stereo import RAFTStereo  # noqa: E402

    if not os.path.isfile(ckpt):
        sys.exit('Checkpoint not found: ' + ckpt)

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    state = torch.load(ckpt, map_location=device, weights_only=True) \
        if 'weights_only' in torch.load.__code__.co_varnames \
        else torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model = model.module
    model.to(device)
    model.eval()
    return model


def load_gray_as_rgb_tensor(path: str, device: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit('Failed to read ' + path)
    img3 = np.stack([img, img, img], axis=0)            # [3, H, W] uint8
    t = torch.from_numpy(img3).float().unsqueeze(0)     # [1, 3, H, W] in [0, 255]
    return t.to(device)


def disp_to_viz(disp: np.ndarray, max_disp: float) -> np.ndarray:
    vis = disp.copy()
    vis[~np.isfinite(vis)] = 0
    vis = np.clip(vis, 0, max_disp)
    vis_u8 = (vis * (255.0 / max_disp)).astype(np.uint8)
    return cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--left',  required=True)
    ap.add_argument('--right', required=True)
    ap.add_argument('--out',   required=True)
    ap.add_argument('--raft_repo', default='/home/vsjay/ifros_hop_ws/RAFT-Stereo',
                    help='Path to the RAFT-Stereo repo (must contain core/raft_stereo.py).')
    ap.add_argument('--ckpt', default=None,
                    help='RAFT-Stereo checkpoint .pth. Default: <raft_repo>/models/raftstereo-middlebury.pth')
    ap.add_argument('--valid_iters', type=int, default=32,
                    help='Number of GRU update iterations per forward pass. Default 32.')
    ap.add_argument('--mixed_precision', action='store_true',
                    help='Run inference under torch.cuda.amp.autocast for ~2x speedup.')
    # Architecture flags — keep in sync with the checkpoint. Defaults match
    # raftstereo-middlebury.pth / -eth3d.pth / -sceneflow.pth.
    ap.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 128, 128])
    ap.add_argument('--corr_implementation', default='reg',
                    choices=['reg', 'alt', 'reg_cuda', 'alt_cuda'])
    ap.add_argument('--shared_backbone', action='store_true')
    ap.add_argument('--corr_levels', type=int, default=4)
    ap.add_argument('--corr_radius', type=int, default=4)
    ap.add_argument('--n_downsample', type=int, default=2)
    ap.add_argument('--context_norm', default='batch',
                    choices=['group', 'batch', 'instance', 'none'])
    ap.add_argument('--slow_fast_gru', action='store_true')
    ap.add_argument('--n_gru_layers', type=int, default=3)
    # Pairing.
    ap.add_argument('--max_dt_ms',     type=float, default=10.0)
    ap.add_argument('--skip_warmup_s', type=float, default=0.0)
    ap.add_argument('--limit', type=int, default=0)
    # Viz scale.
    ap.add_argument('--viz_max_disp', type=float, default=384.0,
                    help='Disparity value mapped to the top of the TURBO colormap in viz_*.png.')
    # Geometry workaround.
    ap.add_argument('--hflip_inputs', action='store_true',
                    help='Horizontally flip both L and R images before inference and '
                         'flip the disparity back. Use this when your stereo rig has a '
                         'reversed disparity sign (e.g. EVB_CIRS toe-in geometry, where '
                         "ORB matches show x_L < x_R) so a network trained on standard "
                         'left-positive disparity (RAFT-Stereo) can still produce '
                         'sensible output without renaming the L/R cameras.')
    args = ap.parse_args()

    if args.ckpt is None:
        args.ckpt = os.path.join(args.raft_repo, 'models', 'raftstereo-middlebury.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[env]  device = {}, ckpt = {}'.format(device, args.ckpt))

    pairs = pair_by_timestamp(args.left, args.right,
                              max_dt_ms=args.max_dt_ms,
                              skip_warmup_s=args.skip_warmup_s)
    if not pairs:
        sys.exit('No L/R pairs within {} ms tolerance.'.format(args.max_dt_ms))
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print('[pair] {} pairs (max |dt| = {:.2f} ms)'
          .format(len(pairs), max(abs(p.dt_ms) for p in pairs)))

    # Strip non-architecture args before passing to RAFTStereo (it just reads attrs by name,
    # so extras don't hurt — but pass an explicit namespace for clarity).
    model_args = SimpleNamespace(
        hidden_dims=args.hidden_dims,
        corr_implementation=args.corr_implementation,
        shared_backbone=args.shared_backbone,
        corr_levels=args.corr_levels,
        corr_radius=args.corr_radius,
        n_downsample=args.n_downsample,
        context_norm=args.context_norm,
        slow_fast_gru=args.slow_fast_gru,
        n_gru_layers=args.n_gru_layers,
        mixed_precision=args.mixed_precision,
    )
    model = build_model(args.raft_repo, args.ckpt, model_args, device)

    from utils.utils import InputPadder  # noqa: E402  (after sys.path inserted in build_model)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, 'pairs.csv')

    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['idx', 'l_path', 'r_path', 'ts_l', 'ts_r', 'dt_ms',
                     'mean_disp', 'pct_valid'])

        with torch.no_grad():
            for k, p in enumerate(pairs):
                im1 = load_gray_as_rgb_tensor(p.l_path, device)
                im2 = load_gray_as_rgb_tensor(p.r_path, device)

                if args.hflip_inputs:
                    # Flip horizontally and swap L/R: turns reversed-disparity
                    # geometry into a standard left-positive-disparity pair.
                    im1f = torch.flip(im1, dims=[-1])
                    im2f = torch.flip(im2, dims=[-1])
                    im1, im2 = im2f, im1f

                padder = InputPadder(im1.shape, divis_by=32)
                im1p, im2p = padder.pad(im1, im2)

                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        _, flow_up = model(im1p, im2p,
                                           iters=args.valid_iters, test_mode=True)
                else:
                    _, flow_up = model(im1p, im2p,
                                       iters=args.valid_iters, test_mode=True)

                flow_up = padder.unpad(flow_up)             # [1, 1, H, W]
                if args.hflip_inputs:
                    # Undo the H-flip: flow was computed in the mirrored frame.
                    flow_up = torch.flip(flow_up, dims=[-1])
                # RAFT-Stereo outputs negative horizontal flow; positive disparity = -flow_x.
                disp = -flow_up.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

                # Mark non-finite or non-positive disparities as invalid.
                disp_out = disp.copy()
                disp_out[~np.isfinite(disp_out) | (disp_out < 0)] = np.nan

                base = 'disp_{:06d}'.format(k)
                np.save(os.path.join(args.out, base + '.npy'), disp_out)

                disp_u16 = np.zeros_like(disp_out, dtype=np.uint16)
                valid = np.isfinite(disp_out) & (disp_out >= 0)
                disp_u16[valid] = np.clip(disp_out[valid] * 256.0, 0, 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(args.out, base + '.png'), disp_u16)

                cv2.imwrite(os.path.join(args.out, 'viz_{:06d}.png'.format(k)),
                            disp_to_viz(disp_out, args.viz_max_disp))

                vd = disp_out[valid]
                mean_disp = float(vd.mean()) if vd.size else float('nan')
                pct_valid = float(valid.mean() * 100.0)
                wr.writerow([k, p.l_path, p.r_path,
                             '{:.6f}'.format(p.ts_l), '{:.6f}'.format(p.ts_r),
                             '{:.3f}'.format(p.dt_ms),
                             '{:.2f}'.format(mean_disp),
                             '{:.1f}'.format(pct_valid)])

                if (k + 1) % 10 == 0 or k + 1 == len(pairs):
                    print('  [{:>4}/{}] ts={:.3f}s  dt={:+.2f}ms  '
                          'mean_disp={:.1f}px  valid={:.1f}%'
                          .format(k + 1, len(pairs), p.ts_l, p.dt_ms,
                                  mean_disp, pct_valid))

    print('[done] disparities -> {}'.format(args.out))
    print('       summary CSV -> {}'.format(csv_path))


if __name__ == '__main__':
    main()
