# Event → Frame → Stereo disparity pipeline

How to turn an EVB_CIRS HDF5 recording into a per-frame disparity map by:
**(1)** reconstructing intensity images from the events with FireNet/E2VID,
**(2)** stereo-rectifying them, and
**(3)** running OpenCV SGBM as a frame-based stereo matcher.

All commands are run from `/home/vsjay/ifros_hop_ws/rpg_e2vid/` inside the
`e2vid` conda env.

---

## 0. Prerequisites (one-time)

```bash
cd /home/vsjay/ifros_hop_ws/rpg_e2vid
conda activate e2vid                          # PyTorch 2.5 + CUDA + OpenCV 4.13
ls pretrained/firenet_1000.pth.tar            # FireNet weights, ships with repo
ls ../EMatch/data/EVB_CIRS/calibration/rectify_maps_left.npz \
   ../EMatch/data/EVB_CIRS/calibration/rectify_maps_right.npz   # rectify maps
```

If the rectify maps are missing, regenerate them by running
`EMatch/datasets/EVB_CIRS/calibration/StereoCalibration` once
(see `EMatch/data/EVB_CIRS/rectification_test/load_voxel_and_rectify.py`
for the call pattern).

Pick a sequence and export it as a shell variable so the rest of the runbook
is copy-pasteable:

```bash
SEQ=1_0m_out                                                # logical name
H5=../EMatch/data/EVB_CIRS/hdf5/${SEQ}.hdf5                 # source HDF5
```

---

## 1. HDF5 → E2VID input zips (left and right)

`scripts/convert_evb_cirs_hdf5.py` reads `evk4_hd/{left,right}/events`
out of the HDF5, drops glitch events with negative timestamps, sorts by time,
remaps polarity to `{0,1}`, and writes the zipped ASCII format that
`run_reconstruction.py` expects.

```bash
mkdir -p data
python scripts/convert_evb_cirs_hdf5.py \
    -i $H5 --event_path evk4_hd/left/events  -o data/${SEQ}_left.zip
python scripts/convert_evb_cirs_hdf5.py \
    -i $H5 --event_path evk4_hd/right/events -o data/${SEQ}_right.zip
```

Expected runtime: ~2 min per side for ~90 M events. Output: ~450 MB zip
per side.

---

## 2. FireNet reconstruction (left and right)

`--fixed_duration -T 50` produces one frame every 50 ms (20 Hz) on **both**
sides, so left and right share an identical timestamp grid — pairing in
step 4 becomes trivial. `--auto_hdr` keeps the intensity range consistent
across frames; `--use_gpu` runs FireNet on CUDA.

```bash
for SIDE in left right; do
  python run_reconstruction.py \
      -c pretrained/firenet_1000.pth.tar \
      -i data/${SEQ}_${SIDE}.zip \
      --fixed_duration -T 50 \
      --auto_hdr \
      --output_folder output --dataset_name ${SEQ}_${SIDE} \
      --use_gpu
done
```

Outputs (per side): `output/${SEQ}_${SIDE}/frame_XXXXXXXXXX.png` (1280×720
grayscale uint8) and `timestamps.txt` (one float-seconds value per frame).
The `XXXXXXXXXX` digits in the filename are a cumulative event count, **not**
a timestamp — always rely on `timestamps.txt` for time.

Expected runtime: ~2.5 min per side on a recent GPU.

---

## 3. Stereo rectification

`scripts/rectify_e2vid_frames.py` applies the precomputed
`map_x` / `map_y` arrays from `rectify_maps_{left,right}.npz` via
`cv2.remap`, and copies `timestamps.txt` through unchanged.

```bash
python scripts/rectify_e2vid_frames.py \
    --in  output/${SEQ}_left \
    --map ../EMatch/data/EVB_CIRS/calibration/rectify_maps_left.npz \
    --out output/${SEQ}_left_rect

python scripts/rectify_e2vid_frames.py \
    --in  output/${SEQ}_right \
    --map ../EMatch/data/EVB_CIRS/calibration/rectify_maps_right.npz \
    --out output/${SEQ}_right_rect
```

Expected runtime: ~30 s per side.

Sanity check: open `output/${SEQ}_left_rect/frame_*.png` and the matching
right frame side-by-side. Common features should fall on the same horizontal
row. If they don't, the rectification maps are stale and SGBM in step 4 will
return mostly garbage.

---

## 4. Pair frames + run SGBM

`scripts/pair_and_stereo_sgbm.py` reads both `timestamps.txt` files, pairs
each left frame with the nearest right frame within `--max_dt_ms`, runs
`cv2.StereoSGBM`, and writes per-pair disparities + a colorised viz +
`pairs.csv` (one row per pair: filenames, timestamps, dt, mean disparity,
% valid pixels).

```bash
python scripts/pair_and_stereo_sgbm.py \
    --left  output/${SEQ}_left_rect \
    --right output/${SEQ}_right_rect \
    --out   output/${SEQ}_disp_sgbm \
    --max_disp 384 \
    --block_size 9 \
    --uniqueness_ratio 5 \
    --pre_blur 1.0 \
    --speckle_window 50 \
    --skip_warmup_s 1.0
```

Outputs in `output/${SEQ}_disp_sgbm/`:

| file | format | meaning |
| --- | --- | --- |
| `disp_<idx>.npy` | float32 H×W | disparity in pixels, NaN where invalid |
| `disp_<idx>.png` | uint16 H×W | DSEC convention: value = `disp × 256`. Reload with `cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(float)/256.0` |
| `viz_<idx>.png`  | uint8 BGR  | colour-mapped quick look (TURBO, scaled to `--max_disp`) |
| `pairs.csv`      | text       | pair manifest with mean disp + % valid per frame |

Expected runtime: ~3 min for ~300 pairs at full 1280×720 / `max_disp=384`.

Convert the colour viz frames to a video for quick eyeballing:

```bash
( cd output/${SEQ}_disp_sgbm && \
  ffmpeg -y -framerate 20 -pattern_type glob -i 'viz_*.png' \
         -c:v libx264 -pix_fmt yuv420p -crf 22 \
         ../${SEQ}_disp_sgbm.mp4 )
```

---

## 5. Tuning the SGBM step

Defaults above were tuned for `1_0m_out`. For other sequences, the knobs
worth touching first:

| flag | typical range | when to change |
| --- | --- | --- |
| `--min_disp` | 0 (default) or negative (e.g. -512) | Use a negative value to search reversed-sign disparities (EVB_CIRS toe-in: features in `*_right_rect` sit at *higher* x than `*_left_rect`). Pair with `--num_disp` so the search range `[min_disp, min_disp + num_disp)` covers your scene's true range. |
| `--num_disp` (alias `--max_disp`) | 192 / 320 / 384 / 512 (×16) | Width of the search range. With `fx ≈ 1694`, `B ≈ 0.175 m`: 1.5 m → \|disp\| 198 px, 1.0 m → 296 px, 0.5 m → 593 px. If too small, closest objects fall outside the range. |
| `--block_size` | 5 / 7 / 9 / 11 (odd) | Bigger = smoother + denser, less fine detail. |
| `--uniqueness_ratio` | 5–15 | Lower keeps weaker matches (more coverage, more noise). |
| `--pre_blur` | 0.0–2.0 | Gaussian σ on both frames before SGBM; suppresses E2VID texture noise. |
| `--speckle_window` | 0–200 | 0 disables the speckle filter (more coverage). |
| `--max_dt_ms` | 1–20 | Pair tolerance. With `-T 50` reconstructions dt is < 0.01 ms; loosen only if frames don't pair. |
| `--skip_warmup_s` | 0.5–2.0 | Drops noisy frames from the FireNet recurrent state warm-up. |
| `--limit N` | 5–20 | Only process the first N pairs while iterating on parameters. |

### EVB_CIRS toe-in: signed-disparity SGBM run

The two EVK4-HD cameras have a non-trivial toe-in angle, so after rectification
the disparity sign is *reversed* relative to a standard parallel rig (ORB
matches give median `x_L − x_R ≈ −500 px` on `1_0m_out`). To match the actual
sign convention of this rig, run SGBM with a negative search range:

```bash
python scripts/pair_and_stereo_sgbm.py \
    --left  output/${SEQ}_left_rect \
    --right output/${SEQ}_right_rect \
    --out   output/${SEQ}_disp_sgbm_signed \
    --min_disp -512 --num_disp 512 \
    --block_size 9 --uniqueness_ratio 5 --pre_blur 1.0 \
    --speckle_window 50 --skip_warmup_s 1.0
```

In this mode the `disp_*.npy` values are negative (e.g. mean ≈ −430 px on
`1_0m_out`); the `disp_*.png` files store `|disp| × 256` (DSEC convention is
unsigned, so magnitude is what gets encoded — the sign is preserved only in
the `.npy`); `viz_*.png` is colour-mapped on `|disp|` so the same colormap
works for either sign. Coverage on `1_0m_out` rises from ~20 % (positive
search, mostly noise matches) to ~30 % with the correct negative search.

This is the **flip-free** path through the pipeline — no `--hflip_inputs`,
no L/R swap, no rectification regeneration needed. RAFT-Stereo doesn't have
an equivalent knob (its correlation lookup is hard-wired to one direction),
so for the deep backend you still need `--hflip_inputs` (see section 8).

Quick coverage-vs-noise rule of thumb on E2VID frames:
**raise `--pre_blur` and lower `--uniqueness_ratio`** for more dense maps;
**lower `--pre_blur` and raise `--block_size`** for cleaner edges.

---

## 6. Sanity checks

1. **Pairing aligned?** After step 4, look at the `dt_ms` column of
   `pairs.csv` — it should be ~0 across all rows (≤ 0.01 ms with the 50 ms
   reconstruction grid).
2. **Disparity in plausible range?** Median disparity should land near
   `fx · B / Z_typical` for the scene. For `1_0m_out`: median ≈ 165 px →
   ≈ 1.8 m (table-top depth), p95 ≈ 382 px → ≈ 0.78 m (closest objects).
3. **No clipping at the upper bound?** If the disparity histogram piles up
   right at `max_disp − 1`, increase `--max_disp` (next multiple of 16).
4. **Reload .png correctly:**
   ```python
   import cv2, numpy as np
   disp = cv2.imread('disp_000150.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
   depth_m = 1694.55 * 0.17459 / np.where(disp > 0, disp, np.nan)
   ```

---

## 7. Disk usage

For a ~16 s sequence at 20 Hz, the full pipeline produces ~1.2 GB of output
under `output/${SEQ}_disp_sgbm/` — most of it the float32 `.npy` files
(~3.7 MB each × ~300 frames). The 16-bit PNGs round-trip the same
disparities to 1/256 px precision, so once you've validated the run you can
free the `.npy` files:

```bash
rm output/${SEQ}_disp_sgbm/disp_*.npy        # keep only .png + viz + csv
```

---

## 8. RAFT-Stereo (deep) backend

`scripts/pair_and_stereo_raft.py` reuses the same pairing helper and writes
the same `disp_*.npy` / `disp_*.png` (16-bit, ×256) / `viz_*.png` /
`pairs.csv` outputs as the SGBM driver, so the two backends are drop-in
comparable. Run inside the `raftstereo` conda env (PyTorch + CUDA + cv2 are
all there) using the prebuilt RAFT-Stereo repo at
`/home/vsjay/ifros_hop_ws/RAFT-Stereo/` and its `models/raftstereo-middlebury.pth`
checkpoint:

```bash
conda activate raftstereo
python scripts/pair_and_stereo_raft.py \
    --left  output/${SEQ}_left_rect \
    --right output/${SEQ}_right_rect \
    --out   output/${SEQ}_disp_raft \
    --raft_repo /home/vsjay/ifros_hop_ws/RAFT-Stereo \
    --ckpt /home/vsjay/ifros_hop_ws/RAFT-Stereo/models/raftstereo-middlebury.pth \
    --skip_warmup_s 1.0 \
    --mixed_precision \
    --hflip_inputs \
    --viz_max_disp 600
```

Expected runtime: ~6–8 min for ~300 pairs at full 1280×720 with
`--valid_iters 32` on a recent GPU. Output: ~2–3 GB (mostly the float `.npy`
files; same cleanup advice as in section 7 applies).

### `--hflip_inputs` and the EVB_CIRS toe-in

The two EVK4-HD cameras in this rig have a **toe-in angle** (cameras
converge on a target). After `cv2.stereoRectify`, the rectified frames have
a *reversed* disparity sign relative to a standard parallel rig — ORB matches
between `*_left_rect/` and `*_right_rect/` give a *negative* median
`x_L − x_R` (~−500 px on `1_0m_out`). RAFT-Stereo (trained on standard
left-positive-disparity data) hallucinates a near-uniform output if you
feed it the rectified frames as-is.

`--hflip_inputs` works around this **without renaming the cameras**: it
horizontally flips both L and R images before inference, which makes the
pair look "standard" to the network, then flips the resulting disparity
back. SGBM doesn't need this trick because it accepts a configurable
disparity range — see the SGBM section's tuning notes if you want to
experiment with `--minDisparity` < 0 there too.

Other notable RAFT-Stereo flags (defaults match the middlebury / eth3d /
sceneflow checkpoints; the `iraftstereo_rvc.pth` and `raftstereo-realtime.pth`
variants need different `--corr_implementation`, `--shared_backbone`,
`--n_downsample`, `--n_gru_layers` settings — see their README):

| flag | default | what to change |
| --- | --- | --- |
| `--ckpt` | `models/raftstereo-middlebury.pth` | swap in `-eth3d.pth` for outdoor scenes, `-sceneflow.pth` for synthetic |
| `--valid_iters` | 32 | drop to 8–16 for ~2× speed at modest accuracy cost |
| `--mixed_precision` | off | on → ~2× faster, fits on smaller GPUs |
| `--viz_max_disp` | 384 | raise so the colormap doesn't clip on close objects |

Note: RAFT-Stereo is a regression network — every pixel gets a disparity
(`pct_valid` is always 100 %). Treat the value in textureless or border
regions with skepticism. To get a confidence/validity mask, run a separate
left-right consistency check (run the model a second time with L and R
swapped, then compare disparities at warped positions).

---

## File map

| path | role |
| --- | --- |
| `scripts/convert_evb_cirs_hdf5.py` | HDF5 → E2VID `.zip` (existing) |
| `run_reconstruction.py` + `pretrained/firenet_1000.pth.tar` | event → frame (existing) |
| `scripts/rectify_e2vid_frames.py` | apply `cv2.remap` rectification (new) |
| `scripts/_pairing.py` | timestamp-based L/R pair list (new) |
| `scripts/pair_and_stereo_sgbm.py` | SGBM driver writing disparity outputs (new) |
| `scripts/pair_and_stereo_raft.py` | RAFT-Stereo driver (new) |
| `/home/vsjay/ifros_hop_ws/RAFT-Stereo/` | RAFT-Stereo repo + 5 pretrained checkpoints |
| `../EMatch/data/EVB_CIRS/calibration/rectify_maps_{left,right}.npz` | precomputed rectification maps |
| `../EMatch/data/EVB_CIRS/calibration/camchain-cirs_evb.yaml` | intrinsics + extrinsics (`fx ≈ 1694.55`, baseline ≈ 0.175 m) |
