#!/usr/bin/env python3
"""
Convert an EVB_CIRS HDF5 recording (left camera only) into the input format
expected by rpg_e2vid (`run_reconstruction.py`).

EVB_CIRS HDF5 layout (see EMatch/datasets/EVB_CIRS/CustomSequence_ematch.py):
    evk4_hd/left/events  - [N, 4] float64, columns = (x, y, t, polarity)
    evk4_hd/right/events - [N, 4] float64, columns = (x, y, t, polarity)
  - x in [0, width-1], y in [0, height-1]   (EVK4 HD: 1280 x 720)
  - t in microseconds (auto-detected from range; can be unsorted with glitch
    samples that have very negative timestamps)
  - polarity in {-1, +1}

rpg_e2vid input format (.txt or .zip; see utils/event_readers.py):
    line 1     : "<width> <height>"
    line 2..N  : "<t_seconds> <x> <y> <pol>"   pol in {0, 1}
    Events must be sorted by ascending timestamp.

Usage:
    python scripts/convert_evb_cirs_hdf5.py \\
        -i ../EMatch/data/EVB_CIRS/hdf5/0_5m_indoor.hdf5 \\
        -o data/0_5m_indoor.zip
"""

import argparse
import io
import os
import sys
import zipfile

import h5py
import numpy as np


def detect_ts_scale(t_first, t_last):
    """Infer multiplier that converts source timestamps to seconds.

    Mirrors the heuristic used in EMatch/datasets/EVB_CIRS/CustomSequence_ematch.py.
    """
    delta = float(t_last - t_first)
    if delta > 1e8:
        return 1e-9, 'ns'
    if delta > 1e5:
        return 1e-6, 'us'
    if delta > 100:
        return 1e-3, 'ms'
    return 1.0, 's'


def write_header(stream, width, height):
    stream.write('{} {}\n'.format(width, height).encode('ascii'))


def format_chunk(events_t_s, events_x, events_y, events_p):
    """Format one chunk of events as ASCII bytes ready for stream.write()."""
    # numpy >= 1.14 supports per-column formatting via savetxt; build via join for speed.
    # Each row: "%.9f %d %d %d\n"
    lines = np.char.add(
        np.char.add(
            np.char.add(
                np.char.add(
                    np.char.mod('%.9f', events_t_s),
                    ' ',
                ),
                np.char.mod('%d', events_x),
            ),
            np.char.add(' ', np.char.mod('%d', events_y)),
        ),
        np.char.add(' ', np.char.mod('%d', events_p)),
    )
    return ('\n'.join(lines.tolist()) + '\n').encode('ascii')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Path to EVB_CIRS HDF5 file '
                             '(e.g. EMatch/data/EVB_CIRS/hdf5/0_5m_indoor.hdf5)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output path. Use .zip for a zipped events.txt '
                             '(matches rpg_e2vid example datasets) or .txt for plain text.')
    parser.add_argument('--event_path', default='evk4_hd/left/events',
                        help='HDF5 dataset path for events. '
                             'Default uses the left EVK4 HD camera.')
    parser.add_argument('--width',  type=int, default=1280,
                        help='Sensor width  (default: 1280, EVK4 HD)')
    parser.add_argument('--height', type=int, default=720,
                        help='Sensor height (default: 720,  EVK4 HD)')
    parser.add_argument('--ts_unit', choices=['auto', 's', 'ms', 'us', 'ns'],
                        default='auto',
                        help='Unit of timestamps in the HDF5. "auto" infers from range.')
    parser.add_argument('--keep_negative_ts', action='store_true',
                        help='Keep events with negative source timestamps. '
                             'By default these (likely glitches in the EVK4 stream) are dropped.')
    parser.add_argument('--no_zero_start', action='store_true',
                        help='Do not shift the first timestamp to 0; keep absolute seconds.')
    parser.add_argument('--t_start', type=float, default=None,
                        help='Optional crop start time, seconds (after unit conversion '
                             'and optional zero-shift).')
    parser.add_argument('--t_end',   type=float, default=None,
                        help='Optional crop end   time, seconds.')
    parser.add_argument('--chunk', type=int, default=2_000_000,
                        help='Events per write chunk (controls peak memory of formatting).')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit('Input HDF5 not found: {}'.format(args.input))

    out_ext = os.path.splitext(args.output)[1].lower()
    if out_ext not in ('.zip', '.txt'):
        sys.exit('Output extension must be .zip or .txt (got "{}")'.format(out_ext))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.input, 'r') as f:
        if args.event_path not in f:
            sys.exit('HDF5 path "{}" not found. Top-level keys: {}'
                     .format(args.event_path, list(f.keys())))
        ev = f[args.event_path]
        if ev.ndim != 2 or ev.shape[1] != 4:
            sys.exit('Expected events of shape [N, 4], got {}'.format(ev.shape))

        n_total = ev.shape[0]
        print('[load] {} : {} events of shape {}'.format(args.input, n_total, ev.shape))

        # Load whole array. Source is float64 -> ~16 bytes/event * cols.
        # Cast columns to compact dtypes for the rest of the pipeline.
        raw = ev[:]  # [N, 4] float64
        x = raw[:, 0].astype(np.int32, copy=False)
        y = raw[:, 1].astype(np.int32, copy=False)
        t_src = raw[:, 2]                     # keep float64 for safe scaling
        p = raw[:, 3].astype(np.int8, copy=False)
        del raw

    # --- timestamp unit ---
    if args.ts_unit == 'auto':
        ts_scale, unit = detect_ts_scale(np.min(t_src), np.max(t_src))
    else:
        ts_scale = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}[args.ts_unit]
        unit = args.ts_unit
    print('[ts]   detected unit = {}  (scale = {:g})'.format(unit, ts_scale))

    # --- drop glitch samples with negative source timestamps ---
    if not args.keep_negative_ts:
        valid = t_src >= 0
        n_drop = int((~valid).sum())
        if n_drop:
            print('[drop] {} events with negative source timestamps'.format(n_drop))
            x = x[valid]; y = y[valid]; t_src = t_src[valid]; p = p[valid]

    if t_src.size == 0:
        sys.exit('No events left after filtering.')

    # --- convert to seconds and sort ---
    t_s = t_src * ts_scale
    del t_src

    if not np.all(np.diff(t_s) >= 0):
        print('[sort] timestamps not monotonic, sorting...')
        order = np.argsort(t_s, kind='stable')
        t_s = t_s[order]; x = x[order]; y = y[order]; p = p[order]
        del order

    if not args.no_zero_start:
        t0 = float(t_s[0])
        if t0 != 0.0:
            print('[shift] subtracting first timestamp {:.9f}s'.format(t0))
            t_s = t_s - t0

    # --- optional time crop ---
    if args.t_start is not None or args.t_end is not None:
        lo = args.t_start if args.t_start is not None else -np.inf
        hi = args.t_end   if args.t_end   is not None else  np.inf
        m = (t_s >= lo) & (t_s <= hi)
        n_kept = int(m.sum())
        print('[crop] [{}s, {}s] -> {} / {} events'
              .format(lo, hi, n_kept, t_s.size))
        x = x[m]; y = y[m]; t_s = t_s[m]; p = p[m]

    # --- normalize polarity to {0, 1} ---
    uniq = np.unique(p)
    if set(uniq.tolist()).issubset({-1, 1}):
        p_out = ((p + 1) // 2).astype(np.int8, copy=False)         # -1 -> 0,  +1 -> 1
    elif set(uniq.tolist()).issubset({0, 1}):
        p_out = p.astype(np.int8, copy=False)
    else:
        sys.exit('Unexpected polarity values: {}'.format(uniq))

    # --- bounds sanity ---
    if x.size:
        if x.min() < 0 or x.max() >= args.width or y.min() < 0 or y.max() >= args.height:
            print('[warn] event coords out of [0,{})x[0,{}); '
                  'x in [{}, {}], y in [{}, {}]'.format(
                      args.width, args.height, x.min(), x.max(), y.min(), y.max()))

    n_out = t_s.size
    print('[write] {} events -> {}'.format(n_out, args.output))

    # --- write ---
    if out_ext == '.zip':
        inner_name = os.path.splitext(os.path.basename(args.output))[0] + '.txt'
        zf = zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED)
        # ZipFile.open in 'w' mode requires force_zip64 for >4 GB members.
        with zf.open(inner_name, 'w', force_zip64=True) as out:
            write_header(out, args.width, args.height)
            for i in range(0, n_out, args.chunk):
                j = min(i + args.chunk, n_out)
                out.write(format_chunk(t_s[i:j], x[i:j], y[i:j], p_out[i:j]))
                if (i // args.chunk) % 10 == 0:
                    print('  wrote {}/{} events'.format(j, n_out))
        zf.close()
    else:
        with open(args.output, 'wb') as out:
            write_header(out, args.width, args.height)
            for i in range(0, n_out, args.chunk):
                j = min(i + args.chunk, n_out)
                out.write(format_chunk(t_s[i:j], x[i:j], y[i:j], p_out[i:j]))
                if (i // args.chunk) % 10 == 0:
                    print('  wrote {}/{} events'.format(j, n_out))

    print('[done] wrote {}'.format(args.output))


if __name__ == '__main__':
    main()
