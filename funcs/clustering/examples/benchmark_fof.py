#!/usr/bin/env python3
"""
Benchmark Friends-of-Friends (FOF) clustering in PLASMAtools.

Generates synthetic point clouds, warms up JIT, and reports wall-time.

Usage examples:
  python examples/benchmark_fof.py --n 200000 --dims 3 --ll 0.05 --runs 3
  python examples/benchmark_fof.py --n 5_000_000 --tile 256 --ll 0.02 --periodic
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from PLASMAtools.funcs.clustering.operations import ClusteringOperations
from PLASMAtools.funcs.clustering.constants import PERIODIC, NEUMANN


def make_positions(n: int, dims: int, box: float, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.random((n, dims), dtype=np.float32) * box
    return pos


def run_once(n: int, dims: int, ll: float, box: float, periodic: bool,
             precision: str, use_parallel: bool, warmup: bool) -> tuple[float, int]:
    positions = make_positions(n, dims, box, seed=42 if warmup else 123)
    bc = np.full(dims, PERIODIC if periodic else NEUMANN, dtype=np.int32)

    ops = ClusteringOperations(num_of_dims=dims, precision=precision, use_parallel=use_parallel)

    # Explicit box_size ensures no extra shifting and keeps positions in [0, box)
    box_vec = np.array([box] * dims, dtype=np.float32 if precision == 'float32' else np.float64)

    t0 = time.perf_counter()
    labels = ops.friends_of_friends(
        positions=positions,
        linking_length=ll,
        box_size=box_vec,
        boundary_conditions=bc,
        min_cluster_size=1,
    )
    dt = time.perf_counter() - t0
    n_clusters = int(np.max(labels) + 1) if labels.size else 0
    return dt, n_clusters


def main():
    p = argparse.ArgumentParser(description="Benchmark FOF clustering")
    p.add_argument('--n', type=int, default=500_000, help='Number of points')
    p.add_argument('--dims', type=int, default=3, choices=[2, 3], help='Spatial dimensions')
    p.add_argument('--ll', type=float, default=0.05, help='Linking length (absolute units)')
    p.add_argument('--box', type=float, default=1.0, help='Domain size per dimension')
    p.add_argument('--runs', type=int, default=3, help='Timed runs (excludes warmup)')
    p.add_argument('--precision', type=str, default='float32', choices=['float32', 'float64'])
    p.add_argument('--parallel', action='store_true', help='Enable Numba parallel kernels if available')
    p.add_argument('--no-warmup', dest='warmup', action='store_false', help='Skip JIT warmup run')
    p.add_argument('--periodic', action='store_true', help='Use periodic BCs (min-image)')
    args = p.parse_args()

    print(f"FOF benchmark: n={args.n:,}, dims={args.dims}, ll={args.ll}, box={args.box}, precision={args.precision}, parallel={args.parallel}, periodic={args.periodic}")

    # Warmup to exclude JIT compile time from measured runs
    if args.warmup:
        dt_warm, nc_warm = run_once(min(args.n, 50_000), args.dims, args.ll, args.box, args.periodic, args.precision, args.parallel, warmup=True)
        print(f"Warmup: {dt_warm:.3f}s (n={min(args.n, 50_000):,}, clusters≈{nc_warm})")

    times = []
    clusters = None
    for r in range(args.runs):
        dt, nc = run_once(args.n, args.dims, args.ll, args.box, args.periodic, args.precision, args.parallel, warmup=False)
        times.append(dt)
        clusters = nc
        print(f"Run {r+1}/{args.runs}: {dt:.3f}s")

    arr = np.array(times)
    mean = float(arr.mean())
    p95 = float(np.percentile(arr, 95)) if len(arr) > 1 else mean
    throughput_mpps = (args.n / 1e6) / mean

    print("--- Summary ---")
    print(f"Mean: {mean:.3f}s | p95: {p95:.3f}s | throughput: {throughput_mpps:.2f} Mpts/s | clusters≈{clusters}")


if __name__ == '__main__':
    main()

