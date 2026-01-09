#!/usr/bin/env python3
"""
Grid-based clustering demo.

Author:
James R. Beattie.

Usage:
  python examples/main_grid_demo.py --data ../data/info_00100.npz --out clustering_cache_00100 --threshold -1.5
"""

import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cmasher as cmr
import h5py
from celluloid import Camera

from PLASMAtools.funcs.clustering import ClusteringOperations
from PLASMAtools.funcs.derived_vars import DerivedVars as DV
from PLASMAtools.funcs.clustering.constants import PERIODIC, NEUMANN


def load_fields(path):
    data = np.load(path)
    density = data["rho"].astype("float32")
    pressure = data["P"].astype("float32")
    velocity = np.array([data["vx"], data["vy"], data["vz"]]).astype("float32")
    return density, pressure, velocity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to NPZ with rho,P,vx,vy,vz")
    ap.add_argument("--out", default="clustering_cache", help="Output basename (HDF5 cache)")
    ap.add_argument("--threshold", type=float, default=-1.5, help="Log10(T/std(T)) threshold")
    ap.add_argument("--min-size", type=int, default=500, help="Min cluster size (voxels)")
    ap.add_argument("--thickness", type=int, default=2, help="Morphology min thickness (voxels)")
    ap.add_argument("--force", action="store_true", help="Force recluster even if cache exists")
    ap.add_argument("--cubes", action="store_true", help="run cube extraction and plotting")
    ap.add_argument("--cube-size", type=int, default=64, help="Cube side length (voxels, even)")
    ap.add_argument("--save-cubes", action="store_true", help="Save per-cluster cubes and slice plots")
    ap.add_argument("--animate", action="store_true", help="Create clustering slice animation (MP4)")
    ap.add_argument("--anim-out", default="clustering_animation.mp4", help="Animation output filename")
    ap.add_argument("--anim-fps", type=int, default=10, help="Animation frames per second")
    ap.add_argument("--anim-step", type=int, default=1, help="Slice step (every Nth slice)")
    ap.add_argument("--slice-tolerance", type=int, default=2, help="Mark centers within +/- tolerance slices")
    args = ap.parse_args()

    density, pressure, velocity = load_fields(args.data)
    temperature = pressure / density
    clustering = ClusteringOperations(num_of_dims=3, precision='float32')
    cache = args.out + ".h5"

    if os.path.exists(cache) and not args.force:
        print(f"Loading cached clustering results from {cache} ...")
        results = clustering.load_clustering_results(args.out)
    else:
        temp_normalized = np.log10(temperature / np.std(temperature))
        print("Clustering (grid mode)...")
        results = clustering.cluster_3d_field(
            field=temp_normalized,
            threshold=args.threshold,
            threshold_type='greater',
            grid_spacing=1.0,
            morphological_filter=True,
            morph_operation='open',
            min_thickness=args.thickness,
            min_cluster_size=args.min_size,
            separate_overlapping=False,
            method='grid',
            radius_method='effective90',
            connectivity=26,
            boundary_conditions=np.array([PERIODIC, PERIODIC, NEUMANN], dtype=np.int32),
            split_large_clusters=False,
            split_size_factor=10.0,
            split_min_separation_vox=5,
            max_peaks=3,
            max_clusters_to_process=5,
            return_field=True,
        )
        
        clustering.save_clustering_results(args.out, results)

    props = results['properties']
    print(f"Clusters: {props['n_clusters']}")
    if props['n_clusters'] > 0:
        print("Sizes:", props['cluster_sizes'])
        radii = props['cluster_radii']
        print("Mean radius:", float(np.mean(radii)))
        # Report top-3 largest radii
        if len(radii) > 0:
            topk = min(3, len(radii))
            top_idx = np.argsort(radii)[-topk:][::-1]
            print("Top radii:")
            for rank, idx in enumerate(top_idx, 1):
                print(f"  #{rank}: C{idx}  r={radii[idx]:.2f} (size={props['cluster_sizes'][idx]})")

    if 'cluster_field' in results and results['cluster_field'] is not None:
        print("Label grid shape:", results['cluster_field'].shape)
        
    # Optionally extract, save, and/or plot cubes around each detected cluster
    if args.cubes:
        if props['n_clusters'] > 0 and args.save_cubes:
            cubes_h5 = os.path.splitext(args.out)[0] + '_cubes.h5'
            if not os.path.exists(cubes_h5):
                print("Saving cubes via PLASMAtools (grid-native)...")
                clustering.save_cluster_cubes(
                    filename_base=os.path.splitext(args.out)[0],
                    results=results,
                    fields={
                        'temperature': temperature,
                        'density': density,
                        'pressure': pressure,
                        'vx': velocity[0],
                        'vy': velocity[1],
                        'vz': velocity[2],
                    },
                    cube_size=args.cube_size,
                    periodic_axes=(True, True, False),
                    include_derived=True,
                )

            print(f"Loading cubes from {cubes_h5} ...")
            with h5py.File(cubes_h5, 'r') as hf:
                    for i in range(props['n_clusters']):
                        key = f'cluster_{i:02d}'
                        if key not in hf:
                            print(f"Warning: missing {key} in cubes file; skipping")
                            continue
                        grp = hf[key]
                        # Required datasets for plotting; skip if missing
                        try:
                            temp_cube = grp['temperature'][:]
                            dens_cube = grp['density'][:]
                            pres_cube = grp['pressure'][:]
                            vx_cube = grp['vx'][:]
                            vy_cube = grp['vy'][:]
                            vz_cube = grp['vz'][:]
                            mask_cube = grp['cluster_mask'][:]
                            # Optional derived
                            have_derived = all(k in grp for k in ['u_cx','u_cy','u_cz','u_sx','u_sy','u_sz','omega_mag','baro_mag'])
                            if have_derived:
                                u_cx = grp['u_cx'][:]; u_cy = grp['u_cy'][:]; u_cz = grp['u_cz'][:]
                                u_sx = grp['u_sx'][:]; u_sy = grp['u_sy'][:]; u_sz = grp['u_sz'][:]
                                omega_mag = grp['omega_mag'][:]
                                baro_mag = grp['baro_mag'][:]
                        except Exception as e:
                            print(f"  Skipping plot for {key} due to missing data: {e}")
                            continue

                        half_size = temp_cube.shape[0] // 2
                        mid = half_size
                        cx_local = mid; cz_local = mid

                        f, axes = plt.subplots(4, 4, figsize=(18, 16))
                        im00 = axes[0,0].imshow(np.log10(np.maximum(temp_cube[:,:,mid], 1e-20)/np.std(temp_cube)), cmap=cmr.sunburst)
                        axes[0,0].set_title('log10 T (mid-Y)'); axes[0,0].plot(cx_local, cz_local, 'k*', markersize=10)
                        plt.colorbar(im00, ax=axes[0,0], fraction=0.046, pad=0.04)
                        im01 = axes[0,1].imshow(dens_cube[:,:,mid], cmap=cmr.ember)
                        axes[0,1].set_title('density (mid-Y)'); axes[0,1].plot(cx_local, cz_local, 'k*', markersize=10)
                        plt.colorbar(im01, ax=axes[0,1], fraction=0.046, pad=0.04)
                        im02 = axes[0,2].imshow(pres_cube[:,:,mid], cmap=cmr.iceburn)
                        axes[0,2].set_title('pressure (mid-Y)'); axes[0,2].plot(cx_local, cz_local, 'k*', markersize=10)
                        plt.colorbar(im02, ax=axes[0,2], fraction=0.046, pad=0.04)
                        im03 = axes[0,3].imshow(mask_cube[:,:,mid], cmap='binary')
                        axes[0,3].set_title('cluster mask (mid-Y)'); axes[0,3].plot(cx_local, cz_local, 'r*', markersize=10)
                        plt.colorbar(im03, ax=axes[0,3], fraction=0.046, pad=0.04)
                        vmax = np.nanmax(np.abs(vx_cube[:,:,mid])); vmax = vmax if vmax > 0 else 1e-12
                        im10 = axes[1,0].imshow(vx_cube[:,:,mid], cmap='coolwarm', vmin=-vmax, vmax=vmax); axes[1,0].set_title('vx (mid-Y)'); plt.colorbar(im10, ax=axes[1,0], fraction=0.046, pad=0.04)
                        vmax = np.nanmax(np.abs(vy_cube[:,:,mid])); vmax = vmax if vmax > 0 else 1e-12
                        im11 = axes[1,1].imshow(vy_cube[:,:,mid], cmap='coolwarm', vmin=-vmax, vmax=vmax); axes[1,1].set_title('vy (mid-Y)'); plt.colorbar(im11, ax=axes[1,1], fraction=0.046, pad=0.04)
                        vmax = np.nanmax(np.abs(vz_cube[:,:,mid])); vmax = vmax if vmax > 0 else 1e-12
                        im12 = axes[1,2].imshow(vz_cube[:,:,mid], cmap='coolwarm', vmin=-vmax, vmax=vmax); axes[1,2].set_title('vz (mid-Y)'); plt.colorbar(im12, ax=axes[1,2], fraction=0.046, pad=0.04)
                        if have_derived:
                            im13 = axes[1,3].imshow(omega_mag[:,:,mid], cmap=cmr.cosmic); axes[1,3].set_title('|omega| (mid-Y)'); plt.colorbar(im13, ax=axes[1,3], fraction=0.046, pad=0.04)
                            vmax = np.nanmax(np.abs(u_cx[:,:,mid])); vmax = vmax if vmax > 0 else 1e-12
                            im20 = axes[2,0].imshow(u_cx[:,:,mid], cmap='coolwarm', vmin=-vmax, vmax=vmax); axes[2,0].set_title('u_cx (mid-Y)'); plt.colorbar(im20, ax=axes[2,0], fraction=0.046, pad=0.04)
                            vmax = np.nanmax(np.abs(u_cy[:,:,mid])); vmax = vmax if vmax > 0 else 1e-12
                            im21 = axes[2,1].imshow(u_cy[:,:,mid], cmap='coolwarm', vmin=-vmax, vmax=vmax); axes[2,1].set_title('u_cy (mid-Y)'); plt.colorbar(im21, ax=axes[2,1], fraction=0.046, pad=0.04)
                            vmax = np.nanmax(np.abs(u_cz[:,:,mid])); vmax = vmax if vmax > 0 else 1e-12
                            im22 = axes[2,2].imshow(u_cz[:,:,mid], cmap='coolwarm', vmin=-vmax, vmax=vmax); axes[2,2].set_title('u_cz (mid-Y)'); plt.colorbar(im22, ax=axes[2,2], fraction=0.046, pad=0.04)
                            im23 = axes[2,3].imshow(baro_mag[:,:,mid], cmap=cmr.fusion); axes[2,3].set_title('|baro| (mid-Y)'); plt.colorbar(im23, ax=axes[2,3], fraction=0.046, pad=0.04)
                            uc_mag = np.sqrt(u_cx**2 + u_cy**2 + u_cz**2)
                            im33 = axes[3,3].imshow(uc_mag[:,:,mid], cmap='viridis'); axes[3,3].set_title('|u_c| (mid-Y)'); plt.colorbar(im33, ax=axes[3,3], fraction=0.046, pad=0.04)
                            vmax = np.nanmax(np.abs(grp['u_sx'][:,:,mid])); vmax = vmax if vmax > 0 else 1e-12
                            im30 = axes[3,0].imshow(grp['u_sx'][:,:,mid], cmap='coolwarm', vmin=-vmax, vmax=vmax); axes[3,0].set_title('u_sx (mid-Y)'); plt.colorbar(im30, ax=axes[3,0], fraction=0.046, pad=0.04)
                            vmax = np.nanmax(np.abs(grp['u_sy'][:,:,mid])); vmax = vmax if vmax > 0 else 1e-12
                            im31 = axes[3,1].imshow(grp['u_sy'][:,:,mid], cmap='coolwarm', vmin=-vmax, vmax=vmax); axes[3,1].set_title('u_sy (mid-Y)'); plt.colorbar(im31, ax=axes[3,1], fraction=0.046, pad=0.04)
                            vmax = np.nanmax(np.abs(grp['u_sz'][:,:,mid])); vmax = vmax if vmax > 0 else 1e-12
                            im32 = axes[3,2].imshow(grp['u_sz'][:,:,mid], cmap='coolwarm', vmin=-vmax, vmax=vmax); axes[3,2].set_title('u_sz (mid-Y)'); plt.colorbar(im32, ax=axes[3,2], fraction=0.046, pad=0.04)
                        for ax in axes.flat:
                            ax.set_xlabel('x'); ax.set_ylabel('z')
                        plt.suptitle(f'Cluster {i} local cube (loaded)')
                        out_png = f'cluster_{i:02d}_slices.png'
                        plt.savefig(out_png, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"Saved slice visualization to {out_png}")

    # Overview figure: all clusters in a single plot
    if props['n_clusters'] > 0:
        centers = props['cluster_centers']
        radii = props['cluster_radii']
        print(radii)
        temp_norm = np.log10(temperature / np.std(temperature))
        temp_proj = np.max(temp_norm, axis=1)  # project along Y
        f, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(temp_proj.T, cmap=cmr.sunburst, aspect='auto', extent=[0, temp_proj.shape[0], 0, temp_proj.shape[1]])
        sc = ax.scatter(centers[:, 0], centers[:, 2], s=600*radii, c=radii, cmap=cmr.cosmic, alpha=0.4, edgecolors='white', linewidths=1.5)
        for i, c in enumerate(centers):
            ax.text(c[0]+2, c[2]+2, f'C{i}', fontsize=10, color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log10 T proj')
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='radius')
        ax.set_xlim(0, temperature.shape[0])
        ax.set_ylim(0, temperature.shape[2])
        plt.savefig('clusters_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('Saved clusters overview to clusters_overview.png')

    # Optional: create clustering animation across slices (Y index)
    if args.animate and 'cluster_field' in results and results['cluster_field'] is not None:
        cluster_field = results['cluster_field']
        mask = results['mask']
        temp_norm = np.log10(temperature / np.std(temperature))
        ny = mask.shape[1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        camera = Camera(fig)
        # Stable label colors across frames
        k = int(props['n_clusters']) if props['n_clusters'] > 0 else 1
        base = plt.cm.get_cmap('tab20', 20)
        color_list = [(0.8, 0.8, 0.8, 1.0)] + [base(i % base.N) for i in range(k)]
        stable_cmap = mcolors.ListedColormap(color_list, name='stable_clusters')
        boundaries = np.arange(-1.5, k - 0.5 + 1, 1.0)
        stable_norm = mcolors.BoundaryNorm(boundaries, stable_cmap.N)
    
        for slice_idx in range(0, ny, max(1, args.anim_step)):
            axes[0].imshow(mask[:, slice_idx, :].T, cmap='binary', aspect='auto')
            axes[1].imshow(temp_norm[:, slice_idx, :].T, cmap='coolwarm', aspect='auto')

            cl_slice = cluster_field[:, slice_idx, :].T
            axes[2].imshow(cl_slice, cmap=stable_cmap, norm=stable_norm, aspect='auto')

            # Mark centers near this slice
            if props['n_clusters'] > 0:
                radii_anim = props['cluster_radii']
                for i, c in enumerate(props['cluster_centers']):
                    if abs(int(round(c[1])) - slice_idx) <= args.slice_tolerance:
                        axes[2].plot(c[0], c[2], 'w*', markersize=12, markeredgecolor='black', markeredgewidth=1)
                        axes[2].text(c[0], c[2]+2, f'C{i} (r={radii_anim[i]:.1f})', color='white', fontsize=9, ha='center')
            camera.snap()

        anim = camera.animate()
        anim.save("clustering_animation.mp4", writer='ffmpeg', fps=args.anim_fps, dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
