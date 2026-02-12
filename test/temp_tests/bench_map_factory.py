#!/usr/bin/env python
# coding=utf-8
"""Performance benchmark for MapFactory.prepare_maps with Unity URP config.

Creates realistic TGA textures (mimicking c130j_fuselage style) and measures
processing time for the Unity URP pipeline. Run with:

    python test/temp_tests/bench_map_factory.py
"""
import os
import sys
import time
import shutil
import tempfile

# Ensure pythontk is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pythontk import ImgUtils
from pythontk.img_utils.map_factory import MapFactory


def create_test_textures(base_dir, num_sets=5, size=512):
    """Create realistic TGA texture sets mimicking c130j_fuselage naming."""
    texture_paths = []
    map_configs = {
        "DIFF": ("RGB", (128, 128, 128)),
        "NORM": ("RGB", (128, 128, 255)),
        "SPEC": ("L", 128),
        "AO": ("L", 200),
    }
    for i in range(1, num_sets + 1):
        set_name = f"c130j_fuselage_{i:02d}"
        for suffix, (mode, color) in map_configs.items():
            filename = f"{set_name}_{suffix}.tga"
            filepath = os.path.join(base_dir, filename)
            img = ImgUtils.create_image(mode, (size, size), color)
            ImgUtils.save_image(img, filepath)
            texture_paths.append(filepath)
    return texture_paths


def bench_prepare_maps(texture_paths, output_dir, config_name, **extra):
    """Benchmark prepare_maps and return elapsed time."""
    from pythontk.img_utils.map_registry import MapRegistry

    cfg = MapRegistry().resolve_config(config_name)
    cfg["output_extension"] = "png"
    cfg.update(extra)

    start = time.perf_counter()
    results = MapFactory.prepare_maps(
        texture_paths,
        output_dir=output_dir,
        group_by_set=True,
        **cfg,
    )
    elapsed = time.perf_counter() - start
    return elapsed, results


def bench_resolve_map_type(texture_paths, iterations=100):
    """Benchmark resolve_map_type calls."""
    start = time.perf_counter()
    for _ in range(iterations):
        for path in texture_paths:
            MapFactory.resolve_map_type(path)
    elapsed = time.perf_counter() - start
    total_calls = iterations * len(texture_paths)
    return elapsed, total_calls


def main():
    test_dir = tempfile.mkdtemp(prefix="bench_map_factory_")
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        num_sets = 10
        tex_size = 512

        print(f"Creating {num_sets} texture sets at {tex_size}x{tex_size}...")
        textures = create_test_textures(input_dir, num_sets=num_sets, size=tex_size)
        print(f"  Created {len(textures)} texture files")

        total_size_mb = sum(os.path.getsize(t) for t in textures) / (1024 * 1024)
        print(f"  Total input size: {total_size_mb:.1f} MB\n")

        # Benchmark resolve_map_type
        print("--- resolve_map_type benchmark ---")
        elapsed, total_calls = bench_resolve_map_type(textures, iterations=500)
        print(
            f"  {total_calls} calls in {elapsed:.3f}s ({total_calls/elapsed:.0f} calls/sec)\n"
        )

        # Benchmark Unity URP (metallic_smoothness packing)
        print("--- Unity URP prepare_maps benchmark ---")
        elapsed, results = bench_prepare_maps(textures, output_dir, "Unity URP Lit")
        result_count = sum(
            len(v) if isinstance(v, list) else 1
            for v in (results.values() if isinstance(results, dict) else [results])
        )
        print(
            f"  {num_sets} sets processed in {elapsed:.3f}s ({elapsed/num_sets:.3f}s per set)"
        )
        print(f"  Output files: {result_count}\n")

        # Benchmark with max_workers=4
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("--- Unity URP prepare_maps (max_workers=4) ---")
        elapsed_par, results_par = bench_prepare_maps(
            textures, output_dir, "Unity URP Lit", max_workers=4
        )
        result_count_par = sum(
            len(v) if isinstance(v, list) else 1
            for v in (
                results_par.values() if isinstance(results_par, dict) else [results_par]
            )
        )
        print(
            f"  {num_sets} sets processed in {elapsed_par:.3f}s ({elapsed_par/num_sets:.3f}s per set)"
        )
        print(f"  Output files: {result_count_par}")

        if elapsed > 0:
            speedup = elapsed / elapsed_par
            print(f"  Parallel speedup: {speedup:.1f}x\n")

        # Benchmark Standard PBR (no packing)
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("--- Standard PBR prepare_maps benchmark ---")
        elapsed_std, results_std = bench_prepare_maps(
            textures, output_dir, "PBR Metallic/Roughness"
        )
        result_count_std = sum(
            len(v) if isinstance(v, list) else 1
            for v in (
                results_std.values() if isinstance(results_std, dict) else [results_std]
            )
        )
        print(
            f"  {num_sets} sets processed in {elapsed_std:.3f}s ({elapsed_std/num_sets:.3f}s per set)"
        )
        print(f"  Output files: {result_count_std}\n")

        print("=== Summary ===")
        print(f"resolve_map_type: {total_calls/elapsed:.0f} calls/sec")
        print(
            f"Unity URP (serial):   {elapsed:.3f}s total, {elapsed/num_sets:.3f}s/set"
        )
        print(
            f"Unity URP (parallel): {elapsed_par:.3f}s total, {elapsed_par/num_sets:.3f}s/set"
        )
        print(
            f"Standard PBR:         {elapsed_std:.3f}s total, {elapsed_std/num_sets:.3f}s/set"
        )

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
