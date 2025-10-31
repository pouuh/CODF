#!/usr/bin/env python3
"""
GLANCE vs MapBiomas Forest Comparison (HPC + ODC + Dask)
--------------------------------------------------------
Lazy, HPC-optimized implementation using odc-stac and dask-distributed.

Author: Chishan Zhang
"""
import os
from functools import partial

import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask import delayed
from dask.distributed import Client, LocalCluster
from odc.stac import load
import hvplot.xarray  # noqa
import matplotlib.pyplot as plt

# =====================================================
# 1. Load STAC and MapBiomas Data
# =====================================================
def load_glance_stac(stac_path, year, mapbiomas_geobox):
    """Load GLANCE tiles for a given year (already projected to MapBiomas geobox)."""
    import pystac
    from datetime import datetime, timezone

    cat = pystac.Catalog.from_file(stac_path)

    def drop_tz(dt):
        return dt.replace(tzinfo=None) if dt.tzinfo else dt

    items = [
        i for i in cat.get_items(recursive=True)
        if i.datetime and year == drop_tz(i.datetime).year
    ]

    print(f"✓ {len(items)} GLANCE items found for {year}")
    ds = load(
        items,
        geobox=mapbiomas_geobox,
        chunks={"x": 1024, "y": 1024},
        groupby="solar_day",
    )
    return ds


# =====================================================
# 2. Forest Reclassification
# =====================================================
def reclassify_to_forest(arr: xr.DataArray, forest_values, nodata=255):
    """Convert land-cover classes to binary forest map (1=forest, 0=non-forest)."""
    return xr.where(arr.notnull(), arr.isin(forest_values).astype("uint8"), nodata)


# =====================================================
# 3. Confusion Matrix helpers
# =====================================================

def _confusion_counts_block(ref_block, pred_block, nodata):
    ref_block = np.asarray(ref_block)
    pred_block = np.asarray(pred_block)
    valid = (ref_block != nodata) & (pred_block != nodata)
    if not np.any(valid):
        return np.zeros(4, dtype=np.int64)
    ref_valid = ref_block[valid].astype(np.uint8)
    pred_valid = pred_block[valid].astype(np.uint8)
    codes = (ref_valid << 1) | pred_valid
    hist = np.bincount(codes.ravel(), minlength=4).astype(np.int64)
    return hist  # [tn, fp, fn, tp]

def _sum_counts(chunks):
    total = np.zeros(4, dtype=np.int64)
    for arr in chunks:
        total += arr
    return total


def _align_and_chunk(ref: xr.DataArray, pred: xr.DataArray, chunk_hint=None):
    """Align coordinates and rechunk arrays to same chunk sizes."""
    # Align coordinates (assumes dimensions already match)
    ref, pred = xr.align(ref, pred, join="exact")

    # Determine chunk sizes
    if not chunk_hint:
        chunk_hint = {}
        for arr in (ref, pred):
            if arr.chunks:
                for dim, sizes in arr.chunks.items():
                    chunk_hint[dim] = min(chunk_hint.get(dim, sizes[0]), sizes[0]) if dim in chunk_hint else sizes[0]
        if not chunk_hint:
            chunk_hint = {dim: min(1024, size) for dim, size in ref.sizes.items()}

    ref = ref.chunk(chunk_hint)
    pred = pred.chunk(chunk_hint)
    return ref, pred


def compute_confusion_lazy(ref, pred, nodata=255, chunk_hint=None, metrics=("tn", "fp", "fn", "tp")):
    if not isinstance(ref, xr.DataArray) or not isinstance(pred, xr.DataArray):
        raise TypeError("Expected xarray.DataArray inputs.")

    ref, pred = _align_and_chunk(ref, pred, chunk_hint)

    ref_da = ref.data
    pred_da = pred.data
    if ref_da.chunks != pred_da.chunks:
        fallback = {
            dim: min(ref.chunks[dim][0], pred.chunks[dim][0])
            for dim in ref.dims
        }
        ref = ref.chunk(fallback)
        pred = pred.chunk(fallback)
        ref_da, pred_da = ref.data, pred.data

    ref_blocks = ref_da.to_delayed().ravel()
    pred_blocks = pred_da.to_delayed().ravel()
    if ref_blocks.size != pred_blocks.size:
        raise RuntimeError("Chunk topology mismatch after alignment.")

    tasks = [
        delayed(_confusion_counts_block)(rb, pb, nodata)
        for rb, pb in zip(ref_blocks.tolist(), pred_blocks.tolist())
    ]
    total = delayed(_sum_counts)(tasks)
    counts = dask.compute(total)[0]

    mapping = {"tn": 0, "fp": 1, "fn": 2, "tp": 3}
    return {m: int(counts[mapping[m]]) for m in metrics}

def compute_metrics(cm):
    tn, fp, fn, tp = cm
    tn, fp, fn, tp = map(float, (tn, fp, fn, tp))
    total = tn + fp + fn + tp
    overall = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(overall=overall, precision=precision, recall=recall, f1=f1)


# =====================================================
# 4. Quick Visualization
# =====================================================
def preview_agreement(gl, mb, valid):
    """Show small sample hvplot agreement map."""
    win = dict(latitude=slice(0, 2000), longitude=slice(0, 2000))
    agree = (gl.isel(**win) == mb.isel(**win)).where(valid.isel(**win))
    return agree.hvplot.image(
        x="longitude",
        y="latitude",
        rasterize=True,
        cmap=["red", "green"],
        title="Forest Agreement (sample window)",
    )


# =====================================================
# 5. Main
# =====================================================
def main():
    year = 2016

    # client = init_dask(n_workers=1, threads_per_worker=6, memory_limit="240GB", dashboard=True)
    # ---- 1. 配置单机调度器 ----
    dask.config.set({
        'scheduler': 'threads',
        'num_workers': 8,
        'array.slicing.split_large_chunks': True,
    })
    print("✅ Dask threads scheduler enabled (8 threads)")

    stac_path = "/projectnb/modislc/users/chishan/stac_glance_SA_fixed_m/catalog.json"
    mapbiomas_tif = f"/projectnb/modislc/users/chishan/data/MapBiomas/COG/AMZ.{year}.M.cog.tif"

    import rioxarray
    mb = rioxarray.open_rasterio(mapbiomas_tif, chunks={"x": 1024, "y": 1024})
    mb = mb.squeeze(drop=True)
    mb_geobox = mb.odc.geobox
    print(f"✓ MapBiomas loaded: shape={mb.shape}, dims={mb.dims}, CRS={mb.rio.crs}")

    ds = load_glance_stac(stac_path, year, mapbiomas_geobox=mb_geobox)
    gl = ds["data"].isel(time=0)
    print(f"✓ GLANCE loaded: shape={gl.shape}, dims={gl.dims}, CRS={gl.rio.crs}")

    # 🔧 统一维度名（关键修复：防止广播）
    if gl.dims != mb.dims:
        print(f"⚠️  Dimension mismatch detected: mb={mb.dims}, gl={gl.dims}")
        if 'latitude' in gl.dims and 'y' in mb.dims:
            gl = gl.rename({"latitude": "y", "longitude": "x"})
            print(f"✓ Renamed GLANCE dims to {gl.dims}")
        elif 'y' in gl.dims and 'latitude' in mb.dims:
            mb = mb.rename({"y": "latitude", "x": "longitude"})
            print(f"✓ Renamed MapBiomas dims to {mb.dims}")
    
    # 验证维度一致
    assert gl.dims == mb.dims, f"Dimension mismatch after rename: {gl.dims} != {mb.dims}"
    assert gl.shape == mb.shape, f"Shape mismatch: {gl.shape} != {mb.shape}"
    print(f"✅ Dimensions aligned: {mb.dims}, shape={mb.shape}")

    mb_bin = reclassify_to_forest(mb, [1, 2, 9])
    gl_bin = reclassify_to_forest(gl, [5])
    print("✓ Reclassification complete")

    counts = compute_confusion_lazy(mb_bin, gl_bin, nodata=255, chunk_hint={list(mb.dims)[1]: 1024, list(mb.dims)[0]: 1024})
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    tp = counts["tp"]
    print(f"Confusion counts -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    stats = compute_metrics((tn, fp, fn, tp))
    print("Metrics:", stats)

if __name__ == "__main__":
    main()
