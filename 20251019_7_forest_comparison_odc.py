#!/usr/bin/env python3
"""
GLANCE vs MapBiomas Forest Comparison (HPC + ODC + Dask)
--------------------------------------------------------
Lazy, HPC-friendly implementation using odc-stac, dask, and map_blocks.
"""
import os
import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask import config as dask_config
from odc.stac import load as odc_load
import odc.geo.xr  # 激活 xarray 的 odc accessor

import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import rioxarray


# =====================================================
# 1. Load STAC and MapBiomas Data
# =====================================================
def load_glance_stac(stac_path, year, mapbiomas_geobox):
    """Load GLANCE tiles using odc.stac.load (original stable version)."""
    import pystac
    from datetime import datetime

    catalog = pystac.Catalog.from_file(stac_path)

    def drop_tz(dt):
        return dt.replace(tzinfo=None) if dt.tzinfo else dt

    items = [
        item
        for item in catalog.get_items(recursive=True)
        if item.datetime and drop_tz(item.datetime).year == year
    ]
    print(f"✓ {len(items)} GLANCE items found for {year}")

    # 使用 odc.stac.load（原始方案，最稳定）
    ds = odc_load(
        items,
        geobox=mapbiomas_geobox,
        chunks={"x": 1024, "y": 1024},  # 与 MapBiomas 保持一致
        resampling="nearest",
        fail_on_error=False,
    )
    
    print(f"✓ ODC loaded: shape={ds.dims}, chunks={ds.chunks}")
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
def _ensure_shared_chunks(ref: xr.DataArray, pred: xr.DataArray, chunk_hint=None):
    """Align coordinates and make sure both arrays share the same chunk structure."""
    ref_aligned, pred_aligned = xr.align(ref, pred, join="exact")

    if chunk_hint is not None:
        ref_aligned = ref_aligned.chunk(chunk_hint)
        pred_aligned = pred_aligned.chunk(chunk_hint)
        return ref_aligned, pred_aligned

    if ref_aligned.chunks and pred_aligned.chunks and ref_aligned.chunks == pred_aligned.chunks:
        return ref_aligned, pred_aligned

    target = {}
    for dim in ref_aligned.dims:
        ref_chunks = ref_aligned.chunks.get(dim)
        pred_chunks = pred_aligned.chunks.get(dim)
        ref_first = ref_chunks[0] if ref_chunks else ref_aligned.sizes[dim]
        pred_first = pred_chunks[0] if pred_chunks else pred_aligned.sizes[dim]
        target[dim] = min(ref_first, pred_first)
    ref_aligned = ref_aligned.chunk(target)
    pred_aligned = pred_aligned.chunk(target)
    return ref_aligned, pred_aligned


def _confusion_counts_block(ref_block, pred_block, nodata):
    """Compute TN/FP/FN/TP for a single chunk."""
    ref_arr = ref_block.astype(np.uint8, copy=False)
    pred_arr = pred_block.astype(np.uint8, copy=False)
    valid = (ref_arr != nodata) & (pred_arr != nodata)

    if not np.any(valid):
        counts = np.zeros(4, dtype=np.int64)
    else:
        tn = np.sum((ref_arr == 0) & (pred_arr == 0) & valid)
        fp = np.sum((ref_arr == 0) & (pred_arr == 1) & valid)
        fn = np.sum((ref_arr == 1) & (pred_arr == 0) & valid)
        tp = np.sum((ref_arr == 1) & (pred_arr == 1) & valid)
        counts = np.array([tn, fp, fn, tp], dtype=np.int64)

    out_shape = (1,) * ref_arr.ndim + (4,)
    return counts.reshape(out_shape)


def compute_confusion_lazy(ref, pred, nodata=255, chunk_hint=None, metrics=("tn", "fp", "fn", "tp")):
    """Chunk-aware confusion-matrix aggregation using dask.map_blocks."""
    if not isinstance(ref, xr.DataArray) or not isinstance(pred, xr.DataArray):
        raise TypeError("Expected xarray.DataArray inputs.")

    ref, pred = _ensure_shared_chunks(ref, pred, chunk_hint=chunk_hint)

    output_chunks = (1,) * ref.ndim + (4,)
    block_results = da.map_blocks(
        _confusion_counts_block,
        ref.data,
        pred.data,
        dtype=np.int64,
        chunks=output_chunks,
        nodata=nodata,
    )

    summed = block_results.sum(axis=tuple(range(ref.ndim)))
    # totals = summed.compute()
    # tn, fp, fn, tp = totals.tolist()
    totals = np.asarray(summed.compute()).ravel()
    tn, fp, fn, tp = totals[:4]


    mapping = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return {key: int(mapping[key]) for key in metrics}


def compute_metrics(cm):
    tn, fp, fn, tp = map(float, cm)
    total = tn + fp + fn + tp
    overall = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(overall=overall, precision=precision, recall=recall, f1=f1)


# =====================================================
# 4. Quick Visualization
# =====================================================
def preview_agreement(gl, mb, valid, window=2000):
    """Show a small sample hvplot agreement map."""
    y_dim, x_dim = gl.dims[-2], gl.dims[-1]
    win = {y_dim: slice(0, window), x_dim: slice(0, window)}
    agree = (gl.isel(**win) == mb.isel(**win)).where(valid.isel(**win))
    return agree.hvplot.image(
        x=x_dim,
        y=y_dim,
        rasterize=True,
        cmap=["red", "green"],
        title="Forest Agreement (sample window)",
    )


# =====================================================
# 5. Main
# =====================================================
def main():
    year = 2016

    # # 🔧 限制 GDAL/线程库，避免与 Dask 竞争
    # os.environ.setdefault("GDAL_CACHEMAX", "128")      # 减少 GDAL 内存缓存 (MB)
    # os.environ.setdefault("OMP_NUM_THREADS", "1")      # 禁用 OpenMP 多线程
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    
    # 🔧 启动 Dask LocalCluster（进程模式 + 内存限制）
    from dask.distributed import LocalCluster, Client
    
    n_workers = 8
    per_worker_mem = "30GB"  # 调整使 n_workers * per_worker_mem < 申请的总内存
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,  # 单线程更稳定
        memory_limit=per_worker_mem,
        processes=True,  # 进程模式避免 GIL
    )
    client = Client(cluster)
    
    print(f"✅ Dask cluster: workers={n_workers}, memory={per_worker_mem}/worker, dashboard={client.dashboard_link}")
    
    stac_path = "/projectnb/modislc/users/chishan/stac_glance_SA_fixed_m/catalog.json"
    mapbiomas_tif = f"/projectnb/modislc/users/chishan/data/MapBiomas/COG/AMZ.{year}.M.cog.tif"
  
    # 先用 1024 chunk 对比效果
    mb = rioxarray.open_rasterio(mapbiomas_tif, chunks={"x": 1024, "y": 1024}).squeeze(drop=True)
    mb_geobox = mb.odc.geobox  # 获取 geobox 用于精确对齐
    
    # 🔧 读取 MapBiomas 的 nodata 值
    mb_nodata = mb.rio.nodata
    if mb_nodata is None:
        mb_nodata = 0  # MapBiomas 默认 0 为 nodata
    print(f"✓ MapBiomas loaded: shape={mb.shape}, CRS={mb.rio.crs}, nodata={mb_nodata}")
    print(f"  Geobox: width={mb_geobox.width}, height={mb_geobox.height}, resolution={mb_geobox.resolution}")
    
    ds = load_glance_stac(stac_path, year, mapbiomas_geobox=mb_geobox)
    
    # odc.stac.load 返回 Dataset，选择数据变量
    # 假设你的 STAC items 有 "forest" 或其他 asset，需要根据实际情况调整
    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        raise ValueError("No data variables found in GLANCE dataset")
    
    gl = ds[data_vars[0]].squeeze(drop=True)  # 选择第一个变量并移除单维度
    gl = gl.rename({'latitude': 'y', 'longitude': 'x'})

    # 🔧 读取 GLANCE 的 nodata 值
    gl_nodata = gl.rio.nodata if hasattr(gl, 'rio') and gl.rio.nodata is not None else 255
    
    print(f"✓ GLANCE loaded: shape={gl.shape}, dims={gl.dims}, nodata={gl_nodata}")
    
    # 验证维度一致（odc.stac.load 使用 geobox 应该完全对齐）
    assert gl.dims == mb.dims, f"Dimension mismatch: {gl.dims} != {mb.dims}"
    assert gl.shape == mb.shape, f"Shape mismatch: {gl.shape} != {mb.shape}"
    print(f"✅ Dimensions aligned: {mb.dims}, shape={mb.shape}")

    # 🔧 重分类时使用各自的 nodata 值
    mb_bin = reclassify_to_forest(mb, [1, 2, 9], nodata=mb_nodata)
    gl_bin = reclassify_to_forest(gl, [5], nodata=gl_nodata)
    print("✓ Reclassification complete")

    # 🔧 统一 nodata 为 255（用于混淆矩阵计算）
    # 如果原始 nodata 不是 255，需要先替换
    unified_nodata = 255
    if mb_nodata != unified_nodata:
        mb_bin = xr.where(mb_bin == mb_nodata, unified_nodata, mb_bin)
        print(f"✓ Unified MapBiomas nodata: {mb_nodata} → {unified_nodata}")
    if gl_nodata != unified_nodata:
        gl_bin = xr.where(gl_bin == gl_nodata, unified_nodata, gl_bin)
        print(f"✓ Unified GLANCE nodata: {gl_nodata} → {unified_nodata}")

    # 先用 1024 chunk 对比效果
    counts = compute_confusion_lazy(mb_bin, gl_bin, nodata=unified_nodata, chunk_hint={"y": 1024, "x": 1024})
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    print(f"Confusion counts -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    stats = compute_metrics((tn, fp, fn, tp))
    print("Metrics:", stats)
    
    # 关闭 Dask cluster
    client.close()
    print("✓ Dask cluster closed")


if __name__ == "__main__":
    main()
