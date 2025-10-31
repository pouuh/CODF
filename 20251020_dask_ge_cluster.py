#!/usr/bin/env python3
"""
GLANCE vs MapBiomas Forest Comparison (HPC + ODC + Dask on BU SCC)
------------------------------------------------------------------
- Multi-node via dask-jobqueue SGECluster (GridEngine on SCC)
- Auto NIC detection (ib0/p3p1/p3p2/em1/eth0) for stable TCP
- odc-stac load aligned to MapBiomas geobox
- Chunk-wise confusion counts with dask.map_blocks (memory-safe)
"""

import os
import time
import numpy as np
import xarray as xr
import dask.array as da
from dask.distributed import Client, performance_report
from dask_jobqueue import SGECluster
from odc.stac import load as odc_load
import odc.geo.xr  # activate xarray ODC accessor
import rioxarray


# =====================================================
# 0. Utilities
# =====================================================
def detect_interface():
    """Return a best-effort network interface name present on SCC nodes."""
    try:
        import psutil
        nics = set(psutil.net_if_addrs().keys())
    except Exception:
        nics = set()

    for name in ("ib0", "p3p1", "p3p2", "em1", "eth0"):
        if name in nics:
            print(f"✅ Using network interface: {name}")
            return name
    print("⚠️ No preferred NIC found, letting Dask choose automatically.")
    return None


def ensure_uint8(arr, nodata):
    """Cast to uint8 safely and ensure nodata value fits 0-255."""
    nodata_u8 = np.uint8(nodata if 0 <= int(nodata) <= 255 else 255)
    out = arr.astype("uint8", copy=False)
    if arr.dtype.kind != "u" or int(nodata) != int(nodata_u8):
        out = xr.where(arr == nodata, nodata_u8, out).astype("uint8")
    return out, int(nodata_u8)


# =====================================================
# 1. Load STAC and MapBiomas Data
# =====================================================
def load_glance_stac(stac_path, year, mapbiomas_geobox):
    """Load GLANCE tiles using odc.stac.load aligned to MapBiomas geobox."""
    import pystac

    catalog = pystac.Catalog.from_file(stac_path)

    def drop_tz(dt):
        return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt

    items = [
        it for it in catalog.get_items(recursive=True)
        if getattr(it, "datetime", None) and drop_tz(it.datetime).year == year
    ]
    print(f"✓ {len(items)} GLANCE items found for {year}")

    ds = odc_load(
        items,
        geobox=mapbiomas_geobox,          # force exact same grid
        chunks={"x": 1024, "y": 1024},    # match MapBiomas chunking
        resampling="nearest",
        fail_on_error=False,
    )

    # Safer printing for sizes/chunks
    sizes = dict(ds.sizes)
    chunks = {k: tuple(map(int, v)) for k, v in getattr(ds, "chunks", {}).items()} if hasattr(ds, "chunks") else {}
    print(f"✓ ODC loaded: sizes={sizes}, chunks={chunks}")
    return ds


# =====================================================
# 2. Forest Reclassification
# =====================================================
def reclassify_to_forest(arr: xr.DataArray, forest_values, nodata=255):
    """Convert land-cover classes to binary forest map (1=forest, 0=non-forest, nodata preserved)."""
    # keep dtype compact, avoid object promotion
    out = xr.full_like(arr, fill_value=np.uint8(nodata), dtype="uint8")
    mask_valid = arr.notnull()
    mask_forest = arr.isin(forest_values)
    out = out.where(~mask_valid, np.uint8(0))           # default non-forest for valid pixels
    out = out.where(~mask_forest, np.uint8(1))          # set forest
    return out


# =====================================================
# 3. Confusion Matrix helpers
# =====================================================
def _ensure_shared_chunks(ref: xr.DataArray, pred: xr.DataArray, chunk_hint=None):
    """Align coords and enforce shared chunking for stable map_blocks."""
    ref_aligned, pred_aligned = xr.align(ref, pred, join="exact")

    if chunk_hint is not None:
        return ref_aligned.chunk(chunk_hint), pred_aligned.chunk(chunk_hint)

    # If both have chunks and equal
    if getattr(ref_aligned, "chunks", None) and getattr(pred_aligned, "chunks", None):
        if ref_aligned.chunks == pred_aligned.chunks:
            return ref_aligned, pred_aligned

    # Fallback: choose min of first-chunk sizes per dim
    target = {}
    for dim in ref_aligned.dims:
        rc = ref_aligned.chunks.get(dim) if getattr(ref_aligned, "chunks", None) else None
        pc = pred_aligned.chunks.get(dim) if getattr(pred_aligned, "chunks", None) else None
        r0 = rc[0] if rc else ref_aligned.sizes[dim]
        p0 = pc[0] if pc else pred_aligned.sizes[dim]
        target[dim] = min(int(r0), int(p0))
    return ref_aligned.chunk(target), pred_aligned.chunk(target)


def _confusion_counts_block(ref_block, pred_block, nodata):
    """Compute TN/FP/FN/TP for a single numpy block."""
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

    # map_blocks expects a block-shaped array; add singleton dims for all inputs
    out_shape = (1,) * ref_arr.ndim + (4,)
    return counts.reshape(out_shape)


# def compute_confusion_lazy(ref: xr.DataArray, pred: xr.DataArray, nodata=255, chunk_hint=None,
#                            metrics=("tn", "fp", "fn", "tp")):
#     """Chunk-aware confusion-matrix aggregation using dask.map_blocks."""
#     if not isinstance(ref, xr.DataArray) or not isinstance(pred, xr.DataArray):
#         raise TypeError("Expected xarray.DataArray inputs.")

#     ref, pred = _ensure_shared_chunks(ref, pred, chunk_hint=chunk_hint)

#     # Build proper "tuple of tuples" chunks spec for map_blocks
#     # e.g., if ref dims = (y,x) -> output chunks = ((1,), (1,), (4,))
#     out_chunks = tuple(((1,),) * ref.ndim + ((4,),))

#     block_results = da.map_blocks(
#         _confusion_counts_block,
#         ref.data,
#         pred.data,
#         dtype=np.int64,
#         chunks=out_chunks,
#         nodata=np.uint8(nodata),
#     )

#     # Sum over all singleton spatial dims ⇒ final shape (4,)
#     summed = block_results.sum(axis=tuple(range(ref.ndim)))
#     totals = np.asarray(summed.compute()).ravel()
#     tn, fp, fn, tp = map(int, totals[:4])

#     mapping = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
#     return {key: mapping[key] for key in metrics}

# def compute_confusion_lazy(ref, pred, nodata=255, chunk_hint=None, metrics=("tn","fp","fn","tp")):
#     if not isinstance(ref, xr.DataArray) or not isinstance(pred, xr.DataArray):
#         raise TypeError("Expected xarray.DataArray inputs.")

#     ref, pred = _ensure_shared_chunks(ref, pred, chunk_hint=chunk_hint)

#     # Build output chunk spec that matches the number of input blocks per dim
#     # e.g. if ref has chunks: y=(1024,1024,...), x=(1024,1024,...)
#     # then out_chunks should be: y=(1,1,...), x=(1,1,...) and a new axis (4,)
#     ref_chunks = ref.data.chunks  # tuple of tuples
#     out_chunks = tuple(tuple(1 for _ in dim_chunks) for dim_chunks in ref_chunks) + ((4,),)

#     block_results = da.map_blocks(
#         _confusion_counts_block,
#         ref.data,
#         pred.data,
#         dtype=np.int64,
#         chunks=out_chunks,
#         new_axis=ref.ndim,          # add the last axis of size 4
#         nodata=np.uint8(nodata),
#     )

#     summed = block_results.sum(axis=tuple(range(ref.ndim)))
#     totals = np.asarray(summed.compute()).ravel()
#     tn, fp, fn, tp = map(int, totals[:4])
#     return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

def compute_confusion_lazy(ref, pred, nodata=255, chunk_hint=None, metrics=("tn","fp","fn","tp")):
    if not isinstance(ref, xr.DataArray) or not isinstance(pred, xr.DataArray):
        raise TypeError("Expected xarray.DataArray inputs.")

    ref, pred = _ensure_shared_chunks(ref, pred, chunk_hint=chunk_hint)

    # mirror block counts: one 1-sized block per input block, plus a new axis (4,)
    ref_chunks = ref.data.chunks                # tuple of tuples per dim
    out_chunks = tuple(tuple(1 for _ in d) for d in ref_chunks) + ((4,),)

    block_results = da.map_blocks(
        _confusion_counts_block,
        ref.data, pred.data,
        dtype=np.int64,
        chunks=out_chunks,
        new_axis=ref.ndim,           # add the last 4-class axis
        nodata=np.uint8(nodata),
    )
    totals = np.asarray(block_results.sum(axis=tuple(range(ref.ndim))).compute()).ravel()
    tn, fp, fn, tp = map(int, totals[:4])
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

def compute_metrics(cm_tuple):
    tn, fp, fn, tp = map(float, cm_tuple)
    total = tn + fp + fn + tp
    overall = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(overall=overall, precision=precision, recall=recall, f1=f1)


# =====================================================
# 4. Main
# =====================================================
def main():
    year = 2016

    # -------- Dask cluster on SCC via SGECluster (scheduler light) --------
    nic = detect_interface()
    cluster = SGECluster(
        project="modislc",
        queue="geo",

        # per-worker resources (match SCC doc)
        cores=8,
        memory="100GB",
        walltime="12:00:00",
        resource_spec="h_rt=12:00:00,mem_free=100G",
        job_extra_directives=[
            "-l mem_per_core=12G",
            "-pe omp 8",
            "-V",
        ],

        # environment + paths
        local_directory=os.environ.get("TMPDIR", "/projectnb/modislc/users/chishan/tmp"),
        log_directory="/projectnb/modislc/users/chishan/logs",
        interface=nic,  # may be None → Dask auto-pick

        env_extra=[
            # 避免多重并行：都设为 1
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export NUMEXPR_MAX_THREADS=1",
            # Dask/GDAL 缓存
            "export DASK_TEMPORARY_DIRECTORY=/projectnb/modislc/users/chishan/dask_tmp",
            "export GDAL_CACHEMAX=256",   # 每进程 256 MB；processes=8 时总缓存≈2 GB
        ],

        # new API for dashboard
        scheduler_options={"dashboard_address": ":8787"},
    )

    # scale out multi-node workers; can tweak via env if needed
    jobs = int(os.environ.get("DASK_JOBS", "4"))
    cluster.scale(jobs=jobs)
    client = Client(cluster)

    print(f"\n🟢 Dask Dashboard : {cluster.dashboard_link}")
    import socket
    print(f"🟢 Scheduler Node : {socket.gethostname()}\n")

    # Wait workers
    target = jobs
    print("⏳ Waiting for workers to connect...")
    for i in range(60):
        n = len(client.scheduler_info()["workers"])
        if n >= target:
            break
        time.sleep(5)
        print(f"   {i*5:>3d}s: {n}/{target} workers connected")
    print("✅ Workers connected!")
    for w in client.scheduler_info()["workers"].keys():
        print("   ", w)

    # -------- Paths --------
    stac_path = "/projectnb/modislc/users/chishan/stac_glance_SA_fixed_m/catalog.json"
    mapbiomas_tif = f"/projectnb/modislc/users/chishan/data/MapBiomas/COG/AMZ.{year}.M.cog.tif"

    # -------- Load MapBiomas --------
    mb = rioxarray.open_rasterio(mapbiomas_tif, chunks={"x": 1024, "y": 1024}).squeeze(drop=True)
    # keep names consistent (should already be y/x when opened via rioxarray)
    if "latitude" in mb.dims and "longitude" in mb.dims:
        mb = mb.rename({"latitude": "y", "longitude": "x"})

    mb_geobox = mb.odc.geobox
    mb_nodata = mb.rio.nodata
    if mb_nodata is None:
        mb_nodata = 0  # MapBiomas 常见约定
    print(f"✓ MapBiomas loaded: shape={mb.shape}, dims={mb.dims}, CRS={mb.rio.crs}, nodata={mb_nodata}")
    print(f"  Geobox: width={mb_geobox.width}, height={mb_geobox.height}, res={mb_geobox.resolution}")

    # -------- Load GLANCE aligned to MapBiomas grid --------
    ds = load_glance_stac(stac_path, year, mapbiomas_geobox=mb_geobox)
    if len(ds.data_vars) == 0:
        raise ValueError("No data variables found in GLANCE dataset")
    # pick the first data var (adjust if you have a specific asset name)
    first_var = list(ds.data_vars)[0]
    gl = ds[first_var].squeeze(drop=True)

    # standardize dims to y/x if needed
    rename_map = {}
    if "latitude" in gl.dims: rename_map["latitude"] = "y"
    if "longitude" in gl.dims: rename_map["longitude"] = "x"
    if rename_map:
        gl = gl.rename(rename_map)

    gl_nodata = gl.rio.nodata if hasattr(gl, "rio") and (gl.rio.nodata is not None) else 255
    print(f"✓ GLANCE loaded: shape={gl.shape}, dims={gl.dims}, nodata={gl_nodata}")

    # ensure shapes/dims identical
    assert gl.dims == mb.dims, f"Dimension mismatch: {gl.dims} != {mb.dims}"
    assert gl.shape == mb.shape, f"Shape mismatch: {gl.shape} != {mb.shape}"
    print(f"✅ Dimensions aligned: {mb.dims}, shape={mb.shape}")

    # -------- Reclassify to binary --------
    mb_bin = reclassify_to_forest(mb, forest_values=[1, 2, 9], nodata=mb_nodata)
    gl_bin = reclassify_to_forest(gl, forest_values=[5], nodata=gl_nodata)
    print("✓ Reclassification complete")

    # -------- Unify nodata to 255 and cast to uint8 --------
    unified_nodata = 255
    mb_bin_u8, _ = ensure_uint8(mb_bin, mb_nodata)
    gl_bin_u8, _ = ensure_uint8(gl_bin, gl_nodata)
    if mb_nodata != unified_nodata:
        mb_bin_u8 = xr.where(mb_bin_u8 == mb_nodata, np.uint8(unified_nodata), mb_bin_u8)
        print(f"✓ Unified MapBiomas nodata: {mb_nodata} → {unified_nodata}")
    if gl_nodata != unified_nodata:
        gl_bin_u8 = xr.where(gl_bin_u8 == gl_nodata, np.uint8(unified_nodata), gl_bin_u8)
        print(f"✓ Unified GLANCE nodata: {gl_nodata} → {unified_nodata}")

    # -------- Confusion counts (chunk-wise) --------
    from dask.distributed import performance_report
    import os
    report_path = f"/usr2/postdoc/chishan/logs/dask_report_{os.environ.get('JOB_ID','manual')}.html"
    with performance_report(filename=report_path):
        counts = compute_confusion_lazy(
            mb_bin_u8, gl_bin_u8,
            nodata=unified_nodata,
            chunk_hint={"y": 1024, "x": 1024}
        )
        tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
        print(f"Confusion counts -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

        stats = compute_metrics((tn, fp, fn, tp))
        print("Metrics:", stats)

    client.close()
    print("✓ Dask cluster closed; performance report: dask_report.html")


if __name__ == "__main__":
    # Optional: tame thread oversubscription from math libs
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

    main()