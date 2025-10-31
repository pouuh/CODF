"""
GLANCE vs MapBiomas Comparison using ODC-Geo
像GEE一样自动处理内存和分块！
"""
#!/usr/bin/env python3

import os
import numpy as np
import xarray as xr
import dask
import dask.array as da
from odc.stac import load
import rioxarray
import pystac

# =====================================================
# 1. Load STAC and MapBiomas Data
# =====================================================
def load_glance_stac(stac_path, year, mapbiomas_geobox):
    """Load GLANCE tiles for a given year."""
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
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    return ds

# =====================================================
# 2. Forest Reclassification
# =====================================================
def reclassify_to_forest(arr: xr.DataArray, forest_values, nodata=255):
    """Convert land-cover classes to binary forest map."""
    return xr.where(arr.notnull(), arr.isin(forest_values).astype("uint8"), nodata)

# =====================================================
# 3. Confusion Matrix (Optimized)
# =====================================================
def compute_confusion_lazy(ref, pred, nodata=255):
    """Process block-by-block to avoid memory issues."""
    ref_da = ref.data if hasattr(ref, 'data') else ref
    pred_da = pred.data if hasattr(pred, 'data') else pred
    
    print(f"✓ Input shapes: ref={ref_da.shape}, pred={pred_da.shape}")
    print(f"✓ Chunk sizes: ref={ref_da.chunksize}, pred={pred_da.chunksize}")
    
    def process_block(ref_block, pred_block):
        valid = (ref_block != nodata) & (pred_block != nodata)
        return np.array([
            np.sum((ref_block == 0) & (pred_block == 0) & valid),  # TN
            np.sum((ref_block == 0) & (pred_block == 1) & valid),  # FP
            np.sum((ref_block == 1) & (pred_block == 0) & valid),  # FN
            np.sum((ref_block == 1) & (pred_block == 1) & valid),  # TP
        ], dtype=np.int64)
    
    results = da.map_blocks(
        process_block,
        ref_da, pred_da,
        dtype=np.int64,
        drop_axis=[0, 1],
        new_axis=0,
        chunks=(4,)
    )
    
    print("⏳ Computing confusion matrix...")
    totals = results.sum(axis=0).compute()
    return tuple(map(int, totals))

def compute_metrics(cm):
    tn, fp, fn, tp = map(float, cm)
    total = tn + fp + fn + tp
    overall = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return dict(overall=overall, precision=precision, recall=recall, f1=f1)

# =====================================================
# 4. Main
# =====================================================
def main():
    year = 2016

    # ---- 1. 配置单机调度器 ----
    dask.config.set({
        'scheduler': 'threads',
        'num_workers': 28,
        'array.slicing.split_large_chunks': True,
    })
    print("✅ Dask threads scheduler enabled (28 threads)")

    # ---- 2. Paths ----
    stac_path = "/projectnb/modislc/users/chishan/stac_glance_SA/catalog.json"
    mapbiomas_tif = f"/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/AMZ.{year}.M.tif"

    # ---- 3. Load MapBiomas ----
    mb = rioxarray.open_rasterio(mapbiomas_tif, chunks={"x": 2048, "y": 2048})
    mb = mb.squeeze(drop=True)
    mb_geobox = mb.odc.geobox
    print(f"✓ MapBiomas loaded: {mb.shape}, {mb.rio.crs}")

    # ---- 4. Load GLANCE ----
    ds = load_glance_stac(stac_path, year, mapbiomas_geobox=mb_geobox)
    gl = ds["data"].isel(time=0)
    print(f"✓ GLANCE loaded: {gl.shape}, {gl.rio.crs}")

    # ---- 5. Reclassify ----
    mb_bin = reclassify_to_forest(mb, [1, 2, 9])
    gl_bin = reclassify_to_forest(gl, [5])
    print("✓ Reclassification prepared")

    # ---- 6. Compute confusion matrix ----
    tn, fp, fn, tp = compute_confusion_lazy(mb_bin, gl_bin)
    metrics = compute_metrics((tn, fp, fn, tp))

    print("\n=== CONFUSION MATRIX ===")
    print(f"[[TN {tn:>12,}, FP {fp:>12,}]")
    print(f" [FN {fn:>12,}, TP {tp:>12,}]]")
    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()