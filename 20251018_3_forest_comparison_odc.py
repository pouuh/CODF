"""
GLANCE vs MapBiomas Comparison using ODC-Geo
像GEE一样自动处理内存和分块！

安装依赖:
pip install odc-geo rioxarray geopandas xarray dask distributed
"""
#!/usr/bin/env python3
"""
GLANCE vs MapBiomas Forest Comparison (HPC + ODC + Dask)
--------------------------------------------------------
Lazy, HPC-optimized implementation using odc-stac and dask-distributed.

Author: Chishan Zhang
"""

# =====================================================
# 0. Imports & Dask Initialization
# =====================================================
import os
import numpy as np
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
from odc.stac import load
import hvplot.xarray  # noqa
import matplotlib.pyplot as plt


# ---------- Dask Cluster Setup ----------
def init_dask(n_workers=1, threads_per_worker=28, memory_limit="240GB", dashboard=True):
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=":8787" if dashboard else None,
    )
    client = Client(cluster)
    print("✅ Dask initialized:", client.dashboard_link if dashboard else "no dashboard")
    return client


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
        if i.datetime and 2016 == drop_tz(i.datetime).year
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
    """Convert land-cover classes to binary forest map (1=forest, 0=non-forest)."""
    return xr.where(arr.notnull(), arr.isin(forest_values).astype("uint8"), nodata)

# =====================================================
# 3. Agreement / Confusion Matrix
# =====================================================
def compute_confusion_lazy(ref, pred, nodata=255):
    """Compute confusion matrix lazily using Dask reduction."""
    valid = (ref != nodata) & (pred != nodata)
    tn = ((ref == 0) & (pred == 0) & valid).sum().compute()
    return tn
    
def compute_metrics(cm):
    tn, fp, fn, tp = cm
    tn, fp, fn, tp = map(float, (tn, fp, fn, tp))
    total = tn + fp + fn + tp
    overall = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return dict(overall=overall, precision=precision, recall=recall, f1=f1)

# =====================================================
# 4. Quick Visualization
# =====================================================
def preview_agreement(gl, mb, valid):
    """Show small sample hvplot agreement map."""
    win = dict(latitude=slice(0, 2000), longitude=slice(0, 2000))
    agree = (gl.isel(**win) == mb.isel(**win)).where(valid.isel(**win))
    return agree.hvplot.image(
        x="longitude", y="latitude", rasterize=True,
        cmap=["red", "green"], title="Forest Agreement (sample window)"
    )


# =====================================================
# 5. Main
# =====================================================
def main():
    year = 2016

    # ---- 1. Dask init ----
    client = init_dask(n_workers=1, threads_per_worker=28, memory_limit="240GB", dashboard=True)

    # ---- 2. Paths ----
    stac_path = "/projectnb/modislc/users/chishan/stac_glance_SA/catalog.json"
    mapbiomas_tif = f"/projectnb/modislc/users/chishan/data/MapBiomas/COG/AMZ.{year}.M.cog.tif"

    # ---- 3. Load MapBiomas ----
    import rioxarray
    mb = rioxarray.open_rasterio(mapbiomas_tif, chunks={"x": 2048, "y": 2048})
    mb = mb.squeeze(drop=True)
    mb_geobox = mb.odc.geobox
    print("✓ MapBiomas loaded:", mb.shape, mb.rio.crs)

    # ---- 4. Load GLANCE aligned to MapBiomas ----
    ds = load_glance_stac(stac_path, year, mapbiomas_geobox=mb_geobox)
    gl = ds["data"].isel(time=0)
    print("✓ GLANCE loaded:", gl.shape, gl.rio.crs)

    # ---- 5. Reclassify ----
    mb_bin = reclassify_to_forest(mb, [1, 2, 9])
    gl_bin = reclassify_to_forest(gl, [5])
    # valid = (mb_bin != 255) & (gl_bin != 255)

    # ---- 6. Compute confusion matrix ----
    tn = compute_confusion_lazy(mb_bin, gl_bin)
    print(f"TN: {tn}")

    # ---- 8. Clean up ----
    client.close()
    print("🧹 Dask cluster closed.")

# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    main()
