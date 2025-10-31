#!/usr/bin/env python3
"""
GLANCE vs MapBiomas Amazon Forest Comparison (Lazy + HPC-Optimized)
====================================================================
Fully Dask-parallel implementation for large-scale binary
forest/non-forest comparison between GLANCE and MapBiomas datasets.

Optimized for:
  - Lazy computation with xarray + dask
  - Distributed chunk processing
  - Low memory footprint (~chunk-size)
  - Compatible with HPC clusters (Frontera, Slurm, etc.)

Date: 2025-10-15
"""

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import rioxarray
import dask
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. QML LEGEND PARSING
# =============================================================================
def parse_qml_legend(qml_path):
    """Parse QGIS QML legend to extract class value-label pairs."""
    tree = ET.parse(qml_path)
    root = tree.getroot()

    classes = {}
    for entry in root.findall(".//paletteEntry"):
        val, label = entry.get("value"), entry.get("label")
        if val and label:
            classes[int(val)] = label

    if not classes:
        for cat in root.findall(".//category"):
            val, label = cat.get("value"), cat.get("label")
            if val and label:
                classes[int(val)] = label

    print(f"✓ Parsed {len(classes)} classes from {os.path.basename(qml_path)}")
    return classes


# =============================================================================
# 2. MAPBIOMAS RECLASSIFICATION
# =============================================================================
def create_mapbiomas_forest_map(classes_dict, include_plantations=True):
    """Define which MapBiomas classes are forest vs non-forest."""
    forest_classes = [1, 2, 9] if include_plantations else [1, 2]
    reclass_map = {v: (1 if v in forest_classes else 0) for v in classes_dict.keys()}
    print(f"✓ Forest classes: {forest_classes}")
    return reclass_map

def reclassify_to_binary_forest(raster_array, reclass_map, nodata_value=0):
    """Lazy + chunk-safe reclassification using dask.map_blocks."""
    if not isinstance(raster_array, xr.DataArray):
        raise TypeError("Input must be xarray.DataArray")

    arr = raster_array.squeeze(drop=True)
    lut = np.zeros(256, dtype="uint8")
    for k, v in reclass_map.items():
        lut[k] = v

    # def _reclassify_block(block, lut):
    #     block = block.astype(np.int64, copy=False)
    #     mask = (block >= 0) & (block < len(lut))
    #     out = np.zeros_like(block, dtype="uint8")
    #     out[mask] = lut[block[mask]]
    #     return out

    def _reclassify_block(block, lut):
        data = block.data if hasattr(block, "data") else block
        data = data.astype(np.int64, copy=False)
        mask = (data >= 0) & (data < len(lut))
        out = np.zeros_like(data, dtype="uint8")
        out[mask] = lut[data[mask]]
        
        # ✅ Must return a DataArray with same dims/coords
        return xr.DataArray(
            out,
            dims=block.dims,
            coords=block.coords,
            attrs=block.attrs,
        )


    # template = xr.full_like(arr, 0, dtype="uint8")
    template = xr.zeros_like(arr, dtype="uint8")
    out = xr.map_blocks(_reclassify_block, arr, kwargs={"lut": lut}, template=template)

    valid_vals = np.array(list(reclass_map.keys()))
    is_valid = xr.apply_ufunc(np.isin, arr, valid_vals, dask="allowed", vectorize=True)
    out = xr.where(is_valid, out, nodata_value)

    if hasattr(arr, "rio"):
        out.rio.write_crs(arr.rio.crs, inplace=True)
        out.rio.write_nodata(nodata_value, inplace=True)

    print(f"✓ Reclassification graph built safely | chunks={out.chunks}")
    return out

# =============================================================================
# 3. GLANCE TILE LOADING
# =============================================================================
def find_glance_tiles_by_year(base_folder, year, region="SA"):
    pattern = os.path.join(base_folder, f"GLANCE.A{year}0701.h*v*.001.*.{region}.LC.tif")
    files = sorted(glob.glob(pattern))
    print(f"✓ Found {len(files)} GLANCE tiles for {year}")
    return files


def load_glance_tiles_lazy(file_list, chunks={"x": 4096, "y": 4096}):
    from rioxarray.merge import merge_arrays
    datasets = [rioxarray.open_rasterio(f, chunks=chunks, lock=False) for f in file_list]
    mosaic = merge_arrays(datasets, nodata=0)
    print(f"✓ Merged GLANCE mosaic | shape={mosaic.shape} | chunks={mosaic.chunks}")
    return mosaic


# =============================================================================
# 4. RECLASSIFY GLANCE
# =============================================================================
def reclassify_glance_to_forest(glance_array, forest_classes=[5]):
    reclass_map = {i: (1 if i in forest_classes else 0) for i in range(1, 8)}
    return reclassify_to_binary_forest(glance_array, reclass_map, nodata_value=0)


# =============================================================================
# 5. ALIGN & COMPARE
# =============================================================================
def align_and_compare_maps(glance, reference):
    """Align reference map to GLANCE grid and compute lazy agreement map."""
    if glance.rio.crs != reference.rio.crs:
        reference = reference.rio.reproject_match(glance, resampling=5)

    bounds_g, bounds_r = glance.rio.bounds(), reference.rio.bounds()
    overlap = (
        max(bounds_g[0], bounds_r[0]),
        max(bounds_g[1], bounds_r[1]),
        min(bounds_g[2], bounds_r[2]),
        min(bounds_g[3], bounds_r[3]),
    )

    g_clip = glance.rio.clip_box(*overlap).chunk({"x": 4096, "y": 4096})
    r_clip = reference.rio.clip_box(*overlap).rio.reproject_match(g_clip, resampling=5)
    r_clip = r_clip.chunk({"x": 4096, "y": 4096})

    valid_mask = (g_clip != 0) & (r_clip != 0) & g_clip.notnull() & r_clip.notnull()
    agreement_map = xr.where(valid_mask, (g_clip == r_clip).astype("int8"), -1)

    n_agree = (agreement_map == 1).sum()
    n_valid = valid_mask.sum()
    return dict(glance=g_clip, ref=r_clip, mask=valid_mask, agree=agreement_map,
                n_agree=n_agree, n_valid=n_valid)


# =============================================================================
# 6. CONFUSION MATRIX (CHUNK REDUCE)
# =============================================================================
def compute_confusion_matrix_lazy(glance_arr, ref_arr, valid_mask):
    """Chunk-wise reduction for 2x2 confusion matrix."""
    mask = valid_mask & glance_arr.notnull() & ref_arr.notnull()
    tn = ((ref_arr == 0) & (glance_arr == 0) & mask).sum()
    fp = ((ref_arr == 0) & (glance_arr == 1) & mask).sum()
    fn = ((ref_arr == 1) & (glance_arr == 0) & mask).sum()
    tp = ((ref_arr == 1) & (glance_arr == 1) & mask).sum()
    tn, fp, fn, tp = dask.compute(tn, fp, fn, tp)
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    print(f"✓ Confusion matrix computed lazily | total={cm.sum():,}")
    return cm


# =============================================================================
# 7. METRICS + VISUALIZATION
# =============================================================================
def compute_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    overall = (tp + tn) / total
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    kappa = ((tp + tn) / total - (((tp + fp) * (tp + fn) +
                                   (tn + fp) * (tn + fn)) / total ** 2)) / \
            (1 - (((tp + fp) * (tp + fn) +
                   (tn + fp) * (tn + fn)) / total ** 2))
    return dict(overall=overall, precision=precision, recall=recall, f1=f1, kappa=kappa)


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-forest", "Forest"],
                yticklabels=["Non-forest", "Forest"])
    ax.set_xlabel("Predicted (GLANCE)")
    ax.set_ylabel("Reference (MapBiomas)")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved confusion matrix → {path}")


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    year = 2016
    folder_glance = "/projectnb/measures/products/SA/v001/DAAC/LC/"
    folder_mapbiomas = "/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/"
    outdir = f"/projectnb/modislc/users/chishan/data/forest_comparison_{year}"
    os.makedirs(outdir, exist_ok=True)

    print(f"\n=== GLANCE vs MapBiomas Forest Comparison {year} ===")

    # 1. Legend
    qml = os.path.join(folder_mapbiomas, f"AMZ.{year}.M.qml")
    classes = parse_qml_legend(qml)
    reclass_map = create_mapbiomas_forest_map(classes)

    # 2. MapBiomas
    tif = os.path.join(folder_mapbiomas, f"AMZ.{year}.M.tif")
    mapb = rioxarray.open_rasterio(tif, chunks={"x": 4096, "y": 4096}, lock=False)
    mapb_bin = reclassify_to_binary_forest(mapb, reclass_map)

    # 3. GLANCE
    files = find_glance_tiles_by_year(folder_glance, year, "SA")
    glance = load_glance_tiles_lazy(files)
    glance_bin = reclassify_glance_to_forest(glance)

    # 4. Align + compare
    cmp = align_and_compare_maps(glance_bin, mapb_bin)

    # 5. Confusion + metrics
    cm = compute_confusion_matrix_lazy(cmp["glance"], cmp["ref"], cmp["mask"])
    metrics = compute_metrics(cm)
    print(metrics)

    # 6. Save
    cmp["agree"].rio.to_raster(os.path.join(outdir, f"agreement_map_{year}.tif"),
                               compress="LZW", tiled=True,
                               blockxsize=512, blockysize=512, BIGTIFF="YES")
    plot_confusion_matrix(cm, os.path.join(outdir, f"confmat_{year}.png"))

    print(f"✓ DONE → {outdir}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("Initializing Dask cluster...")
    cluster = LocalCluster(n_workers=1, threads_per_worker=16,
                           memory_limit="128GB", dashboard_address=":8787")
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    main()

    client.close()
    cluster.close()