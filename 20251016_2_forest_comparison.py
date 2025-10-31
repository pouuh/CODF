"""
GLANCE vs MapBiomas Amazon Forest Comparison (Lazy + HPC-Optimized)
-------------------------------------------------------------------
- Loads ONLY GLANCE tiles overlapping MapBiomas extent for the year
- Clips each tile to MapBiomas bbox before merging (low memory)
- Fully lazy xarray+dask; chunk-safe reclassification
"""

import os
import glob
import numpy as np
import xarray as xr
import rioxarray
import rasterio
import dask
from dask.distributed import Client, LocalCluster
# add import near the top
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import re
import geopandas as gpd
from shapely.geometry import box

# -----------------------------
# Dask safety knobs
# -----------------------------
dask.config.set({
    "array.slicing.split_large_chunks": True,
    "optimization.fuse.active": True,
    "array.chunk-size": "128 MiB",              # keep chunks modest
    "distributed.worker.memory.target": 0.6,    # start spilling sooner
    "distributed.worker.memory.spill": 0.7,
    "distributed.worker.memory.pause": 0.8,
})

from rasterio.crs import CRS

TILE_CRS_WKT = """PROJCS["BU MEaSUREs Lambert Azimuthal Equal Area - SA - V01",
GEOGCS["GCS_WGS_1984",
    DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],
    PRIMEM["Greenwich",0.0],UNIT["degree",0.0174532925199433]],
PROJECTION["Lambert_Azimuthal_Equal_Area"],
PARAMETER["false_easting",0.0],
PARAMETER["false_northing",0.0],
PARAMETER["longitude_of_center",-60],
PARAMETER["latitude_of_center",-15],
UNIT["meter",1.0]]"""

TILE_CRS = CRS.from_wkt(TILE_CRS_WKT)

def reclassify_to_binary_forest(da, forest_values=(1,2,9), nodata_value=255):
    arr = da.squeeze(drop=True)
    if arr.chunks is None:
        arr = arr.chunk({"x": 2048, "y": 2048})
    valid  = arr.notnull()
    forest = arr.isin(list(forest_values))
    out = xr.where(valid, xr.where(forest, 1, 0), nodata_value).astype("uint8")
    if hasattr(arr, "rio"):
        out.rio.write_crs(arr.rio.crs, inplace=True)
        out.rio.write_nodata(nodata_value, inplace=True)
    return out

# =============================================================================
# 2) MAPBIOMAS LOADING (and bounds)
# =============================================================================
def load_mapbiomas_year(tif_path, chunks={"x": 2048, "y": 2048}):
    mb = rioxarray.open_rasterio(tif_path, chunks=chunks, lock=True)  # 2D [band,y,x] expected
    # ensure one band semantics
    if "band" in mb.dims and mb.sizes["band"] == 1:
        mb = mb.isel(band=0, drop=True)
    return mb

def mapbiomas_bounds(mb_da: xr.DataArray):
    """Return (minx, miny, maxx, maxy) in the CRS of MapBiomas."""
    return mb_da.rio.bounds()


# =============================================================================
# 3) GLANCE TILE DISCOVERY + FILTER BY BBOX
# =============================================================================
def list_glance_tiles(folder_glance, year, region="SA"):
    pattern = os.path.join(folder_glance, f"GLANCE.A{year}0701.h*v*.001.*.{region}.LC.tif")
    return sorted(glob.glob(pattern))

def _tif_bounds(path):
    with rasterio.Env():
        with rasterio.open(path) as ds:
            return ds.bounds, ds.crs
def _is_geographic(crs_obj):
    try:
        return CRS.from_user_input(crs_obj).is_geographic
    except Exception:
        return False

def _bboxes_intersect(a, b):
    # a,b are rasterio BoundingBox (left,bottom,right,top)
    return not (a.right <= b.left or a.left >= b.right or
                a.top   <= b.bottom or a.bottom >= b.top)

def filter_tiles_by_intersection(files, target_bounds, target_crs):
    """
    Return only GLANCE tiles whose bbox intersects the target bbox (MapBiomas).
    If the output CRS is geographic, use densify_pts >= 2 (21) to satisfy GDAL.
    """
    selected = []
    minx, miny, maxx, maxy = target_bounds
    target_bbox = rasterio.coords.BoundingBox(minx, miny, maxx, maxy)
    target_is_geo = _is_geographic(target_crs)

    for f in files:
        bnds, crs = _tif_bounds(f)

        if crs == target_crs:
            tb = bnds
        else:
            # densify only when transforming *to* geographic
            densify = 21 if target_is_geo else 0
            tb_tuple = transform_bounds(crs, target_crs,
                                        bnds.left, bnds.bottom, bnds.right, bnds.top,
                                        densify_pts=densify)
            tb = rasterio.coords.BoundingBox(*tb_tuple)

        if _bboxes_intersect(tb, target_bbox):
            selected.append(f)

    return selected

import re, glob, os
import geopandas as gpd
import rasterio
from shapely.geometry import box
from rasterio.crs import CRS

def _pick_one_glance_file(folder_glance, year, region="SA"):
    pattern = os.path.join(folder_glance, f"GLANCE.A{year}0701.h*v*.001.*.{region}.LC.tif")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No GLANCE files found: {pattern}")
    return files[0], files

def _read_glance_crs(sample_tif):
    with rasterio.open(sample_tif) as ds:
        return ds.crs

def _looks_like_meters_bounds(bounds):
    # 粗略判断：如果坐标绝对值动辄上百万，基本就是投影米，而不是经纬度
    minx, miny, maxx, maxy = bounds
    return max(abs(minx), abs(miny), abs(maxx), abs(maxy)) > 1000.0

import re, glob, os
import geopandas as gpd
from shapely.geometry import box

def select_glance_tiles_from_geojson_fixed(json_file, mb_bounds, mb_crs, folder_glance, year, region="SA"):
    """
    Use the provided GLANCE tile CRS (LAEA SA V01) to fix the GeoJSON,
    then reproject to MapBiomas CRS and intersect by bbox.
    """
    # 1) read + fix CRS
    gdf = gpd.read_file(json_file)
    gdf = gdf.set_crs(TILE_CRS, allow_override=True)  # <- critical
    # quick sanity print
    print("Tile index (fixed) bounds in meters:", gdf.total_bounds)

    # 2) project tiles to MapBiomas CRS (EPSG:4674)
    gdf_mb = gdf.to_crs(mb_crs)

    # 3) intersect with MapBiomas bbox
    minx, miny, maxx, maxy = mb_bounds
    bbox = box(minx, miny, maxx, maxy)
    gsel = gdf_mb[gdf_mb.intersects(bbox)]
    if gsel.empty:
        print("⚠ No tiles intersect MapBiomas bbox.")
        return []

    # 4) match file names by tileID (e.g., 'h46v4')
    pattern = os.path.join(folder_glance, f"GLANCE.A{year}0701.h*v*.001.*.{region}.LC.tif")
    all_files = sorted(glob.glob(pattern))
    tile_ids = set(gsel["tileID"].astype(str))

    selected = [f for f in all_files if (m := re.search(r'h\d+v\d+', os.path.basename(f))) and m.group(0) in tile_ids]
    print(f"✓ GeoJSON-selected GLANCE tiles: {len(selected)} / {len(all_files)}")
    return selected

# =============================================================================
# 4) LOAD + CLIP GLANCE LAZILY (PER TILE) THEN MERGE
# =============================================================================
def load_glance_overlap_mosaic(files, clip_bounds, target_crs, chunks={"x": 2048, "y": 2048}):
    """Open only overlapping tiles, clip each to bounds, then merge lazily."""
    from rioxarray.merge import merge_arrays
    clipped = []
    minx, miny, maxx, maxy = clip_bounds

    for f in files:
        da = rioxarray.open_rasterio(f, chunks=chunks, lock=True)
        # ensure 2D
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.isel(band=0, drop=True)

        # Reproject *bounds* to this tile CRS for precise clip
        if da.rio.crs != target_crs:
            # cheaper: reproject MapBiomas bounds to tile CRS
            from rasterio.warp import transform_bounds
            tb = transform_bounds(target_crs, da.rio.crs, minx, miny, maxx, maxy, densify_pts=0)
            minx_t, miny_t, maxx_t, maxy_t = tb
        else:
            minx_t, miny_t, maxx_t, maxy_t = minx, miny, maxx, maxy

        # Clip each tile to the overlap; if no overlap, skip
        try:
            da_clip = da.rio.clip_box(minx_t, miny_t, maxx_t, maxy_t)
            if da_clip.size > 0:
                clipped.append(da_clip)
        except Exception:
            continue

    if not clipped:
        raise RuntimeError("No GLANCE tiles overlap MapBiomas extent.")

    # Merge clipped tiles (low memory since each is already cropped)
    mosaic = merge_arrays(clipped, nodata=0)
    # restore 2D if merge_arrays reintroduces band
    if "band" in mosaic.dims and mosaic.sizes["band"] == 1:
        mosaic = mosaic.isel(band=0, drop=True)
    return mosaic

from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays  # still imported but we won't use it here

import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

def _bbox_intersection(a, b):
    ixmin, iymin = max(a[0], b[0]), max(a[1], b[1])
    ixmax, iymax = min(a[2], b[2]), min(a[3], b[3])
    return (ixmin, iymin, ixmax, iymax) if (ixmax > ixmin and iymax > iymin) else None

def load_glance_reprojected_to_mapbiomas(sel_files, mapb_template, chunks={"x": 2048, "y": 2048}):
    out_tiles = []
    mb_crs = mapb_template.rio.crs
    mb_shape = (mapb_template.sizes["y"], mapb_template.sizes["x"])
    mb_transform = mapb_template.rio.transform()
    mb_bounds = mapb_template.rio.bounds()

    for f in sel_files:
        da = rioxarray.open_rasterio(f, chunks=chunks, lock=True)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.isel(band=0, drop=True)
        if da.chunks is None:
            da = da.chunk(chunks)

        # tile CRS should be TILE_CRS; if not, treat it as such (rare)
        if da.rio.crs != TILE_CRS:
            da = da.rio.write_crs(TILE_CRS, inplace=True)

        tile_bounds = da.rio.bounds()

        # transform MapBiomas bbox into tile CRS (projected meters → densify_pts=0)
        mb_in_tile = transform_bounds(mb_crs, TILE_CRS, *mb_bounds, densify_pts=0)

        inter = _bbox_intersection(tile_bounds, mb_in_tile)
        if inter is None:
            continue

        try:
            da_clip = da.rio.clip_box(*inter)
        except Exception:
            continue

        da_mb = da_clip.rio.reproject(
            dst_crs=mb_crs,
            shape=mb_shape,
            transform=mb_transform,
            resampling=Resampling.nearest,
            nodata=0,
        )
        if da_mb.chunks is None:
            da_mb = da_mb.chunk(chunks)

        out_tiles.append(da_mb)

    if not out_tiles:
        raise RuntimeError("No GLANCE tiles after clip & reproject; check CRS and bbox.")

    gl_stack = xr.concat(out_tiles, dim="tile")
    gl_proj = gl_stack.max(dim="tile")   # prefer non-zero over nodata=0
    return gl_proj

def align_and_compare_maps(glance_bin_on_mb, mapb_bin):
    """Both are on the SAME grid (MapBiomas). No reproject_match, no giant mosaic."""
    # ensure same chunking
    if glance_bin_on_mb.chunks is None:
        glance_bin_on_mb = glance_bin_on_mb.chunk({"x": 2048, "y": 2048})
    if mapb_bin.chunks is None:
        mapb_bin = mapb_bin.chunk({"x": 2048, "y": 2048})

    valid = (glance_bin_on_mb != 255) & (mapb_bin != 255) & glance_bin_on_mb.notnull() & mapb_bin.notnull()
    # we DO NOT build a full agreement raster unless requested; compute confusion directly
    return glance_bin_on_mb, mapb_bin, valid

# =============================================================================
# 6) CONFUSION + METRICS
# =============================================================================
def compute_confusion_matrix_lazy(glance_arr, ref_arr, valid_mask):
    mask = valid_mask
    tn = ((ref_arr == 0) & (glance_arr == 0) & mask).sum()
    fp = ((ref_arr == 0) & (glance_arr == 1) & mask).sum()
    fn = ((ref_arr == 1) & (glance_arr == 0) & mask).sum()
    tp = ((ref_arr == 1) & (glance_arr == 1) & mask).sum()
    tn, fp, fn, tp = dask.compute(tn, fp, fn, tp)
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)

def compute_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum() if cm.sum() else 1
    overall = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    kappa_num = (tp + tn) / total - (((tp + fp)*(tp + fn) + (tn + fp)*(tn + fn)) / total**2)
    kappa_den = 1 - (((tp + fp)*(tp + fn) + (tn + fp)*(tn + fn)) / total**2)
    kappa = (kappa_num / kappa_den) if kappa_den else 0.0
    return dict(overall=overall, precision=precision, recall=recall, f1=f1, kappa=kappa)


# =============================================================================
# 7) MAIN
# =============================================================================
def main():
    year = 2016
    folder_glance = "/projectnb/measures/products/SA/v001/DAAC/LC/"
    folder_mapbiomas = "/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/"
    outdir = f"/projectnb/modislc/users/chishan/data/forest_comparison_{year}"
    os.makedirs(outdir, exist_ok=True)

    print(f"\n=== GLANCE vs MapBiomas Forest Comparison {year} ===")
        # --- MapBiomas
    tif_mb = os.path.join(folder_mapbiomas, f"AMZ.{year}.M.tif")
    mb = load_mapbiomas_year(tif_mb, chunks={"x": 2048, "y": 2048})
    mb_bin = reclassify_to_binary_forest(mb, forest_values=[1,2,9], nodata_value=255)
    mb_bounds = mb_bin.rio.bounds()
    mb_crs = mb_bin.rio.crs

    # --- GLANCE tiles from GeoJSON (already have your function)
    json_file = "/projectnb/measures/datasets/Grids/FILTERED/json/GLANCE_V01_AS_PROJ_TILE_FILTERED.geojson"
    # sel_tiles = select_glance_tiles_from_geojson(json_file, mb_bounds, mb_crs, folder_glance, year, "SA")
    sel_tiles = select_glance_tiles_from_geojson_fixed(
        json_file="/projectnb/measures/datasets/Grids/FILTERED/json/GLANCE_V01_AS_PROJ_TILE_FILTERED.geojson",
        mb_bounds=mb_bounds,
        mb_crs=mb_crs,
        folder_glance=folder_glance,
        year=year,
        region="SA",
    )

    print("=== CRS diagnostics ===")
    print("MapBiomas:", mb_crs, "bounds:", mb_bounds, "(degrees)")
    print("Tile index (fixed crs):", TILE_CRS, "bounds:", gpd.read_file(json_file).set_crs(TILE_CRS, allow_override=True).total_bounds, "(meters)")

    if not sel_tiles:
        raise RuntimeError("No GLANCE tiles overlap MapBiomas extent per GeoJSON.")

    # --- Reproject each tile directly to the MapBiomas grid; then binary
    gl_mb = load_glance_reprojected_to_mapbiomas(sel_tiles, mapb_template=mb_bin, chunks={"x": 2048, "y": 2048})
    gl_mb_bin = reclassify_to_binary_forest(gl_mb, forest_values=[5], nodata_value=255)

    # --- Compare on SAME grid (no huge arrays)
    g, r, mask = align_and_compare_maps(gl_mb_bin, mb_bin)

    # --- Confusion + metrics (no agreement raster)
    cm = compute_confusion_matrix_lazy(g, r, mask)
    metrics = compute_metrics(cm)
    print("✓ Metrics:", metrics)

if __name__ == "__main__":
    print("Initializing Dask cluster...")
    cluster = LocalCluster(
        n_workers=1, threads_per_worker=16,
        memory_limit="128GB", dashboard_address=":8787"
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    try:
        main()
    finally:
        client.close()
        cluster.close()
