"""
GLANCE vs MapBiomas Amazon Forest Comparison
Memory-efficient implementation using block-wise processing
"""
import os
import glob
import numpy as np
import xarray as xr
import rioxarray
import rasterio
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import re
import geopandas as gpd
from shapely.geometry import box
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# Conservative Dask settings for 128GB memory
dask.config.set({
    "array.slicing.split_large_chunks": True,
    "optimization.fuse.active": True,
    "array.chunk-size": "64 MiB",  # Smaller chunks
    "distributed.worker.memory.target": 0.50,
    "distributed.worker.memory.spill": 0.60,
    "distributed.worker.memory.pause": 0.75,
    "distributed.worker.memory.terminate": 0.95,
})

# GLANCE tile CRS definition
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

# =============================================================================
# UTILITIES
# =============================================================================
def get_optimal_chunks(shape: Tuple[int, int], max_chunk_mb: int = 64) -> dict:
    """Calculate optimal chunk size based on array shape and memory limit."""
    ny, nx = shape
    bytes_per_pixel = 1  # uint8
    max_chunk_bytes = max_chunk_mb * 1024 * 1024
    
    # Start with square chunks
    chunk_size = int(np.sqrt(max_chunk_bytes / bytes_per_pixel))
    chunk_y = min(chunk_size, ny)
    chunk_x = min(chunk_size, nx)
    
    return {"y": chunk_y, "x": chunk_x}


def reclassify_to_binary_forest(da: xr.DataArray, 
                                 forest_values: Tuple[int, ...], 
                                 nodata_value: int = 255) -> xr.DataArray:
    """Reclassify to binary forest map (1=forest, 0=non-forest, 255=nodata)."""
    arr = da.squeeze(drop=True)
    
    # Ensure chunked
    if arr.chunks is None:
        chunks = get_optimal_chunks(arr.shape)
        arr = arr.chunk(chunks)
    
    # Vectorized binary classification
    valid = arr.notnull()
    forest = arr.isin(list(forest_values))
    out = xr.where(valid, xr.where(forest, 1, 0), nodata_value).astype("uint8")
    
    # Preserve CRS metadata
    if hasattr(arr, "rio"):
        out.rio.write_crs(arr.rio.crs, inplace=True)
        out.rio.write_nodata(nodata_value, inplace=True)
    
    return out


# =============================================================================
# MAPBIOMAS OPERATIONS
# =============================================================================
def load_mapbiomas_year(tif_path: str, chunks: dict = None) -> xr.DataArray:
    """Load MapBiomas with automatic optimal chunking."""
    with rasterio.open(tif_path) as src:
        shape = (src.height, src.width)
    
    if chunks is None:
        chunks = get_optimal_chunks(shape, max_chunk_mb=64)
    
    print(f"Loading MapBiomas with chunks: {chunks}")
    mb = rioxarray.open_rasterio(tif_path, chunks=chunks, lock=False)
    
    if "band" in mb.dims and mb.sizes["band"] == 1:
        mb = mb.isel(band=0, drop=True)
    
    return mb


# =============================================================================
# GLANCE TILE SELECTION
# =============================================================================
def select_glance_tiles_from_geojson(
    json_file: str,
    mb_bounds: Tuple[float, float, float, float],
    mb_crs: CRS,
    folder_glance: str,
    year: int,
    region: str = "SA"
) -> List[str]:
    """Select GLANCE tiles that intersect MapBiomas extent."""
    # Read and fix tile index CRS
    gdf = gpd.read_file(json_file)
    gdf = gdf.set_crs(TILE_CRS, allow_override=True)
    print(f"Tile index bounds (meters): {gdf.total_bounds}")
    
    # Reproject to MapBiomas CRS
    gdf_mb = gdf.to_crs(mb_crs)
    
    # Spatial filter
    minx, miny, maxx, maxy = mb_bounds
    bbox = box(minx, miny, maxx, maxy)
    gsel = gdf_mb[gdf_mb.intersects(bbox)]
    
    if gsel.empty:
        raise RuntimeError("No GLANCE tiles intersect MapBiomas extent")
    
    print(f"Found {len(gsel)} intersecting tiles")
    
    # Match files by tile ID
    pattern = os.path.join(folder_glance, f"GLANCE.A{year}0701.h*v*.001.*.{region}.LC.tif")
    all_files = sorted(glob.glob(pattern))
    tile_ids = set(gsel["tileID"].astype(str))
    
    selected = [
        f for f in all_files 
        if (m := re.search(r'h\d+v\d+', os.path.basename(f))) 
        and m.group(0) in tile_ids
    ]
    
    print(f"Selected {len(selected)} GLANCE tiles")
    return selected


# =============================================================================
# BLOCK-WISE CONFUSION MATRIX (KEY OPTIMIZATION)
# =============================================================================
def compute_confusion_blockwise(
    glance_arr: xr.DataArray,
    mapb_arr: xr.DataArray,
    valid_mask: xr.DataArray,
    block_size: int = 4096
) -> np.ndarray:
    """
    Compute confusion matrix in blocks to avoid memory explosion.
    This is the critical fix for your memory error.
    """
    print("Computing confusion matrix block-wise...")
    
    ny, nx = glance_arr.shape
    
    # Initialize confusion matrix
    cm_total = np.zeros((2, 2), dtype=np.int64)
    
    # Process in spatial blocks
    n_blocks_y = int(np.ceil(ny / block_size))
    n_blocks_x = int(np.ceil(nx / block_size))
    total_blocks = n_blocks_y * n_blocks_x
    
    print(f"Processing {total_blocks} blocks ({n_blocks_y}×{n_blocks_x})...")
    
    for iy in range(n_blocks_y):
        y_start = iy * block_size
        y_end = min((iy + 1) * block_size, ny)
        
        for ix in range(n_blocks_x):
            x_start = ix * block_size
            x_end = min((ix + 1) * block_size, nx)
            
            # Extract block
            g_block = glance_arr.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            m_block = mapb_arr.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            v_block = valid_mask.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            
            # Compute to memory (small block)
            g_np = g_block.compute()
            m_np = m_block.compute()
            v_np = v_block.compute()
            
            # Confusion for this block
            mask = v_np.values if hasattr(v_np, 'values') else v_np
            g_vals = g_np.values if hasattr(g_np, 'values') else g_np
            m_vals = m_np.values if hasattr(m_np, 'values') else m_np
            
            tn = np.sum((m_vals == 0) & (g_vals == 0) & mask)
            fp = np.sum((m_vals == 0) & (g_vals == 1) & mask)
            fn = np.sum((m_vals == 1) & (g_vals == 0) & mask)
            tp = np.sum((m_vals == 1) & (g_vals == 1) & mask)
            
            cm_total += np.array([[tn, fp], [fn, tp]], dtype=np.int64)
            
            if (iy * n_blocks_x + ix + 1) % 10 == 0:
                print(f"  Processed {iy * n_blocks_x + ix + 1}/{total_blocks} blocks")
    
    return cm_total


# =============================================================================
# GLANCE REPROJECTION (TILE-BY-TILE)
# =============================================================================
def reproject_glance_tile(
    tile_path: str,
    target_crs: CRS,
    target_bounds: Tuple[float, float, float, float],
    target_shape: Tuple[int, int],
    target_transform,
    chunks: dict
) -> xr.DataArray:
    """Reproject single GLANCE tile with memory safety."""
    # Load tile
    da = rioxarray.open_rasterio(tile_path, chunks=chunks, lock=False)
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.isel(band=0, drop=True)
    
    # Ensure tile CRS is set
    if da.rio.crs != TILE_CRS:
        da = da.rio.write_crs(TILE_CRS, inplace=True)
    
    # Transform MapBiomas bounds to tile CRS for clipping
    tile_bounds = da.rio.bounds()
    mb_in_tile = transform_bounds(target_crs, TILE_CRS, *target_bounds, densify_pts=0)
    
    # Check intersection
    ixmin = max(tile_bounds[0], mb_in_tile[0])
    iymin = max(tile_bounds[1], mb_in_tile[1])
    ixmax = min(tile_bounds[2], mb_in_tile[2])
    iymax = min(tile_bounds[3], mb_in_tile[3])
    
    if ixmax <= ixmin or iymax <= iymin:
        return None
    
    # Clip to intersection
    da_clip = da.rio.clip_box(ixmin, iymin, ixmax, iymax)
    
    # Reproject to MapBiomas grid
    da_reproj = da_clip.rio.reproject(
        dst_crs=target_crs,
        shape=target_shape,
        transform=target_transform,
        resampling=Resampling.nearest,
        nodata=0
    )
    
    return da_reproj


def load_glance_reprojected(
    sel_files: List[str],
    mapb_template: xr.DataArray,
    chunks: dict
) -> xr.DataArray:
    """Load and reproject GLANCE tiles efficiently."""
    print(f"Reprojecting {len(sel_files)} GLANCE tiles...")
    
    mb_crs = mapb_template.rio.crs
    mb_shape = (mapb_template.sizes["y"], mapb_template.sizes["x"])
    mb_transform = mapb_template.rio.transform()
    mb_bounds = mapb_template.rio.bounds()
    
    reprojected = []
    for i, f in enumerate(sel_files):
        print(f"  Processing tile {i+1}/{len(sel_files)}: {os.path.basename(f)}")
        
        tile_reproj = reproject_glance_tile(
            f, mb_crs, mb_bounds, mb_shape, mb_transform, chunks
        )
        
        if tile_reproj is not None:
            reprojected.append(tile_reproj)
    
    if not reprojected:
        raise RuntimeError("No GLANCE tiles after reprojection")
    
    print(f"Successfully reprojected {len(reprojected)} tiles")
    
    # Mosaic: take maximum (prefer non-zero over nodata=0)
    gl_stack = xr.concat(reprojected, dim="tile")
    gl_mosaic = gl_stack.max(dim="tile")
    
    return gl_mosaic


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(cm: np.ndarray) -> dict:
    """Calculate accuracy metrics from confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum() or 1
    
    overall = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    # Cohen's Kappa
    po = overall
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total ** 2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0
    
    return {
        "overall_accuracy": overall,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "kappa": kappa,
        "confusion_matrix": cm.tolist()
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    year = 2016
    folder_glance = "/projectnb/measures/products/SA/v001/DAAC/LC/"
    folder_mapbiomas = "/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/"
    json_file = "/projectnb/measures/datasets/Grids/FILTERED/json/GLANCE_V01_AS_PROJ_TILE_FILTERED.geojson"
    outdir = f"/projectnb/modislc/users/chishan/data/forest_comparison_{year}"
    os.makedirs(outdir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GLANCE vs MapBiomas Forest Comparison {year}")
    print(f"{'='*60}\n")
    
    # -------------------------------------------------------------------------
    # 1. Load MapBiomas
    # -------------------------------------------------------------------------
    print("Step 1: Loading MapBiomas...")
    tif_mb = os.path.join(folder_mapbiomas, f"AMZ.{year}.M.tif")
    mb = load_mapbiomas_year(tif_mb)
    mb_bin = reclassify_to_binary_forest(mb, forest_values=(1, 2, 9), nodata_value=255)
    
    mb_bounds = mb_bin.rio.bounds()
    mb_crs = mb_bin.rio.crs
    print(f"  MapBiomas shape: {mb_bin.shape}")
    print(f"  MapBiomas CRS: {mb_crs}")
    print(f"  MapBiomas bounds: {mb_bounds}\n")
    
    # -------------------------------------------------------------------------
    # 2. Select GLANCE tiles
    # -------------------------------------------------------------------------
    print("Step 2: Selecting GLANCE tiles...")
    sel_tiles = select_glance_tiles_from_geojson(
        json_file, mb_bounds, mb_crs, folder_glance, year, "SA"
    )
    print()
    
    # -------------------------------------------------------------------------
    # 3. Reproject GLANCE
    # -------------------------------------------------------------------------
    print("Step 3: Reprojecting GLANCE to MapBiomas grid...")
    chunks = get_optimal_chunks(mb_bin.shape, max_chunk_mb=64)
    gl_mb = load_glance_reprojected(sel_tiles, mb_bin, chunks)
    gl_mb_bin = reclassify_to_binary_forest(gl_mb, forest_values=(5,), nodata_value=255)
    print()
    
    # -------------------------------------------------------------------------
    # 4. Compute confusion matrix (block-wise)
    # -------------------------------------------------------------------------
    print("Step 4: Computing confusion matrix...")
    valid_mask = (
        (gl_mb_bin != 255) & 
        (mb_bin != 255) & 
        gl_mb_bin.notnull() & 
        mb_bin.notnull()
    )
    
    # CRITICAL: Use block-wise computation
    cm = compute_confusion_blockwise(gl_mb_bin, mb_bin, valid_mask, block_size=4096)
    
    # -------------------------------------------------------------------------
    # 5. Calculate metrics
    # -------------------------------------------------------------------------
    print("\nStep 5: Computing accuracy metrics...")
    metrics = compute_metrics(cm)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\nAccuracy Metrics:")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  F1 Score:         {metrics['f1_score']:.4f}")
    print(f"  Cohen's Kappa:    {metrics['kappa']:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    import json
    results_file = os.path.join(outdir, f"metrics_{year}.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    print("Initializing Dask cluster...")
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=16,
        memory_limit="100GB",  # Leave headroom
        dashboard_address=":8787"
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}\n")
    
    try:
        main()
    finally:
        print("\nShutting down Dask cluster...")
        client.close()
        cluster.close()