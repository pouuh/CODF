"""
Script 1: Generate Forest Comparison Raster
将MapBiomas投影到GLANCE坐标系，生成森林对比影像
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
from rasterio.warp import transform_bounds, calculate_default_transform
import re
import geopandas as gpd
from shapely.geometry import box
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
dask.config.set({
    "array.slicing.split_large_chunks": True,
    "optimization.fuse.active": True,
    "array.chunk-size": "64 MiB",
    "distributed.worker.memory.target": 0.50,
    "distributed.worker.memory.spill": 0.60,
    "distributed.worker.memory.pause": 0.75,
    "distributed.worker.memory.terminate": 0.95,
})

# GLANCE LAEA projection (目标投影)
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
    """Calculate optimal chunk size."""
    ny, nx = shape
    bytes_per_pixel = 1  # uint8
    max_chunk_bytes = max_chunk_mb * 1024 * 1024
    
    chunk_size = int(np.sqrt(max_chunk_bytes / bytes_per_pixel))
    chunk_y = min(chunk_size, ny)
    chunk_x = min(chunk_size, nx)
    
    return {"y": chunk_y, "x": chunk_x}


def reclassify_to_binary_forest(da: xr.DataArray, 
                                 forest_values: Tuple[int, ...], 
                                 nodata_value: int = 255) -> xr.DataArray:
    """Reclassify to binary forest (1=forest, 0=non-forest, 255=nodata)."""
    arr = da.squeeze(drop=True)
    
    if arr.chunks is None:
        chunks = get_optimal_chunks(arr.shape)
        arr = arr.chunk(chunks)
    
    valid = arr.notnull()
    forest = arr.isin(list(forest_values))  # ✅ 使用arr对象调用isin方法
    out = xr.where(valid, xr.where(forest, 1, 0), nodata_value).astype("uint8")
    
    if hasattr(arr, "rio"):
        out.rio.write_crs(arr.rio.crs, inplace=True)
        out.rio.write_nodata(nodata_value, inplace=True)
    
    return out


# =============================================================================
# MAPBIOMAS OPERATIONS
# =============================================================================
def load_mapbiomas_year(tif_path: str, chunks: dict = None) -> xr.DataArray:
    """Load MapBiomas with optimal chunking."""
    with rasterio.open(tif_path) as src:
        shape = (src.height, src.width)
    
    if chunks is None:
        chunks = get_optimal_chunks(shape, max_chunk_mb=64)
    
    print(f"  Loading MapBiomas with chunks: {chunks}")
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
    """Select GLANCE tiles intersecting MapBiomas extent."""
    gdf = gpd.read_file(json_file)
    gdf = gdf.set_crs(TILE_CRS, allow_override=True)
    
    # Transform MapBiomas bounds to GLANCE CRS
    mb_in_glance = transform_bounds(mb_crs, TILE_CRS, *mb_bounds, densify_pts=21)
    bbox_glance = box(*mb_in_glance)
    
    gsel = gdf[gdf.intersects(bbox_glance)]
    
    if gsel.empty:
        raise RuntimeError("No GLANCE tiles intersect MapBiomas extent")
    
    print(f"  Found {len(gsel)} intersecting tiles")
    
    # Match files
    pattern = os.path.join(folder_glance, f"GLANCE.A{year}0701.h*v*.001.*.{region}.LC.tif")
    all_files = sorted(glob.glob(pattern))
    tile_ids = set(gsel["tileID"].astype(str))
    
    selected = [
        f for f in all_files 
        if (m := re.search(r'h\d+v\d+', os.path.basename(f))) 
        and m.group(0) in tile_ids
    ]
    
    print(f"  Selected {len(selected)} GLANCE tiles")
    return selected


# =============================================================================
# GLANCE MOSAIC
# =============================================================================
def load_glance_tiles(tile_files: List[str], chunks: dict) -> xr.DataArray:
    """Load and mosaic GLANCE tiles in native projection."""
    print(f"  Loading {len(tile_files)} GLANCE tiles...")
    
    tiles = []
    for i, f in enumerate(tile_files):
        da = rioxarray.open_rasterio(f, chunks=chunks, lock=False)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.isel(band=0, drop=True)
        
        # Ensure CRS is set
        if da.rio.crs != TILE_CRS:
            da = da.rio.write_crs(TILE_CRS, inplace=True)
        
        tiles.append(da)
        
        if (i + 1) % 5 == 0:
            print(f"    Loaded {i+1}/{len(tile_files)} tiles")
    
    # Merge tiles
    from rioxarray.merge import merge_arrays
    print("  Merging GLANCE tiles...")
    mosaic = merge_arrays(tiles, nodata=0)
    
    if "band" in mosaic.dims and mosaic.sizes["band"] == 1:
        mosaic = mosaic.isel(band=0, drop=True)
    
    return mosaic


# =============================================================================
# REPROJECT MAPBIOMAS TO GLANCE GRID
# =============================================================================
def reproject_mapbiomas_to_glance(
    mb_bin: xr.DataArray,
    glance_template: xr.DataArray
) -> xr.DataArray:
    """Reproject MapBiomas to match GLANCE grid."""
    print("  Reprojecting MapBiomas to GLANCE grid (LAEA)...")
    
    target_crs = glance_template.rio.crs
    target_transform = glance_template.rio.transform()
    target_shape = (glance_template.sizes["y"], glance_template.sizes["x"])
    
    mb_reproj = mb_bin.rio.reproject(
        dst_crs=target_crs,
        shape=target_shape,
        transform=target_transform,
        resampling=Resampling.nearest,
        nodata=255
    )
    
    print(f"    Reprojected shape: {mb_reproj.shape}")
    return mb_reproj


# =============================================================================
# GENERATE COMPARISON RASTER
# =============================================================================
def create_comparison_raster(
    glance_bin: xr.DataArray,
    mapb_bin: xr.DataArray,
    nodata: int = 255
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Create stacked comparison raster:
    Band 1: GLANCE forest (1=forest, 0=non-forest, 255=nodata)
    Band 2: MapBiomas forest (1=forest, 0=non-forest, 255=nodata)
    
    Returns: stacked DataArray and valid mask
    """
    print("  Creating comparison raster...")
    
    # Ensure same chunking
    if glance_bin.chunks is None:
        glance_bin = glance_bin.chunk({"x": 4096, "y": 4096})
    if mapb_bin.chunks is None:
        mapb_bin = mapb_bin.chunk({"x": 4096, "y": 4096})
    
    # Stack bands
    stacked = xr.concat([glance_bin, mapb_bin], dim="band")
    stacked = stacked.assign_coords(band=["glance_forest", "mapbiomas_forest"])
    
    # Valid mask (both datasets have valid data)
    valid_mask = (
        (glance_bin != nodata) & 
        (mapb_bin != nodata) & 
        glance_bin.notnull() & 
        mapb_bin.notnull()
    )
    
    return stacked, valid_mask


def save_raster_blockwise(
    data: xr.DataArray,
    output_path: str,
    block_size: int = 4096
):
    """Save large raster in blocks to avoid memory issues."""
    print(f"  Saving raster to: {output_path}")
    
    # Get metadata
    crs = data.rio.crs
    transform = data.rio.transform()
    nodata = data.rio.nodata if hasattr(data.rio, 'nodata') else 255
    
    # Handle multi-band
    if "band" in data.dims:
        ny, nx = data.sizes["y"], data.sizes["x"]
        count = data.sizes["band"]
    else:
        ny, nx = data.shape
        count = 1
    
    # Create output file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=ny,
        width=nx,
        count=count,
        dtype='uint8',
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress='lzw',
        tiled=True,
        blockxsize=min(512, nx),
        blockysize=min(512, ny)
    ) as dst:
        
        # Write in blocks
        n_blocks_y = int(np.ceil(ny / block_size))
        n_blocks_x = int(np.ceil(nx / block_size))
        total_blocks = n_blocks_y * n_blocks_x
        
        for iy in range(n_blocks_y):
            y_start = iy * block_size
            y_end = min((iy + 1) * block_size, ny)
            
            for ix in range(n_blocks_x):
                x_start = ix * block_size
                x_end = min((ix + 1) * block_size, nx)
                
                # Read block
                if "band" in data.dims:
                    block = data.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
                    block_np = block.compute().values
                    
                    # Write each band
                    for b in range(count):
                        dst.write(block_np[b], b + 1, 
                                window=rasterio.windows.Window(x_start, y_start, 
                                                               x_end - x_start, y_end - y_start))
                else:
                    block = data.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
                    block_np = block.compute().values
                    dst.write(block_np, 1,
                            window=rasterio.windows.Window(x_start, y_start,
                                                           x_end - x_start, y_end - y_start))
                
                if (iy * n_blocks_x + ix + 1) % 20 == 0:
                    print(f"    Written {iy * n_blocks_x + ix + 1}/{total_blocks} blocks")
    
    print(f"  ✓ Saved: {output_path}")


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
    
    print(f"\n{'='*70}")
    print(f"Step 1: Generate Forest Comparison Raster (Year {year})")
    print(f"{'='*70}\n")
    
    # -------------------------------------------------------------------------
    # 1. Load MapBiomas (keep in original CRS temporarily)
    # -------------------------------------------------------------------------
    print("1. Loading MapBiomas...")
    tif_mb = os.path.join(folder_mapbiomas, f"AMZ.{year}.M.tif")
    mb = load_mapbiomas_year(tif_mb)
    mb_bin = reclassify_to_binary_forest(mb, forest_values=(1, 2, 9), nodata_value=255)
    
    mb_bounds = mb_bin.rio.bounds()
    mb_crs = mb_bin.rio.crs
    print(f"  MapBiomas shape: {mb_bin.shape}")
    print(f"  MapBiomas CRS: {mb_crs}")
    print()
    
    # -------------------------------------------------------------------------
    # 2. Select and load GLANCE tiles (in native LAEA projection)
    # -------------------------------------------------------------------------
    print("2. Loading GLANCE tiles...")
    sel_tiles = select_glance_tiles_from_geojson(
        json_file, mb_bounds, mb_crs, folder_glance, year, "SA"
    )
    
    chunks = {"x": 4096, "y": 4096}
    glance_mosaic = load_glance_tiles(sel_tiles, chunks)
    glance_bin = reclassify_to_binary_forest(glance_mosaic, forest_values=(5,), nodata_value=255)
    
    print(f"  GLANCE mosaic shape: {glance_bin.shape}")
    print(f"  GLANCE CRS: {glance_bin.rio.crs}")
    print()
    
    # -------------------------------------------------------------------------
    # 3. Reproject MapBiomas to GLANCE grid (关键改进!)
    # -------------------------------------------------------------------------
    print("3. Reprojecting MapBiomas to GLANCE grid...")
    mb_bin_reproj = reproject_mapbiomas_to_glance(mb_bin, glance_bin)
    print()
    
    # -------------------------------------------------------------------------
    # 4. Create comparison raster
    # -------------------------------------------------------------------------
    print("4. Creating comparison raster...")
    comparison_stack, valid_mask = create_comparison_raster(glance_bin, mb_bin_reproj)
    
    # Statistics
    total_pixels = glance_bin.size
    valid_pixels = valid_mask.sum().compute().item()
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Valid pixels: {valid_pixels:,} ({100*valid_pixels/total_pixels:.2f}%)")
    print()
    
    # -------------------------------------------------------------------------
    # 5. Save outputs
    # -------------------------------------------------------------------------
    print("5. Saving outputs...")
    
    # Save comparison stack (2-band: GLANCE, MapBiomas)
    comparison_file = os.path.join(outdir, f"forest_comparison_{year}.tif")
    save_raster_blockwise(comparison_stack, comparison_file, block_size=4096)
    
    # Save valid mask
    mask_file = os.path.join(outdir, f"valid_mask_{year}.tif")
    valid_mask_uint8 = valid_mask.astype("uint8")
    valid_mask_uint8.rio.write_crs(glance_bin.rio.crs, inplace=True)
    valid_mask_uint8.rio.write_nodata(0, inplace=True)
    save_raster_blockwise(valid_mask_uint8, mask_file, block_size=4096)
    
    print(f"\n{'='*70}")
    print("COMPLETED")
    print(f"{'='*70}")
    print(f"Outputs saved to: {outdir}")
    print(f"  - Comparison raster: {os.path.basename(comparison_file)}")
    print(f"  - Valid mask: {os.path.basename(mask_file)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("Initializing Dask cluster...")
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=16,
        memory_limit="100GB",
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