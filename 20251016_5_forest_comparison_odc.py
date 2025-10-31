"""
Script: Forest Comparison using odc-geo (Optimized)
Strategy: Reproject GLANCE to MapBiomas grid (smaller data movement)
Author: Optimized for memory efficiency
Date: 2025-10-16
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
from typing import List, Tuple, Optional
import warnings

# Import odc-geo
try:
    from odc.geo.xr import xr_reproject
    from odc.geo.geobox import GeoBox
    ODC_AVAILABLE = True
except ImportError:
    print("⚠️  odc-geo not available. Install with: conda install -c conda-forge odc-geo")
    ODC_AVAILABLE = False
    raise

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
dask.config.set({
    "array.slicing.split_large_chunks": True,
    "optimization.fuse.active": True,
    "array.chunk-size": "128 MiB",  # Increased for better performance
    "distributed.worker.memory.target": 0.50,
    "distributed.worker.memory.spill": 0.65,
    "distributed.worker.memory.pause": 0.75,
    "distributed.worker.memory.terminate": 0.95,
})

# GLANCE LAEA projection
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
def get_optimal_chunks(shape: Tuple[int, int], max_chunk_mb: int = 128) -> dict:
    """Calculate optimal chunk size for given shape."""
    ny, nx = shape
    bytes_per_pixel = 1  # uint8
    max_chunk_bytes = max_chunk_mb * 1024 * 1024
    
    chunk_size = int(np.sqrt(max_chunk_bytes / bytes_per_pixel))
    chunk_y = min(chunk_size, ny)
    chunk_x = min(chunk_size, nx)
    
    return {"y": chunk_y, "x": chunk_x}


def print_array_info(da: xr.DataArray, name: str):
    """Print diagnostic information about DataArray."""
    size_gb = da.nbytes / 1e9
    chunked = "Yes" if da.chunks else "No"
    print(f"  {name}:")
    print(f"    Shape: {da.shape}")
    print(f"    Size: {size_gb:.2f} GB")
    print(f"    Chunks: {chunked}")
    if da.chunks:
        print(f"    Chunk sizes: y={da.chunks[da.dims.index('y')][0]}, x={da.chunks[da.dims.index('x')][0]}")
    print(f"    CRS: {da.rio.crs}")
    print(f"    Dtype: {da.dtype}")


def reclassify_to_binary_forest(
    da: xr.DataArray, 
    forest_values: Tuple[int, ...], 
    nodata_value: int = 255
) -> xr.DataArray:
    """
    Reclassify to binary forest map.
    
    Parameters
    ----------
    da : xr.DataArray
        Input land cover array
    forest_values : tuple
        Values representing forest classes
    nodata_value : int
        Value for nodata pixels
        
    Returns
    -------
    xr.DataArray
        Binary forest map (1=forest, 0=non-forest, 255=nodata)
    """
    arr = da.squeeze(drop=True)
    
    # Ensure chunking
    if arr.chunks is None:
        chunks = get_optimal_chunks(arr.shape)
        print(f"    Chunking with: {chunks}")
        arr = arr.chunk(chunks)
    
    # Reclassify
    valid = arr.notnull()
    forest = arr.isin(list(forest_values))
    out = xr.where(valid, xr.where(forest, 1, 0), nodata_value).astype("uint8")
    
    # Preserve spatial metadata
    if hasattr(arr, "rio"):
        out.rio.write_crs(arr.rio.crs, inplace=True)
        out.rio.write_nodata(nodata_value, inplace=True)
    
    return out


# =============================================================================
# MAPBIOMAS OPERATIONS
# =============================================================================
def load_mapbiomas_year(tif_path: str, chunks: Optional[dict] = None) -> xr.DataArray:
    """
    Load MapBiomas data with optimal chunking.
    
    Parameters
    ----------
    tif_path : str
        Path to MapBiomas GeoTIFF
    chunks : dict, optional
        Chunk specification
        
    Returns
    -------
    xr.DataArray
        Loaded MapBiomas data
    """
    # Get shape first
    with rasterio.open(tif_path) as src:
        shape = (src.height, src.width)
        file_size_gb = os.path.getsize(tif_path) / 1e9
        uncompressed_gb = (shape[0] * shape[1]) / 1e9
    
    print(f"  File size: {file_size_gb:.2f} GB (compressed)")
    print(f"  Uncompressed: {uncompressed_gb:.2f} GB")
    
    if chunks is None:
        chunks = get_optimal_chunks(shape, max_chunk_mb=128)
    
    print(f"  Loading with chunks: {chunks}")
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
    """
    Select GLANCE tiles intersecting MapBiomas extent.
    
    Parameters
    ----------
    json_file : str
        Path to GLANCE tile index GeoJSON
    mb_bounds : tuple
        MapBiomas bounds (minx, miny, maxx, maxy)
    mb_crs : CRS
        MapBiomas CRS
    folder_glance : str
        GLANCE data directory
    year : int
        Year to process
    region : str
        Region code (default: "SA")
        
    Returns
    -------
    list
        Paths to selected GLANCE tiles
    """
    # Load tile index
    gdf = gpd.read_file(json_file)
    gdf = gdf.set_crs(TILE_CRS, allow_override=True)
    
    # Transform MapBiomas bounds to GLANCE CRS
    mb_in_glance = transform_bounds(mb_crs, TILE_CRS, *mb_bounds, densify_pts=21)
    bbox_glance = box(*mb_in_glance)
    
    # Select intersecting tiles
    gsel = gdf[gdf.intersects(bbox_glance)]
    
    if gsel.empty:
        raise RuntimeError("No GLANCE tiles intersect MapBiomas extent")
    
    print(f"  Found {len(gsel)} intersecting tiles")
    
    # Match files
    pattern = os.path.join(folder_glance, f"GLANCE.A{year}0701.h*v*.001.*.{region}.LC.tif")
    all_files = sorted(glob.glob(pattern))
    
    if not all_files:
        raise RuntimeError(f"No GLANCE files found matching pattern: {pattern}")
    
    tile_ids = set(gsel["tileID"].astype(str))
    
    selected = [
        f for f in all_files 
        if (m := re.search(r'h\d+v\d+', os.path.basename(f))) 
        and m.group(0) in tile_ids
    ]
    
    if not selected:
        raise RuntimeError("No GLANCE files matched tile IDs")
    
    print(f"  Selected {len(selected)} GLANCE tiles")
    return selected


# =============================================================================
# GLANCE MOSAIC (Native LAEA projection)
# =============================================================================
def load_glance_tiles(tile_files: List[str], chunks: dict) -> xr.DataArray:
    """
    Load and mosaic GLANCE tiles in native LAEA projection.
    
    Parameters
    ----------
    tile_files : list
        Paths to GLANCE tile files
    chunks : dict
        Chunk specification
        
    Returns
    -------
    xr.DataArray
        Mosaicked GLANCE data in LAEA projection
    """
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
        
        if (i + 1) % 5 == 0 or (i + 1) == len(tile_files):
            print(f"    Loaded {i+1}/{len(tile_files)} tiles")
    
    # Merge tiles using rioxarray
    from rioxarray.merge import merge_arrays
    print("  Merging GLANCE tiles...")
    mosaic = merge_arrays(tiles, nodata=0)
    
    if "band" in mosaic.dims and mosaic.sizes["band"] == 1:
        mosaic = mosaic.isel(band=0, drop=True)
    
    # Ensure proper chunking
    if mosaic.chunks is None:
        mosaic = mosaic.chunk(chunks)
    
    return mosaic


# =============================================================================
# REPROJECT GLANCE TO MAPBIOMAS GRID (using odc-geo)
# =============================================================================
def reproject_glance_to_mapbiomas_odc(
    glance_bin: xr.DataArray,
    mapbiomas_template: xr.DataArray,
    nodata: int = 255
) -> xr.DataArray:
    """
    Reproject GLANCE to MapBiomas grid using odc-geo (memory efficient).
    
    Parameters
    ----------
    glance_bin : xr.DataArray
        GLANCE binary forest in LAEA projection
    mapbiomas_template : xr.DataArray
        MapBiomas array defining target grid
    nodata : int
        Nodata value
        
    Returns
    -------
    xr.DataArray
        GLANCE reprojected to MapBiomas grid
    """
    print("  Reprojecting GLANCE → MapBiomas grid using odc-geo...")
    print(f"    Source (GLANCE) shape: {glance_bin.shape}")
    print(f"    Target (MapBiomas) shape: {mapbiomas_template.shape}")
    
    # Create target GeoBox from MapBiomas
    target_geobox = GeoBox.from_xarray(mapbiomas_template)
    
    print(f"    Target CRS: {target_geobox.crs}")
    print(f"    Target resolution: {target_geobox.resolution}")
    
    # Reproject using odc-geo
    glance_reproj = xr_reproject(
        glance_bin,
        how=target_geobox,
        resampling="nearest",
        dst_nodata=nodata
    )
    
    # Ensure proper chunking
    if glance_reproj.chunks is None:
        chunks = get_optimal_chunks(glance_reproj.shape)
        glance_reproj = glance_reproj.chunk(chunks)
    
    print(f"    Reprojected shape: {glance_reproj.shape}")
    print(f"    ✓ Reprojection complete")
    
    return glance_reproj


# =============================================================================
# GENERATE COMPARISON RASTER
# =============================================================================
def create_comparison_raster(
    glance_bin: xr.DataArray,
    mapb_bin: xr.DataArray,
    nodata: int = 255
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Create stacked comparison raster and valid mask.
    
    Parameters
    ----------
    glance_bin : xr.DataArray
        GLANCE binary forest
    mapb_bin : xr.DataArray
        MapBiomas binary forest
    nodata : int
        Nodata value
        
    Returns
    -------
    tuple
        (stacked comparison raster, valid mask)
    """
    print("  Creating comparison raster...")
    
    # Verify alignment
    if glance_bin.shape != mapb_bin.shape:
        raise ValueError(f"Shape mismatch: GLANCE {glance_bin.shape} vs MapBiomas {mapb_bin.shape}")
    
    # Ensure same chunking
    target_chunks = {"x": 4096, "y": 4096}
    if glance_bin.chunks is None:
        glance_bin = glance_bin.chunk(target_chunks)
    if mapb_bin.chunks is None:
        mapb_bin = mapb_bin.chunk(target_chunks)
    
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
    
    print(f"    Stacked shape: {stacked.shape}")
    print(f"    Valid mask shape: {valid_mask.shape}")
    
    return stacked, valid_mask


def save_raster_blockwise(
    data: xr.DataArray,
    output_path: str,
    block_size: int = 4096,
    compression: str = "lzw"
):
    """
    Save large raster in blocks to avoid memory issues.
    
    Parameters
    ----------
    data : xr.DataArray
        Data to save
    output_path : str
        Output file path
    block_size : int
        Block size for reading/writing
    compression : str
        Compression algorithm
    """
    print(f"  Saving raster to: {os.path.basename(output_path)}")
    
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
        compress=compression,
        tiled=True,
        blockxsize=min(512, nx),
        blockysize=min(512, ny)
    ) as dst:
        
        # Write in blocks
        n_blocks_y = int(np.ceil(ny / block_size))
        n_blocks_x = int(np.ceil(nx / block_size))
        total_blocks = n_blocks_y * n_blocks_x
        
        print(f"    Writing {total_blocks} blocks ({n_blocks_y}×{n_blocks_x})...")
        
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
                        dst.write(
                            block_np[b], 
                            b + 1, 
                            window=rasterio.windows.Window(
                                x_start, y_start, 
                                x_end - x_start, y_end - y_start
                            )
                        )
                else:
                    block = data.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
                    block_np = block.compute().values
                    dst.write(
                        block_np, 
                        1,
                        window=rasterio.windows.Window(
                            x_start, y_start,
                            x_end - x_start, y_end - y_start
                        )
                    )
                
                # Progress
                block_idx = iy * n_blocks_x + ix + 1
                if block_idx % 50 == 0 or block_idx == total_blocks:
                    print(f"      Progress: {block_idx}/{total_blocks} blocks ({100*block_idx/total_blocks:.1f}%)")
    
    file_size = os.path.getsize(output_path) / 1e9
    print(f"    ✓ Saved: {file_size:.2f} GB")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    """Main processing pipeline."""
    # Configuration
    year = 2016
    folder_glance = "/projectnb/measures/products/SA/v001/DAAC/LC/"
    folder_mapbiomas = "/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/"
    json_file = "/projectnb/measures/datasets/Grids/FILTERED/json/GLANCE_V01_AS_PROJ_TILE_FILTERED.geojson"
    outdir = f"/projectnb/modislc/users/chishan/data/forest_comparison_{year}_odc"
    os.makedirs(outdir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Forest Comparison using odc-geo (Year {year})")
    print(f"Strategy: Reproject GLANCE → MapBiomas grid (memory efficient)")
    print(f"{'='*80}\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load MapBiomas (in original CRS: EPSG:4674)
    # -------------------------------------------------------------------------
    print("Step 1: Loading MapBiomas...")
    tif_mb = os.path.join(folder_mapbiomas, f"AMZ.{year}.M.tif")
    mb = load_mapbiomas_year(tif_mb)
    print_array_info(mb, "MapBiomas (raw)")
    
    print("  Reclassifying MapBiomas to binary forest...")
    mb_bin = reclassify_to_binary_forest(
        mb, 
        forest_values=(1, 2, 9),  # Forest formation, savanna, mangrove
        nodata_value=255
    )
    print_array_info(mb_bin, "MapBiomas (binary)")
    
    mb_bounds = mb_bin.rio.bounds()
    mb_crs = mb_bin.rio.crs
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Select and load GLANCE tiles (in native LAEA projection)
    # -------------------------------------------------------------------------
    print("Step 2: Loading GLANCE tiles...")
    sel_tiles = select_glance_tiles_from_geojson(
        json_file, mb_bounds, mb_crs, folder_glance, year, "SA"
    )
    
    chunks = {"x": 4096, "y": 4096}
    glance_mosaic = load_glance_tiles(sel_tiles, chunks)
    print_array_info(glance_mosaic, "GLANCE mosaic (raw)")
    
    print("  Reclassifying GLANCE to binary forest...")
    glance_bin = reclassify_to_binary_forest(
        glance_mosaic, 
        forest_values=(5,),  # Tree cover
        nodata_value=255
    )
    print_array_info(glance_bin, "GLANCE (binary)")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Reproject GLANCE to MapBiomas grid using odc-geo ✨
    # -------------------------------------------------------------------------
    print("Step 3: Reprojecting GLANCE to MapBiomas grid...")
    glance_on_mb = reproject_glance_to_mapbiomas_odc(
        glance_bin, 
        mb_bin,
        nodata=255
    )
    print_array_info(glance_on_mb, "GLANCE (reprojected)")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Create comparison raster (on MapBiomas grid)
    # -------------------------------------------------------------------------
    print("Step 4: Creating comparison raster...")
    comparison_stack, valid_mask = create_comparison_raster(
        glance_on_mb, 
        mb_bin,
        nodata=255
    )
    
    # Calculate statistics
    print("  Computing statistics...")
    total_pixels = valid_mask.size
    valid_pixels = valid_mask.sum().compute().item()
    valid_pct = 100 * valid_pixels / total_pixels
    
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Valid pixels: {valid_pixels:,} ({valid_pct:.2f}%)")
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Save outputs
    # -------------------------------------------------------------------------
    print("Step 5: Saving outputs...")
    
    # Save comparison stack (2-band: GLANCE, MapBiomas)
    comparison_file = os.path.join(outdir, f"forest_comparison_{year}_EPSG4674.tif")
    save_raster_blockwise(comparison_stack, comparison_file, block_size=4096)
    
    # Save valid mask
    mask_file = os.path.join(outdir, f"valid_mask_{year}.tif")
    valid_mask_uint8 = valid_mask.astype("uint8")
    valid_mask_uint8.rio.write_crs(mb_bin.rio.crs, inplace=True)
    valid_mask_uint8.rio.write_nodata(0, inplace=True)
    save_raster_blockwise(valid_mask_uint8, mask_file, block_size=4096)
    
    # -------------------------------------------------------------------------
    # (Optional) Step 6: Reproject comparison to LAEA if needed
    # -------------------------------------------------------------------------
    SAVE_LAEA_VERSION = False  # Set to True if you need LAEA output
    
    if SAVE_LAEA_VERSION:
        print("\nStep 6: Reprojecting comparison to LAEA...")
        
        # Use GLANCE's original grid as template
        target_geobox = GeoBox.from_xarray(glance_bin)
        
        comparison_laea = xr_reproject(
            comparison_stack,
            how=target_geobox,
            resampling="nearest",
            dst_nodata=255
        )
        
        laea_file = os.path.join(outdir, f"forest_comparison_{year}_LAEA.tif")
        save_raster_blockwise(comparison_laea, laea_file, block_size=4096)
        print(f"  ✓ LAEA version saved")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("✓ PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {outdir}")
    print(f"Files created:")
    print(f"  1. {os.path.basename(comparison_file)}")
    print(f"     - Band 1: GLANCE forest (1=forest, 0=non-forest, 255=nodata)")
    print(f"     - Band 2: MapBiomas forest")
    print(f"     - CRS: {mb_crs}")
    print(f"  2. {os.path.basename(mask_file)}")
    print(f"     - Valid data mask (1=valid, 0=nodata)")
    if SAVE_LAEA_VERSION:
        print(f"  3. forest_comparison_{year}_LAEA.tif")
        print(f"     - Same as #1 but in LAEA projection")
    print(f"\nValid pixel coverage: {valid_pct:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Initializing Dask cluster...")
    print("="*80)
    
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=16,
        memory_limit="100GB",
        dashboard_address=":8787"
    )
    client = Client(cluster)
    
    print(f"Dask dashboard: {client.dashboard_link}")
    print(f"Workers: {len(client.scheduler_info()['workers'])}")
    print(f"Total memory: 100 GB")
    print("="*80 + "\n")
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down Dask cluster...")
        client.close()
        cluster.close()
        print("✓ Cluster closed\n")
