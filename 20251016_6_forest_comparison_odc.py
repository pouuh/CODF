"""
GLANCE vs MapBiomas Comparison using ODC-Geo
像GEE一样自动处理内存和分块！

安装依赖:
pip install odc-geo rioxarray geopandas xarray dask distributed
"""
import os
import glob
import numpy as np
import xarray as xr
import rioxarray
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject
import odc.geo.xr  # 自动添加.odc扩展到xarray
import geopandas as gpd
from shapely.geometry import box
import re
from typing import Tuple
import json
from datetime import datetime

# =============================================================================
# 1. 加载数据（自动分块）
# =============================================================================
def load_mapbiomas(tif_path: str) -> xr.DataArray:
    """加载MapBiomas，自动chunked"""
    print(f"Loading MapBiomas: {tif_path}")
    mb = rioxarray.open_rasterio(tif_path, chunks={"x": 4096, "y": 4096})
    if "band" in mb.dims and mb.sizes["band"] == 1:
        mb = mb.isel(band=0, drop=True)
    return mb


def load_glance_tiles(folder: str, year: int, tile_ids: set) -> xr.DataArray:
    """加载并合并GLANCE瓦片"""
    pattern = os.path.join(folder, f"GLANCE.A{year}0701.h*v*.001.*.SA.LC.tif")
    all_files = sorted(glob.glob(pattern))
    
    selected = [
        f for f in all_files 
        if (m := re.search(r'h\d+v\d+', os.path.basename(f))) 
        and m.group(0) in tile_ids
    ]
    
    print(f"Loading {len(selected)} GLANCE tiles...")
    tiles = []
    for f in selected:
        da = rioxarray.open_rasterio(f, chunks={"x": 2048, "y": 2048})
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.isel(band=0, drop=True)
        tiles.append(da)
    
    # 合并瓦片
    from rioxarray.merge import merge_arrays
    mosaic = merge_arrays(tiles, nodata=0)
    if "band" in mosaic.dims:
        mosaic = mosaic.isel(band=0, drop=True)
    
    return mosaic


# =============================================================================
# 2. 二值重分类
# =============================================================================
def reclassify_forest(da: xr.DataArray, forest_values: Tuple[int, ...]) -> xr.DataArray:
    """二值森林重分类 - 完全lazy操作"""
    valid = da.notnull()
    forest = da.isin(list(forest_values))
    binary = xr.where(valid, xr.where(forest, 1, 0), 255).astype("uint8")
    
    # 保留CRS元数据
    binary.rio.write_crs(da.rio.crs, inplace=True)
    binary.rio.write_nodata(255, inplace=True)
    
    return binary


# =============================================================================
# 3. ODC-Geo 自动重投影（关键！）
# =============================================================================
def reproject_to_glance_grid(mb_bin: xr.DataArray, glance_bin: xr.DataArray) -> xr.DataArray:
    """
    使用odc-geo自动重投影 - 完全lazy，自动处理内存！
    
    这个函数会：
    1. 自动计算最优分块策略
    2. 自动处理坐标变换
    3. 永远不会加载完整数组到内存
    4. 支持Dask并行处理
    """
    print("Reprojecting MapBiomas to GLANCE grid using odc-geo...")
    
    # 获取GLANCE的GeoBox（定义目标网格）
    target_geobox = glance_bin.odc.geobox
    
    print(f"  Source CRS: {mb_bin.rio.crs}")
    print(f"  Target CRS: {target_geobox.crs}")
    print(f"  Target shape: {target_geobox.shape}")
    
    # ODC-Geo魔法：一行代码完成重投影！
    # 完全lazy，自动分块，自动内存管理
    mb_reproj = mb_bin.odc.reproject(
        how=target_geobox,
        resampling="nearest",
        chunks={"x": 4096, "y": 4096}  # 可选：指定输出分块
    )
    
    print(f"  ✓ Reprojection configured (lazy)")
    return mb_reproj


# =============================================================================
# 4. 瓦片选择
# =============================================================================
def select_glance_tiles(json_file: str, mb_bounds: Tuple, mb_crs) -> set:
    """通过GeoJSON选择瓦片"""
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    
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
    
    gdf = gpd.read_file(json_file)
    gdf = gdf.set_crs(TILE_CRS, allow_override=True)
    
    # 转换MapBiomas边界到GLANCE CRS
    mb_in_glance = transform_bounds(mb_crs, TILE_CRS, *mb_bounds, densify_pts=21)
    bbox_glance = box(*mb_in_glance)
    
    gsel = gdf[gdf.intersects(bbox_glance)]
    print(f"Selected {len(gsel)} GLANCE tiles")
    
    return set(gsel["tileID"].astype(str))


# =============================================================================
# 5. 保存对比影像（分块保存）
# =============================================================================
def save_comparison_raster(glance_bin: xr.DataArray, 
                          mb_bin: xr.DataArray, 
                          output_path: str):
    """保存2-band对比影像 - 自动分块写入"""
    print(f"Saving comparison raster to: {output_path}")
    
    # 合并为2-band数据集
    comparison = xr.concat([glance_bin, mb_bin], dim="band")
    comparison = comparison.assign_coords(band=["glance_forest", "mapbiomas_forest"])
    
    # ODC-Geo可以直接保存大文件（自动分块）
    # 但我们用传统方法确保兼容性
    comparison.rio.to_raster(
        output_path,
        driver="GTiff",
        compress="lzw",
        tiled=True,
        blockxsize=512,
        blockysize=512
    )
    
    print(f"  ✓ Saved: {output_path}")


# =============================================================================
# 6. 计算混淆矩阵（分块）
# =============================================================================
def compute_confusion_matrix(comparison_tif: str, block_size: int = 4096) -> np.ndarray:
    """分块计算混淆矩阵"""
    print("Computing confusion matrix...")
    
    import rasterio
    
    cm = np.zeros((2, 2), dtype=np.int64)
    
    with rasterio.open(comparison_tif) as src:
        ny, nx = src.height, src.width
        n_blocks = int(np.ceil(ny / block_size)) * int(np.ceil(nx / block_size))
        
        print(f"  Processing {n_blocks} blocks...")
        
        processed = 0
        for ji, window in src.block_windows(1):
            # 读取两个波段
            glance = src.read(1, window=window)
            mapb = src.read(2, window=window)
            
            # 有效像素
            valid = (glance != 255) & (mapb != 255)
            
            # 计算混淆矩阵
            tn = np.sum((mapb == 0) & (glance == 0) & valid)
            fp = np.sum((mapb == 0) & (glance == 1) & valid)
            fn = np.sum((mapb == 1) & (glance == 0) & valid)
            tp = np.sum((mapb == 1) & (glance == 1) & valid)
            
            cm += np.array([[tn, fp], [fn, tp]], dtype=np.int64)
            
            processed += 1
            if processed % 50 == 0:
                print(f"    Block {processed}/{n_blocks}")
    
    return cm


def compute_metrics(cm: np.ndarray) -> dict:
    """计算精度指标"""
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum() or 1
    
    oa = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    po = oa
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total ** 2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    
    return {
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "overall_accuracy": float(oa),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "kappa": float(kappa)
    }


# =============================================================================
# 主流程
# =============================================================================
def main():
    year = 2016
    folder_glance = "/projectnb/measures/products/SA/v001/DAAC/LC/"
    folder_mapbiomas = "/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/"
    json_file = "/projectnb/measures/datasets/Grids/FILTERED/json/GLANCE_V01_AS_PROJ_TILE_FILTERED.geojson"
    outdir = f"/projectnb/modislc/users/chishan/data/forest_comparison_{year}"
    os.makedirs(outdir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GLANCE vs MapBiomas Forest Comparison (Year {year})")
    print(f"Using ODC-Geo for automatic memory management")
    print(f"{'='*70}\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load MapBiomas
    # -------------------------------------------------------------------------
    print("Step 1: Loading MapBiomas...")
    tif_mb = os.path.join(folder_mapbiomas, f"AMZ.{year}.M.tif")
    mb = load_mapbiomas(tif_mb)
    mb_bin = reclassify_forest(mb, forest_values=(1, 2, 9))
    print(f"  Shape: {mb_bin.shape}, CRS: {mb_bin.rio.crs}\n")
    
    # -------------------------------------------------------------------------
    # Step 2: Select and load GLANCE tiles
    # -------------------------------------------------------------------------
    print("Step 2: Loading GLANCE tiles...")
    mb_bounds = mb_bin.rio.bounds()
    tile_ids = select_glance_tiles(json_file, mb_bounds, mb_bin.rio.crs)
    
    glance = load_glance_tiles(folder_glance, year, tile_ids)
    glance_bin = reclassify_forest(glance, forest_values=(5,))
    print(f"  Shape: {glance_bin.shape}, CRS: {glance_bin.rio.crs}\n")
    
    # -------------------------------------------------------------------------
    # Step 3: Reproject MapBiomas (ODC-Geo magic!)
    # -------------------------------------------------------------------------
    print("Step 3: Reprojecting MapBiomas...")
    mb_bin_reproj = reproject_to_glance_grid(mb_bin, glance_bin)
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Save comparison raster
    # -------------------------------------------------------------------------
    print("Step 4: Saving comparison raster...")
    comparison_file = os.path.join(outdir, f"forest_comparison_{year}.tif")
    save_comparison_raster(glance_bin, mb_bin_reproj, comparison_file)
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Compute accuracy metrics
    # -------------------------------------------------------------------------
    print("Step 5: Computing accuracy metrics...")
    cm = compute_confusion_matrix(comparison_file)
    metrics = compute_metrics(cm)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Confusion Matrix: TN={metrics['confusion_matrix']['TN']:,}, "
          f"FP={metrics['confusion_matrix']['FP']:,}, "
          f"FN={metrics['confusion_matrix']['FN']:,}, "
          f"TP={metrics['confusion_matrix']['TP']:,}")
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}")
    print(f"F1 Score:         {metrics['f1_score']:.4f}")
    print(f"Cohen's Kappa:    {metrics['kappa']:.4f}")
    print(f"{'='*70}\n")
    
    # Save results
    results_file = os.path.join(outdir, f"metrics_{year}.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    # # 可选：使用Dask进行并行处理
    from dask.distributed import Client, LocalCluster
    
    # print("Initializing Dask cluster...")
    # cluster = LocalCluster(
    #     n_workers=4,  # 可以增加workers
    #     threads_per_worker=4,
    #     memory_limit="30GB",
    #     dashboard_address=":8787"
    # )
    # client = Client(cluster)
    # print(f"Dask dashboard: {client.dashboard_link}\n")
    
    # try:
    #     main()
    # finally:
    #     client.close()
    #     cluster.close()

    cluster = LocalCluster(
        n_workers=1,              # 单worker最优（避免数据复制）
        threads_per_worker=16,    # 使用所有CPU核心
        memory_limit="100GB",     # 为系统预留空间
        dashboard_address=":8787"
    )
    client = Client(cluster)

    try:
        main()
    finally:
        client.close()
        cluster.close()
