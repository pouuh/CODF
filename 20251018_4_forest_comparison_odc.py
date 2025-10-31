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
import rasterio
from rasterio.windows import Window
from odc.geo.xr import write_cog  # 放在文件顶部 import 区

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
# def reclassify_to_forest(arr: xr.DataArray, forest_values, nodata=255):
#     """Convert land-cover classes to binary forest map."""
#     return xr.where(arr.notnull(), arr.isin(forest_values).astype("uint8"), nodata)
def reclassify_to_forest(arr: xr.DataArray, forest_values, nodata=255):
    """Convert land-cover classes to binary forest map."""
    result = xr.where(arr.notnull(), arr.isin(forest_values).astype("uint8"), nodata)
    
    # 保留原始数组的维度、坐标和属性
    result = result.assign_coords(arr.coords)
    result.attrs = arr.attrs
    
    # 如果有 rio 属性，也保留 CRS 和 transform
    if hasattr(arr, 'rio'):
        try:
            if arr.rio.crs is not None:
                result.rio.write_crs(arr.rio.crs, inplace=True)
            result.rio.write_transform(arr.rio.transform(), inplace=True)
            result.rio.write_nodata(nodata, inplace=True)
        except Exception as e:
            print(f"⚠️  Warning: Could not preserve CRS: {e}")
    
    return result

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

def encode_pair(ref, pred, nodata=255):
    """Encode reference and prediction into a single array.
    
    Handles dimension mismatch between different coordinate naming conventions.
    """
    print(f"🔍 Input dimensions: ref={ref.dims}, pred={pred.dims}")
    print(f"🔍 Input shapes: ref={ref.shape}, pred={pred.shape}")
    
    # 检查并统一维度名称 - 防止维度广播导致4D数组
    if ref.dims != pred.dims:
        print(f"⚠️  Dimension mismatch detected!")
        # 如果维度不匹配，重命名 pred 的维度以匹配 ref
        if 'y' in ref.dims and 'latitude' in pred.dims:
            pred = pred.rename({'latitude': 'y', 'longitude': 'x'})
            print(f"✓ Renamed pred dims from (latitude,longitude) to {pred.dims}")
        elif 'latitude' in ref.dims and 'y' in pred.dims:
            pred = pred.rename({'y': 'latitude', 'x': 'longitude'})
            print(f"✓ Renamed pred dims from (y,x) to {pred.dims}")
    
    # 确保坐标对齐 - 这是关键步骤！
    print(f"🔍 Aligning coordinates...")
    pred = pred.reindex_like(ref, method='nearest', tolerance=0.001)
    print(f"✓ After alignment: pred shape={pred.shape}, dims={pred.dims}")
    
    ref_u8 = ref.astype("uint8")
    pred_u8 = pred.astype("uint8")
    valid = (ref_u8 != nodata) & (pred_u8 != nodata)
    
    # 检查有效像素数
    ref_valid = (ref_u8 != nodata).sum().compute()
    pred_valid = (pred_u8 != nodata).sum().compute()
    both_valid = valid.sum().compute()
    print(f"📊 Valid pixels: ref={ref_valid:,}, pred={pred_valid:,}, both={both_valid:,}")

    encoded = ((ref_u8 << 1) | pred_u8).where(valid, nodata)
    
    # Ensure proper metadata for write_cog
    encoded = encoded.squeeze()  # Remove any singleton dimensions
    encoded.attrs.update(ref.attrs)  # Copy attributes from reference
    
    # Ensure CRS and transform are set - check both ref and pred for CRS
    if hasattr(ref, 'rio') and ref.rio.crs is not None:
        encoded.rio.write_crs(ref.rio.crs, inplace=True)
        encoded.rio.write_transform(ref.rio.transform(), inplace=True)
        encoded.rio.write_nodata(nodata, inplace=True)
        
    elif hasattr(pred, 'rio') and pred.rio.crs is not None:
        encoded.rio.write_crs(pred.rio.crs, inplace=True)
        encoded.rio.write_transform(pred.rio.transform(), inplace=True)
        encoded.rio.write_nodata(nodata, inplace=True)
    
    return encoded

# =====================================================
# 3.5. Write COG in Chunks (避免内存溢出)
# =====================================================
def write_cog_chunked(encoded, output_path, chunk_size=8192, nodata=255):
    """
    分块写入大型 COG，避免内存溢出
    """
    print(f"⏳ Writing COG in chunks of {chunk_size}x{chunk_size}...")
    
    # 获取元数据和维度名称
    height, width = encoded.shape
    y_dim, x_dim = encoded.dims  # 动态获取维度名称
    transform = encoded.rio.transform()
    crs = encoded.rio.crs
    
    print(f"✓ Using dimensions: {y_dim}={height}, {x_dim}={width}")
    print(f"✓ CRS: {crs}")
    
    # 检查数据有效性
    valid_pixels = (encoded != nodata).sum().compute()
    print(f"📊 Total valid pixels to write: {valid_pixels:,}")
    
    # 第一步：写入原始数据（分块）
    temp_path = output_path.replace('.tif', '_temp.tif')
    
    with rasterio.open(
        temp_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress='deflate',
        zlevel=9,
    ) as dst:
        # 分块写入
        total_chunks = ((height + chunk_size - 1) // chunk_size) * \
                       ((width + chunk_size - 1) // chunk_size)
        processed = 0
        total_written_valid = 0
        
        for i in range(0, height, chunk_size):
            for j in range(0, width, chunk_size):
                h = min(chunk_size, height - i)
                w = min(chunk_size, width - j)
                
                # 读取这个块的数据 - 使用动态维度名称！
                window = Window(j, i, w, h)
                chunk_data = encoded.isel(
                    {y_dim: slice(i, i + h),
                     x_dim: slice(j, j + w)}
                ).compute()  # 只计算这个块
                
                # 检查这个块的有效像素数
                chunk_valid = np.sum(chunk_data.values != nodata)
                total_written_valid += chunk_valid
                
                # 写入
                dst.write(chunk_data.values, 1, window=window)
                
                processed += 1
                if processed % 10 == 0 or chunk_valid > 0:
                    print(f"  Chunk [{i}:{i+h}, {j}:{j+w}]: {chunk_valid:,} valid pixels | "
                          f"Progress: {processed}/{total_chunks} ({100*processed/total_chunks:.1f}%)")
        
        print(f"📊 Total valid pixels written: {total_written_valid:,}")
    
    print(f"✓ Temporary file written: {temp_path}")
    
    # 第二步：添加 overviews 并转换为 COG
    print("⏳ Adding overviews and converting to COG...")
    import subprocess
    subprocess.run([
        'gdaladdo',
        '-r', 'nearest',
        temp_path,
        '2', '4', '8', '16', '32', '64'
    ], check=True)
    
    subprocess.run([
        'gdal_translate',
        '-co', 'TILED=YES',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'ZLEVEL=9',
        '-co', 'COPY_SRC_OVERVIEWS=YES',
        '-co', 'BLOCKXSIZE=512',
        '-co', 'BLOCKYSIZE=512',
        temp_path,
        output_path
    ], check=True)
    
    # 删除临时文件
    os.remove(temp_path)
    print(f"✅ COG written: {output_path}")

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
    mapbiomas_tif = f"/projectnb/modislc/users/chishan/data/MapBiomas/COG/AMZ.{year}.M.cog.tif"

    # ---- 3. Load MapBiomas ----
    mb = rioxarray.open_rasterio(mapbiomas_tif, chunks={"x": 2048, "y": 2048})
    mb = mb.squeeze(drop=True)
    mb_geobox = mb.odc.geobox
    print(f"✓ MapBiomas loaded: shape={mb.shape}, dims={mb.dims}, CRS={mb.rio.crs}")

    # ---- 4. Load GLANCE ----
    ds = load_glance_stac(stac_path, year, mapbiomas_geobox=mb_geobox)
    gl = ds["data"].isel(time=0)
    print(f"✓ GLANCE loaded: shape={gl.shape}, dims={gl.dims}, CRS={gl.rio.crs}")

    # ---- 5. Reclassify ----
    mb_bin = reclassify_to_forest(mb, [1, 2, 9])
    gl_bin = reclassify_to_forest(gl, [5])
    print("✓ Reclassification prepared")
    print(f"✓ mb_bin: shape={mb_bin.shape}, dims={mb_bin.dims}, CRS={mb_bin.rio.crs if hasattr(mb_bin, 'rio') else 'None'}")
    print(f"✓ gl_bin: shape={gl_bin.shape}, dims={gl_bin.dims}, CRS={gl_bin.rio.crs if hasattr(gl_bin, 'rio') else 'None'}")

    # ---- 6. Encode (lazy) ----
    encoded = encode_pair(mb_bin, gl_bin)
    
    # Ensure 2D shape
    if encoded.ndim > 2:
        encoded = encoded.squeeze()
    
    print(f"✓ Encoded shape: {encoded.shape}, dims: {encoded.dims}")
    
    # ---- 7. Write COG in chunks (避免内存溢出) ----
    out_dir = "/projectnb/modislc/users/chishan/data/forest_comparison/encoded_cogs"
    os.makedirs(out_dir, exist_ok=True)
    encoded_path = os.path.join(out_dir, f"glance_mapbiomas_encoded_{year}.tif")
    
    write_cog_chunked(encoded, encoded_path, chunk_size=8192, nodata=255)

    # # ---- 6. Compute confusion matrix ----
    # tn, fp, fn, tp = compute_confusion_lazy(mb_bin, gl_bin)
    # metrics = compute_metrics((tn, fp, fn, tp))

    # print("\n=== CONFUSION MATRIX ===")
    # print(f"[[TN {tn:>12,}, FP {fp:>12,}]")
    # print(f" [FN {fn:>12,}, TP {tp:>12,}]]")
    # print("\n=== METRICS ===")
    # for k, v in metrics.items():
    #     print(f"{k:10s}: {v:.4f}")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()