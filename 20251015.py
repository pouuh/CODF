import os, glob, re
import dask.array as da
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling

# -------------------------
# 路径与输入
# -------------------------
GLANCE_DIR = "/projectnb/measures/products/SA/v001/DAAC/LC"  # 你的 GLANCE 路径
YEAR = 2018
# 参考图（别人家的 2018 Amazon TIFF）
folder_terra = '/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/'
REF_TIF = folder_terra + 'AMZ.2016.M.tif'
# 亚马逊边界（多边形），EPSG 任意，但最好是地理坐标（WGS84）
AMAZON_SHP = "/projectnb/modislc/users/chishan/data/MapBiomas/Limites_Amazonia_Legal_2024_shp/Limites_Amazonia_Legal_2024.shp"

# -------------------------
# 选出 2018 年的分块
# 名称里通常像 A20180701；保险起见用正则 YEAR
# -------------------------
pattern = os.path.join(GLANCE_DIR, f"*A{YEAR}*.SA.LC.tif")
tiles = sorted(glob.glob(pattern))
assert len(tiles) > 0, f"No tiles found for {YEAR} with pattern: {pattern}"
print(f"Found {len(tiles)} tiles for {YEAR}")

# -------------------------
# 读参考图（作为对齐模板），用 dask 分块
# -------------------------
ref = rxr.open_rasterio(REF_TIF, chunks={"x": 2048, "y": 2048})  # (band, y, x)
# 确保 nodata
if ref.rio.nodata is None:
    ref = ref.rio.write_nodata(0, inplace=False)

# -------------------------
# 读 Amazon 边界
# -------------------------
amazon = gpd.read_file(AMAZON_SHP)
# 投影到参考图 CRS，便于 clip/reproject_match
amazon = amazon.to_crs(ref.rio.crs)

# -------------------------
# 打开 GLANCE 分块（懒加载），统一 nodata，先投到参考 CRS，再裁剪
# 注：如果 GLANCE 原本 CRS 与 ref 不同，直接 reproject_match(ref)
# -------------------------
chunks = {"x": 2048, "y": 2048}
pieces = []
for fp in tiles:
    da_tile = rxr.open_rasterio(fp, chunks=chunks)  # (band, y, x)
    # 写 nodata，假设 >7 为 nodata
    if da_tile.rio.nodata is None:
        da_tile = da_tile.rio.write_nodata(0, inplace=False)

    # 可选：把 >7 的值置 0（避免 255 影响）
    da_tile = da_tile.where((da_tile >= 1) & (da_tile <= 7), other=0)

    # 对齐：重采样到参考图网格（nearest）
    # 如果两个栅格像元大小不同/CRS不同，这一步会触发重投影与重采样
    aligned = da_tile.rio.reproject_match(ref, resampling=Resampling.nearest)

    # 裁剪到 Amazon（加 buffer 可避免边缘空洞）
    aligned = aligned.rio.clip(amazon.geometry, amazon.crs, drop=True, invert=False)
    pieces.append(aligned)

# -------------------------
# 镶嵌：合并所有分块（懒）
# merge_arrays 会按空间位置拼接。默认 nodata=0
# -------------------------
mosaic = merge_arrays(pieces, nodata=0)

# mosaic 和 ref 现已同网格、同 CRS；两者可直接对比
pred = mosaic.squeeze(drop=True)   # (y, x)
gt   = ref.squeeze(drop=True)      # (y, x)

# 再次确保两者对齐（保险）
pred = pred.rio.reproject_match(gt, resampling=Resampling.nearest)

# -------------------------
# 掩膜：同时为 nodata 的不要；仅统计 1..7
# -------------------------
valid = (pred >= 1) & (pred <= 7) & (gt >= 1) & (gt <= 7)
pred_v = pred.where(valid).data.astype("int16").rechunk((4096, 4096))
gt_v   = gt.where(valid).data.astype("int16").rechunk((4096, 4096))

# -------------------------
# 混淆矩阵（Dask, 懒计算）
# 将 (gt, pred) 映射到一个编码：code = gt*100 + pred
# 这样可以用 bincount 一次得到全部配对计数
# -------------------------
code = gt_v * 100 + pred_v
code = code.where(valid.data, other=-1)

# 只保留 1..7 的配对：范围 101..707
min_code, max_code = 100 + 1, 700 + 7
shift = min_code
flat = code.reshape((-1,))
mask = (flat >= min_code) & (flat <= max_code)
sel = da.where(mask, flat - shift, -1)

nbins = (max_code - min_code + 1)
counts = da.bincount(da.where(sel >= 0, sel, 0), minlength=nbins) - da.bincount(da.where(sel >= 0, 0, 0), minlength=nbins)

# 还原成 7x7 混淆矩阵（行: gt, 列: pred）
cm = counts.reshape((7, 7))

# -------------------------
# 指标：OA、UA/PA、IoU（Dask → NumPy 触发计算）
# -------------------------
cm_np = cm.compute()  # 触发并行计算
cm_np = cm_np.astype(np.int64)

gt_sum = cm_np.sum(axis=1)  # 每个真值类像元数
pd_sum = cm_np.sum(axis=0)  # 每个预测类像元数
diag  = np.diag(cm_np)

OA = diag.sum() / cm_np.sum()

# Producer's / User's Accuracy
PA = np.divide(diag, gt_sum, out=np.zeros_like(diag, dtype=float), where=gt_sum>0)
UA = np.divide(diag, pd_sum, out=np.zeros_like(diag, dtype=float), where=pd_sum>0)

# IoU
intersection = diag
union = gt_sum + pd_sum - diag
IoU = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union>0)

print("Confusion matrix (7x7):\n", cm_np)
print(f"Overall Accuracy: {OA:.4f}")
for c in range(1, 8):
    print(f"Class {c}: PA={PA[c-1]:.3f}  UA={UA[c-1]:.3f}  IoU={IoU[c-1]:.3f}")