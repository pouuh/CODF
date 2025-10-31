# select_tiles.py
import os, glob, re
import geopandas as gpd
import rioxarray
from shapely.geometry import box

YEAR = 2016
MAPBIOMAS_TIF = "/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/AMZ.2016.M.tif"
GLANCE_DIR = "/projectnb/measures/products/SA/v001/DAAC/LC"
GEOJSON = "/projectnb/measures/datasets/Grids/FILTERED/json/GLANCE_V01_AS_PROJ_TILE_FILTERED.geojson"
OUTLIST = "/projectnb/modislc/users/chishan/data/forest_comparison_2016/tiles.txt"
os.makedirs(os.path.dirname(OUTLIST), exist_ok=True)

# load mapbiomas template
mb = rioxarray.open_rasterio(MAPBIOMAS_TIF, chunks={"x":1,"y":1}).squeeze("band", drop=True)
mb_bounds = mb.rio.bounds(); mb_crs = mb.rio.crs

gdf = gpd.read_file(GEOJSON)

# 如果 geojson 没有 crs 或标错（坐标量级像米），用一个 sample GLANCE 的 crs 覆盖
sample = sorted(glob.glob(os.path.join(GLANCE_DIR, f"GLANCE.A{YEAR}0701.h*v*.001.*.SA.LC.tif")))[0]
import rasterio
with rasterio.open(sample) as ds:
    glance_crs = ds.crs
if gdf.crs is None or (gdf.crs.is_geographic and max(abs(gdf.total_bounds))>1000):
    gdf = gdf.set_crs(glance_crs, allow_override=True)

gdf_mb = gdf.to_crs(mb_crs)
minx, miny, maxx, maxy = mb_bounds
mb_box = box(minx, miny, maxx, maxy)
sel = gdf_mb[gdf_mb.intersects(mb_box)]
tile_ids = set(sel["tileID"].astype(str).tolist())

pattern = os.path.join(GLANCE_DIR, f"GLANCE.A{YEAR}0701.h*v*.001.*.SA.LC.tif")
allf = sorted(glob.glob(pattern))
selected = [os.path.abspath(f) for f in allf if (m:=re.search(r'h\d+v\d+', os.path.basename(f))) and m.group(0) in tile_ids]

with open(OUTLIST, "w") as fh:
    fh.write("\n".join(selected))
print("Selected tiles:", len(selected), "written to", OUTLIST)