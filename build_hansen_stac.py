import os, re
import pystac
from shapely.geometry import box
from datetime import datetime

COG_DIR = "/projectnb/modislc/users/chishan/data/hansen_gfc_cog"
OUTPUT_DIR = "/projectnb/modislc/users/chishan/data/hansen_stac_catalog"

START_DT = datetime(2000,1,1)
END_DT   = datetime(2024,12,31)

catalog = pystac.Catalog(id="hansen-gfc", description="Hansen Global Forest Change")

for fname in os.listdir(COG_DIR):
    if not fname.endswith(".tif"): continue
    path = os.path.join(COG_DIR, fname)

    m = re.search(r'_(\d+)(N|S)_(\d+)(E|W)', fname)
    if not m: continue
    lat = int(m.group(1)) * (1 if m.group(2)=="N" else -1)
    lon = int(m.group(3)) * (1 if m.group(4)=="E" else -1)

    minx, maxx = lon, lon+10
    maxy, miny = lat, lat-10
    bbox = [minx, miny, maxx, maxy]

    layer = fname.split("_")[2]  # Hansen_GFC2015_lossyear_40N_080W.tif

    item = pystac.Item(
        id=fname.replace(".tif",""),
        geometry=box(*bbox).__geo_interface__,
        bbox=bbox,
        datetime=None,
        start_datetime=START_DT,
        end_datetime=END_DT,
        properties={"layer": layer, "proj:epsg": 4326}
    )
    item.add_asset("cog", pystac.Asset(href=path, media_type=pystac.MediaType.COG))
    catalog.add_item(item)

catalog.normalize_hrefs(OUTPUT_DIR)
catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

print(f"✓ STAC catalog saved to {OUTPUT_DIR}")