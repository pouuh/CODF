#!/usr/bin/env python3
"""
Build STAC Catalog for GLANCE Tiles
-----------------------------------
Each GeoTIFF -> one STAC Item.
Output: STAC-compliant directory ready for odc.stac.load.

Example:
  python /usr2/postdoc/chishan/research/CODF/20251017_build_stac_glance.py \
      --input /projectnb/measures/products/SA/v001/DAAC/LC/ \
      --output /projectnb/modislc/users/chishan/glance_SA_stac/
"""

import os
import re
import json
import rasterio
from datetime import datetime
from pystac import Catalog, Collection, Item, Asset
from shapely.geometry import box, mapping
from pyproj import CRS
import argparse


def parse_filename(filename):
    """Extract year, h, v from filename like GLANCE.A20160701.h46v04.001.SA.LC.tif"""
    m = re.search(r"A(\d{4})\d{4}\.h(\d+)v(\d+)", filename)
    if not m:
        return None
    return dict(year=int(m.group(1)), h=int(m.group(2)), v=int(m.group(3)))


def create_item(tif_path, base_id="glance"):
    """Create a STAC Item from a single GeoTIFF."""
    name = os.path.basename(tif_path)
    meta = parse_filename(name)
    if meta is None:
        return None

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = CRS(src.crs)
        geom = mapping(box(*bounds))
        dt = datetime(meta["year"], 7, 1)

    item_id = f"{base_id}_{meta['year']}_h{meta['h']}v{meta['v']}"
    item = Item(
        id=item_id,
        geometry=geom,
        bbox=list(bounds),
        datetime=dt,
        properties={
            "year": meta["year"],
            "horizontal": meta["h"],
            "vertical": meta["v"],
            "proj:epsg": crs.to_epsg() if crs.is_projected else None,
        },
    )

    item.add_asset(
        "data",
        Asset(
            href=f"file://{os.path.abspath(tif_path)}",
            media_type="image/tiff; application=geotiff",
            roles=["data"],
        ),
    )
    return item


def main(input_dir, output_dir, base_id="glance"):
    os.makedirs(output_dir, exist_ok=True)

    # build collection
    collection = Collection(
        id=f"{base_id}_collection",
        description="GLANCE annual land cover tiles (South America)",
        extent=None,
        license="proprietary",
    )
    catalog = Catalog(id=f"{base_id}_catalog", description="STAC catalog for GLANCE")

    catalog.add_child(collection)

    files = sorted(
        [os.path.join(input_dir, f)
         for f in os.listdir(input_dir)
         if f.endswith(".tif") and f.startswith("GLANCE.")]
    )
    print(f"Found {len(files)} GeoTIFFs in {input_dir}")

    for i, f in enumerate(files, 1):
        item = create_item(f, base_id=base_id)
        if item:
            collection.add_item(item)
            if i % 100 == 0:
                print(f"  Added {i} items...")

    catalog.normalize_hrefs(output_dir)
    catalog.save(catalog_type="SELF_CONTAINED")
    print(f"✓ STAC catalog written to {output_dir}")

# def main(input_dir, output_dir, base_id="glance"):
#     os.makedirs(output_dir, exist_ok=True)

#     # build collection
#     collection = Collection(
#         id=f"{base_id}_collection",
#         description="GLANCE annual land cover tiles (South America)",
#         extent=None,
#         license="proprietary",
#     )
#     catalog = Catalog(id=f"{base_id}_catalog", description="STAC catalog for GLANCE")

#     catalog.add_child(collection)

#     files = sorted(
#         [os.path.join(input_dir, f)
#          for f in os.listdir(input_dir)
#          if f.endswith(".tif") and f.startswith("GLANCE.")]
#     )
#     print(f"Found {len(files)} GeoTIFFs in {input_dir}")

#     for i, f in enumerate(files, 1):
#         item = create_item(f, base_id=base_id)
#         if item:
#             collection.add_item(item)
#             if i % 100 == 0:
#                 print(f"  Added {i} items...")
#             # Save every 5000 items to reduce memory usage
#             if i % 5000 == 0:
#                 catalog.normalize_hrefs(output_dir)
#                 catalog.save(catalog_type="SELF_CONTAINED")
#                 print(f"  Saved checkpoint at {i} items")

#     catalog.normalize_hrefs(output_dir)
#     catalog.save(catalog_type="SELF_CONTAINED")
#     print(f"✓ STAC catalog written to {output_dir}")

# from pystac import Catalog, Collection, Item, Asset, Extent, SpatialExtent, TemporalExtent

# def main(input_dir, output_dir, base_id="glance"):
#     os.makedirs(output_dir, exist_ok=True)

#     files = sorted(
#         [os.path.join(input_dir, f)
#          for f in os.listdir(input_dir)
#          if f.endswith(".tif") and f.startswith("GLANCE.")]
#     )
#     print(f"Found {len(files)} GeoTIFFs in {input_dir}")

#     # Create catalog
#     catalog = Catalog(id=f"{base_id}_catalog", description="STAC catalog for GLANCE")

#     # Create items first to calculate extent
#     items = []
#     for i, f in enumerate(files, 1):
#         item = create_item(f, base_id=base_id)
#         if item:
#             items.append(item)
#             if i % 100 == 0:
#                 print(f"  Created {i} items...")

#     if not items:
#         print("No valid items found!")
#         return

#     # Calculate extent from items
#     all_bboxes = [item.bbox for item in items]
#     min_lon = min(bbox[0] for bbox in all_bboxes)
#     min_lat = min(bbox[1] for bbox in all_bboxes)
#     max_lon = max(bbox[2] for bbox in all_bboxes)
#     max_lat = max(bbox[3] for bbox in all_bboxes)
    
#     all_dates = [item.datetime for item in items]
#     start_date = min(all_dates)
#     end_date = max(all_dates)

#     # Create collection with proper extent
#     collection = Collection(
#         id=f"{base_id}_collection",
#         description="GLANCE annual land cover tiles (South America)",
#         extent=Extent(
#             spatial=SpatialExtent(bboxes=[[min_lon, min_lat, max_lon, max_lat]]),
#             temporal=TemporalExtent(intervals=[[start_date, end_date]])
#         ),
#         license="proprietary",
#     )
    
#     catalog.add_child(collection)

#     # Add items to collection
#     for i, item in enumerate(items, 1):
#         collection.add_item(item)
#         if i % 1000 == 0:
#             print(f"  Added {i}/{len(items)} items to collection...")

#     catalog.normalize_hrefs(output_dir)
#     catalog.save(catalog_type="SELF_CONTAINED")
#     print(f"✓ STAC catalog written to {output_dir}")
#     print(f"  Total items: {len(items)}")
#     print(f"  Spatial extent: [{min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f}]")
#     print(f"  Temporal extent: {start_date.year} - {end_date.year}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input directory containing GLANCE TIFFs")
    p.add_argument("--output", required=True, help="Output STAC directory")
    p.add_argument("--base_id", default="glance", help="Base ID prefix")
    args = p.parse_args()
    main(args.input, args.output, args.base_id)