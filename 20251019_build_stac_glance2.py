# ...existing code...
from dask.distributed import Client, as_completed
import multiprocessing

def read_tif_metadata(tif_path):
    """Worker: open tif and return simple serializable metadata dict."""
    import os
    from datetime import datetime
    import rasterio
    from rasterio.warp import transform_bounds
    from pyproj import CRS
    from shapely.geometry import box, mapping

    name = os.path.basename(tif_path)
    meta = parse_filename(name)
    if meta is None:
        return {"error": "parse_failed", "tif": tif_path}

    try:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            crs = CRS(src.crs)
            transform = src.transform
            shape = src.shape
            epsg = crs.to_epsg()
            wgs84_bounds = transform_bounds(src.crs, "EPSG:4326", *bounds)
            geom = mapping(box(*wgs84_bounds))
            dt = datetime(meta["year"], 7, 1)

        return {
            "tif": os.path.abspath(tif_path),
            "name": name,
            "year": meta["year"],
            "h": meta["h"],
            "v": meta["v"],
            "bounds": list(bounds),
            "wgs84_bounds": list(wgs84_bounds),
            "geom": geom,
            "transform": list(transform)[:6],
            "shape": list(shape),
            "epsg": epsg,
            "datetime": dt.isoformat()
        }
    except Exception as e:
        return {"error": str(e), "tif": tif_path}

def main(input_dir, output_dir, base_id="glance", workers=None):
    os.makedirs(output_dir, exist_ok=True)

    # small placeholder extent (will update later)
    extent = Extent(
        spatial=SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
        temporal=TemporalExtent(intervals=[[None, None]])
    )

    collection = Collection(
        id=f"{base_id}_collection",
        description="GLANCE annual land cover tiles (South America)",
        extent=extent,
        license="proprietary",
    )
    catalog = Catalog(id=f"{base_id}_catalog", description="STAC catalog for GLANCE")
    catalog.add_child(collection)

    files = sorted(
        [os.path.join(input_dir, f)
         for f in os.listdir(input_dir)
         if f.endswith(".tif") and f.startswith("GLANCE.")]
    )
    print(f"🔍 Found {len(files)} GeoTIFFs in {input_dir}")

    if workers is None:
        workers = max(1, min(8, multiprocessing.cpu_count() // 2))

    # start local dask cluster (processes recommended)
    client = Client(processes=True, n_workers=workers, threads_per_worker=1)
    print(f"🚀 Dask client started: {client}")

    futures = client.map(read_tif_metadata, files)

    success_count = 0
    all_bboxes = []
    all_years = []

    for fut in as_completed(futures):
        res = fut.result()
        if not res or res.get("error"):
            if res:
                print(f"⚠️  Skipping {res.get('tif')}: {res.get('error')}")
            continue

        # 在主进程构建 Item
        item_id = f"{base_id}_{res['year']}_h{res['h']:02d}v{res['v']:02d}"
        item = Item(
            id=item_id,
            geometry=res["geom"],
            bbox=res["wgs84_bounds"],
            datetime=datetime.fromisoformat(res["datetime"]),
            properties={
                "year": res["year"],
                "horizontal": res["h"],
                "vertical": res["v"],
                "proj:epsg": res["epsg"],
                "proj:wkt2": None,
                "proj:transform": res["transform"],
                "proj:shape": res["shape"],
                "proj:bbox": res["bounds"],
            },
        )

        item.add_asset(
            "data",
            Asset(
                href=f"file://{res['tif']}",
                media_type="image/tiff; application=geotiff",
                roles=["data"],
                extra_fields={
                    "proj:epsg": res["epsg"],
                    "proj:transform": res["transform"],
                    "proj:shape": res["shape"],
                }
            ),
        )

        collection.add_item(item)
        success_count += 1
        all_bboxes.append(item.bbox)
        all_years.append(item.properties["year"])

        if success_count % 100 == 0:
            print(f"  ✓ Added {success_count} items...")

    # 更新 collection extent（若有 items）
    if all_bboxes:
        min_x = min(b[0] for b in all_bboxes)
        min_y = min(b[1] for b in all_bboxes)
        max_x = max(b[2] for b in all_bboxes)
        max_y = max(b[3] for b in all_bboxes)
        min_year = min(all_years)
        max_year = max(all_years)

        collection.extent = Extent(
            spatial=SpatialExtent(bboxes=[[min_x, min_y, max_x, max_y]]),
            temporal=TemporalExtent(intervals=[[
                datetime(min_year, 1, 1),
                datetime(max_year, 12, 31)
            ]])
        )

    client.close()

    catalog.normalize_hrefs(output_dir)
    catalog.save(catalog_type="SELF_CONTAINED")
    print(f"\n✅ STAC catalog written to {output_dir}")
    print(f"  Total items: {success_count}/{len(files)}")
# ...existing code...


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input directory with GLANCE TIFFs")
    p.add_argument("--output", required=True, help="Output STAC directory")
    p.add_argument("--base_id", default="glance", help="Base ID prefix")
    args = p.parse_args()
    main(args.input, args.output, args.base_id)