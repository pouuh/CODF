#!/usr/bin/env python3
"""
GLANCE vs MapBiomas Forest Comparison on Google Earth Engine
Exports confusion metrics and forest rasters to Google Drive.

Requires: earthengine-api (and optional geemap for map previews).
Authenticate once with ee.Authenticate(), then run.
"""

import ee
from datetime import datetime

# ---------- User configuration ----------
YEARS = [2008, 2010, 2012, 2014, 2016]  # Years to process
MB_ASSET = "projects/ee-zcs/assets/AMZ2018M"
GLANCE_PREFIX = "projects/GLANCE/DATASETS/V001"
     # e.g. GLANCE_2016
AOI_ASSET = "projects/ee-zcs/assets/AMZ2018M"         # FeatureCollection
FOREST_MB = [1, 2, 9]                                 # MapBiomas forest classes
FOREST_GLANCE = [5]                                   # GLANCE forest classes
NODATA = 255
EXPORT_FOLDER = "forest_confusion"                    # Google Drive folder
SCALE = 30
TILE_SCALE = 4                                        # increase if region large
# ---------------------------------------

def init_ee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

def get_aoi():
    return ee.Image(AOI_ASSET).geometry()

def reclass_to_binary(image, forest_classes):
    """
    Reclassifies an image to binary: 1 for forest classes, 0 otherwise.
    
    Args:
        image (ee.Image): The input image to reclassify.
        forest_classes (list): List of class values to map to 1 (forest).
    
    Returns:
        ee.Image: Binary image with 1 for forest and 0 for non-forest.
    """
    return image.remap(
        forest_classes,
        ee.List.repeat(1, len(forest_classes)),
        0
    )

def build_mapbiomas(year):
    # band = f"classification_{year}"
    src = ee.Image(MB_ASSET).select('b1')
    binary = reclass_to_binary(src, FOREST_MB)
    # mask = src.neq(NODATA)
    # return binary.updateMask(mask).rename("reference")
    return binary.rename("reference")

def build_glance(year):
    asset = ee.ImageCollection("projects/GLANCE/DATASETS/V001")
    asset = asset.filterDate(f"{year}-01-01", f"{year}-12-31").mosaic()
    src = ee.Image(asset).select('LC')
    binary = reclass_to_binary(src, FOREST_GLANCE)
    return binary.rename("prediction")

def compute_confusion(reference, prediction, region):
  
    valid = reference.mask().And(prediction.mask())
    encoded = reference.multiply(10).add(prediction).updateMask(valid)

    histogram = ee.Dictionary(encoded.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=region,
            scale=SCALE,
            maxPixels=1e13,
            tileScale=TILE_SCALE,
            bestEffort=True
        ).get('reference'))

    tn = ee.Number(histogram.get('0', 0))
    fp = ee.Number(histogram.get('1', 0))
    fn = ee.Number(histogram.get('10', 0))
    tp = ee.Number(histogram.get('11', 0))

    matrix = ee.Array([[tn, fp],
                    [fn, tp]])
    
    cm = ee.ConfusionMatrix(matrix)

    return ee.Dictionary({
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'accuracy': cm.accuracy(),
        'consumersAccuracy': cm.consumersAccuracy(),
        'producersAccuracy': cm.producersAccuracy(),
        'fscore': cm.fscore(),
        'matrix': matrix
    })


    # return histogram

def export_metrics(year, metrics):
    feature = ee.Feature(
        None,metrics
    )
    task = ee.batch.Export.table.toDrive(
        collection=ee.FeatureCollection([feature]),
        description=f"confusion_{year}_2",
        folder=EXPORT_FOLDER,
        fileNamePrefix=f"confusion_{year}",
        fileFormat="CSV"
    )
    task.start()
    print(f"🚀 Started metrics export for {year}: {task.id}")

def export_forest_rasters(year, reference, prediction, region):

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    agreement = reference.eq(prediction).updateMask(reference.mask())

    ee.batch.Export.image.toAsset(
        image=agreement,
        description=f"agreement_{year}",
        folder=EXPORT_FOLDER,
        fileNamePrefix=f"agreement_{year}_{timestamp}",
        region=region,
        scale=SCALE,
        crs=reference.projection().crs(),
        maxPixels=1e13
    ).start()

    print(f"🚀 Started image exports for {year}")

def main():
    init_ee()
    region = get_aoi()

    for year in YEARS:
        print(f"\n===== Year {year} =====")
        reference = build_mapbiomas(year)
        prediction = build_glance(year)

        results = compute_confusion(reference, prediction, region)

        export_metrics(year, results)
        export_forest_rasters(year, reference, prediction, region)
        # export_forest_rasters(year, reference, prediction, region)

    print("\nAll export tasks launched. Monitor status at https://code.earthengine.google.com/tasks")
