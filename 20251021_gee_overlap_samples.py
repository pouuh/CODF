#!/usr/bin/env python3
"""
TerraClass vs MapBiomas Land Cover Comparison on Google Earth Engine
Compares forest, cropland, and pasture classes between TerraClass and MapBiomas.
Exports confusion metrics to Google Drive.

Requires: earthengine-api (and optional geemap for map previews).
Authenticate once with ee.Authenticate(), then run.
"""
"""
  Terra  Classes:
  1: VEGETACAO NATURAL FLORESTAL PRIMARIA
  2: VEGETACAO NATURAL FLORESTAL SECUNDARIA
  9: SILVICULTURA
  10: PASTAGEM ARBUSTIVA/ARBOREA
  11: PASTAGEM HERBACEA
  12: CULTURA AGRICOLA PERENE
  13: CULTURA AGRICOLA SEMIPERENE
  14: CULTURA AGRICOLA TEMPORARIA DE 1 CICLO
  15: CULTURA AGRICOLA TEMPORARIA DE MAIS DE 1 CICLO
  16: MINERACAO
  17: URBANIZADA
  20: OUTROS USOS
  22: DESFLORESTAMENTO NO ANO
  23: CORPO DAGUA
  25: NAO OBSERVADO
  51: NATURAL NAO FLORESTAL

MapBiomas Classes:

COLEÇÃO 10 - CLASSES COLLECTION 10 - CLASSES Code
ID
Hexacode
Number
Color
ID
1. Floresta 1. Forest 1 #1f8d49
1.1 Formação Florestal 1.1. Forest Formation 3 #1f8d49
1.2. Formação Savânica 1.2. Savanna Formation 4 #7dc975
1.3. Mangue 1.3. Mangrove 5 #04381d
1.4. Floresta Alagável 1.4 Floodable Forest 6 #007785
1.5. Restinga Arbórea 1.5. Wooded Sandbank Vegetation 49 #02d659
2. Vegetação Herbácea e Arbustiva 2. Herbaceous and Shrubby Vegetation 10 #d6bc74
2.1. Campo Alagado e Área Pantanosa 2.1. Wetland 11 #519799
2.2. Formação Campestre 2.2. Grassland 12 #d6bc74
2.3. Apicum 2.3. Hypersaline Tidal Flat 32 #fc8114
2.4. Afloramento Rochoso 2.4. Rocky Outcrop 29 #ffaa5f
2.5. Restinga Herbácea 2.5. Herbaceous Sandbank Vegetation 50 #ad5100
3. Agropecuária 3. Farming 14 #ffefc3
3.1. Pastagem 3.1. Pasture 15 #edde8e
3.2. Agricultura 3.2. Agriculture 18 #E974ED
 3.2.1. Lavoura Temporária 3.2.1. Temporary Crop 19 #C27BA0
 3.2.1.1. Soja 3.2.1.1. Soybean 39 #f5b3c8
 3.2.1.2. Cana 3.2.1.2. Sugar cane 20 #db7093
 3.2.1.3. Arroz 3.2.1.3. Rice 40 #c71585
 3.2.1.4. Algodão (beta) 3.2.1.4. Cotton (beta) 62 #ff69b4
 3.2.1.5. Outras Lavouras Temporárias 3.2.1.5. Other Temporary Crops 41 #f54ca9
 3.2.2. Lavoura Perene 3.2.2. Perennial Crop 36 #d082de
 3.2.2.1. Café 3.2.2.1. Coffee 46 #d68fe2
 3.2.2.2. Citrus 3.2.2.2. Citrus 47 #9932cc
 3.2.2.3. Dendê 3.2.2.3. Palm Oil 35 #9065d0
 3.2.2.4. Outras Lavouras Perenes 3.2.2.4. Other Perennial Crops 48 #e6ccff
3.3. Silvicultura 3.3. Forest Plantation 9 #7a5900
3.4. Mosaico de Usos 3.4. Mosaic of Uses 21 #ffefc3
4. Área não Vegetada 4. Non vegetated area 22 #d4271e
4.1. Praia, Duna e Areal 4.1. Beach, Dune and Sand Spot 23 #ffa07a
4.2. Área Urbanizada 4.2. Urban Area 24 #d4271e
4.3. Mineração 4.3. Mining 30 #9c0027
4.4. Usina Fotovoltaica (beta) 4.4 Photovoltaic Power Plant (beta) 75 #c12100
4.5. Outras Áreas não Vegetadas 4.5. Other non Vegetated Areas 25 #db4d4f
5. Corpo D'água 5. Water 26 #2532e4
5.1 Rio, Lago e Oceano 5.1. River, Lake and Ocean 33 #2532e4
5.2 Aquicultura 5.2. Aquaculture 31 #091077
6. Não observado 6. Not Observed 27 #ffffff
"""

import ee
from datetime import datetime

# ---------- User configuration ----------
YEARS = [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022]  # Years to process
AOI_ASSET = "projects/ee-zcs/assets/AMZ2018M"         # FeatureCollection

# TerraClass class definitions
FOREST_TERRA = [1, 2, 9]        # Primary forest, Secondary forest, Silviculture
CROPLAND_TERRA = [12, 13, 14, 15]  # Perennial crop, Semi-perennial crop, Annual crop (1 cycle), Annual crop (>1 cycle)
PASTURE_TERRA = [10, 11]        # Shrubby/tree pasture, Herbaceous pasture

# MapBiomas class definitions
FOREST_MB = [3, 4, 5, 6, 49]    # Forest Formation, Savanna Formation, Mangrove, Floodable Forest, Wooded Sandbank
CROPLAND_MB = [19, 20, 39, 40, 41, 62, 36, 46, 47, 35, 48]  # All temporary and perennial crops
PASTURE_MB = [15]               # Pasture

NODATA = 255
EXPORT_FOLDER = "GEE_exports"  # Google Drive folder
SCALE = 30
TILE_SCALE = 4                  # increase if region large
NUM_SAMPLES_TO_KEEP = 100       # Number of agreement samples to keep per year
# ---------------------------------------
pasture_samples = ee.FeatureCollection("projects/ee-zcs/assets/samples-BRAZIL")

def init_ee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

def get_aoi():
    return ee.Image(AOI_ASSET).geometry()

def reclass_to_binary(image, land_cover_classes):
    """
    Reclassifies an image to binary: 1 for specified land cover classes, 0 otherwise.
    
    Args:
        image (ee.Image): The input image to reclassify.
        land_cover_classes (list): List of class values to map to 1 (target land cover).
    
    Returns:
        ee.Image: Binary image with 1 for target land cover and 0 for others.
    """
    return image.remap(
        land_cover_classes,
        ee.List.repeat(1, len(land_cover_classes)),
        0
    )

def build_terra(year, land_cover_type):
    """
    Build binary land cover image from TerraClass data.
    
    Args:
        year: Year of data
        land_cover_type: 'forest', 'cropland', or 'pasture'
    """
    TERRA_ASSET = f"projects/ee-zcs/assets/AMZ{year}M"
    src = ee.Image(TERRA_ASSET).select('b1')
    
    # Select appropriate classes based on land cover type
    if land_cover_type == 'forest':
        classes = FOREST_TERRA
    elif land_cover_type == 'cropland':
        classes = CROPLAND_TERRA
    elif land_cover_type == 'pasture':
        classes = PASTURE_TERRA
    else:
        raise ValueError(f"Unknown land cover type: {land_cover_type}")
    
    binary = reclass_to_binary(src, classes)
    return binary.rename("reference")

def build_mb(year, land_cover_type):
    """
    Build binary land cover image from MapBiomas data.
    
    Args:
        year: Year of data
        land_cover_type: 'forest', 'cropland', or 'pasture'
    """
    src = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2').select(f'classification_{year}')
    
    # Select appropriate classes based on land cover type
    if land_cover_type == 'forest':
        classes = FOREST_MB
    elif land_cover_type == 'cropland':
        classes = CROPLAND_MB
    elif land_cover_type == 'pasture':
        classes = PASTURE_MB
    else:
        raise ValueError(f"Unknown land cover type: {land_cover_type}")
    
    binary = reclass_to_binary(src, classes)
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

    # Convert to integers to avoid float precision issues
    tn = ee.Number(histogram.get('0', 0)).toInt64()
    fp = ee.Number(histogram.get('1', 0)).toInt64()
    fn = ee.Number(histogram.get('10', 0)).toInt64()
    tp = ee.Number(histogram.get('11', 0)).toInt64()

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

def export_metrics(year, land_cover_type, metrics):
    feature = ee.Feature(
        None, metrics
    )
    task = ee.batch.Export.table.toDrive(
        collection=ee.FeatureCollection([feature]),
        description=f"confusion_{land_cover_type}_{year}",
        folder=EXPORT_FOLDER,
        fileNamePrefix=f"confusion_{land_cover_type}_{year}",
        fileFormat="CSV"
    )
    task.start()
    print(f"🚀 Started metrics export for {land_cover_type} {year}: {task.id}")

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

def filter_samples_by_agreement(year, samples, reference, prediction):
    """
    Filter samples to keep only those in areas where both datasets agree.
    
    Args:
        year: Year of data
        samples: ee.FeatureCollection of existing samples with 'Year' property
        reference: TerraClass binary image
        prediction: MapBiomas binary image
    
    Returns:
        tuple: (filtered_samples, agreement_percentage, total_count, agreement_count)
    """
    # Filter samples for this year
    year_samples = samples.filter(ee.Filter.eq('year', year)).filter(ee.Filter.eq('reference', 15))
    
    # Create valid data mask (where both datasets have data)
    valid_mask = reference.mask().And(prediction.mask())
    
    # Sample the valid mask at sample points to filter out samples outside the study area
    samples_with_valid = valid_mask.reduceRegions(
        collection=year_samples,
        reducer=ee.Reducer.first(),
        scale=SCALE
    )
    
    # Keep only samples within valid data area
    valid_samples = samples_with_valid.filter(ee.Filter.eq('first', 1))
    
    # Find pixels where both datasets agree on class = 1 (both have pasture)
    agreement = reference.eq(1).And(prediction.eq(1))
    
    # Sample the agreement image at valid sample points
    samples_with_agreement = agreement.reduceRegions(
        collection=valid_samples,
        reducer=ee.Reducer.first(),
        scale=SCALE
    )
    
    # Filter to keep only samples where agreement = 1
    agreed_samples = samples_with_agreement.filter(ee.Filter.eq('first', 1))
    
    # Get counts (now total_count only includes samples within valid data area)
    total_count = valid_samples.size()
    agreement_count = agreed_samples.size()
    
    # Calculate percentage
    agreement_percentage = agreement_count.divide(total_count).multiply(100)
    
    # Randomly select NUM_SAMPLES_TO_KEEP samples from agreed samples
    # Add a random column for sampling
    agreed_samples_random = agreed_samples.randomColumn('random', seed=year)
    filtered_samples = agreed_samples_random.limit(NUM_SAMPLES_TO_KEEP, 'random')
    
    return filtered_samples, agreement_percentage, total_count, agreement_count

def export_filtered_samples(year, samples, stats):
    """
    Export filtered sample points to Google Drive.
    
    Args:
        year: Year of data
        samples: ee.FeatureCollection of filtered samples
        stats: dict with agreement statistics
    """
    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=f"filtered_pasture_samples_{year}",
        folder=EXPORT_FOLDER,
        fileNamePrefix=f"filtered_pasture_samples_{year}",
        fileFormat="KML"
    )
    task.start()
    print(f"📍 Exported samples for year {year}: {task.id}")
    # print(f"   Total samples: {stats['total']}, Agreement: {stats['agreement']} ({stats['percentage']:.2f}%)")
    # print(f"   Kept: {stats['kept_count']} samples")

def main():
    init_ee()
    region = get_aoi()
    
    # Focus on pasture only for sample filtering
    land_cover_type = 'pasture'
    
    # Store statistics for all years
    all_stats = []

    for year in YEARS:
        print(f"\n{'='*50}")
        print(f"Year {year}")
        print(f"{'='*50}")
        
        print(f"\n--- Processing {land_cover_type} ---")
        
        # Build binary images
        reference = build_terra(year, land_cover_type)
        prediction = build_mb(year, land_cover_type)

        # Compute confusion metrics for the whole region
        # results = compute_confusion(reference, prediction, region)
        # export_metrics(year, land_cover_type, results)
        
        # Filter existing samples by agreement
        try:
            filtered_samples, agreement_pct, total_count, agreement_count = filter_samples_by_agreement(
                year, pasture_samples, reference, prediction
            )
            
            # Create a server-side feature with the stats (no getInfo)
            stats_feature = ee.Feature(None, {
                'year': ee.Number(year),
                'total': total_count,
                'agreement': agreement_count,
                'percentage': agreement_pct,
                'kept_count': agreement_count.min(NUM_SAMPLES_TO_KEEP)
            })

            # Export the stats to Drive as CSV
            ee.batch.Export.table.toDrive(
                collection=ee.FeatureCollection([stats_feature]),
                description=f"sample_agreement_stats_{year}",
                folder=EXPORT_FOLDER,
                fileNamePrefix=f"sample_agreement_stats_{year}",
                fileFormat="CSV"
            ).start()
            print(f"📤 Export started: agreement stats for {year} to Drive")

            # # Append a lightweight local summary (avoid client-side getInfo).
            # # This is only for local logging; the full numeric stats are in the exported CSV.
            # stats = {
            #     'year': year,
            #     'total': 0,
            #     'agreement': 0,
            #     'percentage': 0.0,
            #     'kept_count': NUM_SAMPLES_TO_KEEP
            # }
            # all_stats.append(stats)
            
            # Export filtered samples
            # export_filtered_samples(year, filtered_samples, None)

        except Exception as e:
            print(f"⚠️  Error filtering samples for {year}: {e}")

    # print("\n" + "="*50)
    # print("SUMMARY - Sample Agreement Statistics")
    # print("="*50)
    # for stats in all_stats:
    #     print(f"Year {stats['year']}: {stats['agreement']}/{stats['total']} samples agree ({stats['percentage']:.2f}%), kept {stats['kept_count']}")
    
    # print("\n" + "="*50)
    # print("All export tasks launched.")
    # print("Monitor status at https://code.earthengine.google.com/tasks")
    # print("="*50)

if __name__ == "__main__":
    main()
