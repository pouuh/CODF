# CODF Project Data Sources Documentation

**Generated:** 2025-10-31 15:16:08

This document catalogs all data sources used across the CODF project notebooks.

---

## 🗺️ Quick File Location Reference (for AI Programming)

**Key Data Directories:**
- **CODF Geometries:** `/usr2/postdoc/chishan/project_data/CODF/CODF_Chishan.gpkg`
- **TerraClass (AMZ files):** `/projectnb/modislc/users/chishan/data/TerraClass/` or `/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/AMZ.{year}.M.tif`
- **GLANCE Products:** `/projectnb/measures/products/SA/v001/DAAC/LC/GLANCE.A{YEAR}0701.h*v*.001.*.SA.LC.tif`
- **MapBiomas Samples:** `/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/mapbiomas_85k_col2_1_points_english.csv`
- **MapBiomas Boundaries:** `/projectnb/modislc/users/chishan/data/MapBiomas/BR_Municipios_2021/BR_Municipios_2021.shp`
- **GLANCE STAC Catalog:** `/projectnb/modislc/users/chishan/stac_glance_SA_fixed_m/catalog.json`

**Google Earth Engine Assets:**
- **CODF:** `projects/ee-zcs/assets/CODF_{polygons|points|lines}`
- **MapBiomas:** `projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2`
- **Hansen Forest:** `UMD/hansen/global_forest_change_2024_v1_12`
- **GLANCE:** `projects/GLANCE/DATASETS/V001`
- **WDPA Protected:** `WCMC/WDPA/current/polygons`

**Important Note:** AMZ.{year}.M.tif files are **TerraClass data**, not MapBiomas data (despite being in MapBiomas folder).

---

## Table of Contents

- [Quick File Location Reference](#-quick-file-location-reference-for-ai-programming)
- [Google Earth Engine (GEE) Data](#google-earth-engine-gee-data)
  - [GEE Assets](#gee-assets)
  - [GEE Public Datasets](#gee-public-datasets)
- [Static/Reference Datasets](#staticreference-datasets)
- [Local Data Files](#local-data-files)
- [External Data Sources (URLs)](#external-data-sources-urls)
- [Summary Statistics](#summary-statistics)

---

## Google Earth Engine (GEE) Data

### GEE Assets

These are custom assets uploaded to or created in Google Earth Engine projects.

#### CODF Project Assets

**`projects/ee-zcs/assets/CODF_lines`**
- **Description:** CODF investment database geometries
- **Type:** Vector (FeatureCollection)
- **Usage:** Storing CODF investment locations (points, lines, polygons)
- **Access:** Project asset in `projects/ee-zcs/assets/`
- **Used in 4 notebook(s):** 20250910_2.ipynb, 20250912.ipynb, 20250916.ipynb, 20250917.ipynb

**`projects/ee-zcs/assets/CODF_points`**
- **Description:** CODF investment database geometries
- **Type:** Vector (FeatureCollection)
- **Usage:** Storing CODF investment locations (points, lines, polygons)
- **Access:** Project asset in `projects/ee-zcs/assets/`
- **Used in 2 notebook(s):** 20250916.ipynb, 20250917.ipynb

**`projects/ee-zcs/assets/CODF_polygons`**
- **Description:** CODF investment database geometries
- **Type:** Vector (FeatureCollection)
- **Usage:** Storing CODF investment locations (points, lines, polygons)
- **Access:** Project asset in `projects/ee-zcs/assets/`
- **Used in 5 notebook(s):** 20250910_2.ipynb, 20250912.ipynb, 20250916.ipynb, 20250917.ipynb, 20251016.ipynb

#### GLANCE Project Assets

**`projects/GLANCE/ANCILLARY/CONTINENTS_EXPORT/GLANCE_V01_AF_PROJ_LAND_buf`**
- **Description:** GLANCE land cover dataset or related assets
- **Type:** ImageCollection or FeatureCollection
- **Usage:** Land cover classification and analysis
- **Access:** Project asset in GLANCE project
- **Used in 1 notebook(s):** 20250919.ipynb

**`projects/GLANCE/CASE_STUDIES/Clustering/EB_clusters/AS_SR43_KMeans_50cl_50k2020`**
- **Description:** GLANCE land cover dataset or related assets
- **Type:** ImageCollection or FeatureCollection
- **Usage:** Land cover classification and analysis
- **Access:** Project asset in GLANCE project
- **Used in 1 notebook(s):** 20250919.ipynb

**`projects/GLANCE/DATASETS/GLANCE_V01_AF`**
- **Description:** GLANCE land cover dataset or related assets
- **Type:** ImageCollection or FeatureCollection
- **Usage:** Land cover classification and analysis
- **Access:** Project asset in GLANCE project
- **Used in 1 notebook(s):** 20250919.ipynb

**`projects/GLANCE/DATASETS/V001`**
- **Description:** GLANCE land cover dataset or related assets
- **Type:** ImageCollection or FeatureCollection
- **Usage:** Land cover classification and analysis
- **Access:** Project asset in GLANCE project
- **Used in 5 notebook(s):** 20251016.ipynb, 20251018.ipynb, 20251027.ipynb
  and 2 more...

**`projects/GLANCE/SCRATCH/TEST_EMBEDDINGS_10m_AF_Training_Master_V1_2025_09_04_filtered_predictors_subrealm_19_all`**
- **Description:** GLANCE land cover dataset or related assets
- **Type:** ImageCollection or FeatureCollection
- **Usage:** Land cover classification and analysis
- **Access:** Project asset in GLANCE project
- **Used in 1 notebook(s):** 20250923.ipynb

#### Other Project Assets

**`projects/ee-zcs/assets/AMZ`**
- **Description:** Brazilian administrative or regional boundaries
- **Type:** Asset
- **Used in 1 notebook(s):** 20251027.ipynb

**`projects/ee-zcs/assets/AMZ2018M`**
- **Description:** Brazilian administrative or regional boundaries
- **Type:** Asset
- **Used in 2 notebook(s):** 20251018.ipynb, 20251027.ipynb

**`projects/ee-zcs/assets/BR_Municipios_2021`**
- **Description:** Brazilian administrative or regional boundaries
- **Type:** Asset
- **Used in 2 notebook(s):** 20251027.ipynb, 20251029.ipynb

**`projects/ee-zcs/assets/encoded_forest_`**
- **Description:** Custom project asset
- **Type:** Asset
- **Used in 1 notebook(s):** 20251018.ipynb

**`projects/ee-zcs/assets/encoded_forest_2018`**
- **Description:** Custom project asset
- **Type:** Asset
- **Used in 1 notebook(s):** 20251027.ipynb

**`projects/ee-zcs/assets/prodes_amazonia_legal_2024`**
- **Description:** PRODES deforestation monitoring data
- **Type:** Asset
- **Used in 1 notebook(s):** 20251029.ipynb

**`projects/ee-zcs/assets/samples-BRAZIL`**
- **Description:** Custom project asset
- **Type:** Asset
- **Used in 1 notebook(s):** 20251027.ipynb

**`projects/google/embeddings/v1`**
- **Description:** Custom project asset
- **Type:** Asset
- **Used in 1 notebook(s):** 20250919.ipynb

**`projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2`**
- **Description:** MapBiomas land cover data
- **Type:** Asset
- **Used in 3 notebook(s):** 20251027.ipynb, 20251029.ipynb, 20251030_2.ipynb

**`projects/your-project/training_data`**
- **Description:** Custom project asset
- **Type:** Asset
- **Used in 1 notebook(s):** 20250919.ipynb

### GEE Public Datasets

These are publicly available datasets in Google Earth Engine's data catalog.

#### Hansen Global Forest Change

**`UMD/hansen/global_forest_change_2024_v1_12`**
- **Description:** Global forest cover loss and gain dataset
- **Spatial Resolution:** 30 meters
- **Temporal Coverage:** 2000-2024
- **Variables:** Forest cover, loss year, gain, loss by year
- **Use Case:** Forest change detection, deforestation analysis
- **Documentation:** [UMD Hansen Dataset](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2024_v1_12)
- **Used in 3 notebook(s):** 20250916.ipynb, 20250917.ipynb, 20251027.ipynb

#### Protected Areas (WDPA)

**`WCMC/WDPA/current/polygons`**
- **Description:** World Database on Protected Areas
- **Type:** Vector (polygons)
- **Use Case:** Protected area analysis, conservation planning
- **Documentation:** [WDPA Dataset](https://developers.google.com/earth-engine/datasets/catalog/WCMC_WDPA_current_polygons)
- **Used in 4 notebook(s):** 20250910_2.ipynb, 20250912.ipynb, 20250916.ipynb, 20250917.ipynb

#### Landsat Collections

**`LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI`**
- **Description:** Landsat satellite imagery
- **Spatial Resolution:** 30 meters
- **Use Case:** Time series analysis, CCDC, vegetation monitoring
- **Documentation:** [Landsat Collections](https://developers.google.com/earth-engine/datasets/catalog/landsat)
- **Used in 1 notebook(s):** 20251029.ipynb

**`LANDSAT/LC08/C02/T1_L2`**
- **Description:** Landsat satellite imagery
- **Spatial Resolution:** 30 meters
- **Use Case:** Time series analysis, CCDC, vegetation monitoring
- **Documentation:** [Landsat Collections](https://developers.google.com/earth-engine/datasets/catalog/landsat)
- **Used in 3 notebook(s):** 20251029.ipynb, 20251030.ipynb, 20251030_2.ipynb

**`LANDSAT/LC09/C02/T1_L2`**
- **Description:** Landsat satellite imagery
- **Spatial Resolution:** 30 meters
- **Use Case:** Time series analysis, CCDC, vegetation monitoring
- **Documentation:** [Landsat Collections](https://developers.google.com/earth-engine/datasets/catalog/landsat)
- **Used in 2 notebook(s):** 20251029.ipynb, 20251030.ipynb

**`LANDSAT/LE07/C02/T1_L2`**
- **Description:** Landsat satellite imagery
- **Spatial Resolution:** 30 meters
- **Use Case:** Time series analysis, CCDC, vegetation monitoring
- **Documentation:** [Landsat Collections](https://developers.google.com/earth-engine/datasets/catalog/landsat)
- **Used in 2 notebook(s):** 20251029.ipynb, 20251030.ipynb

**`LANDSAT/LT05/C02/T1_L2`**
- **Description:** Landsat satellite imagery
- **Spatial Resolution:** 30 meters
- **Use Case:** Time series analysis, CCDC, vegetation monitoring
- **Documentation:** [Landsat Collections](https://developers.google.com/earth-engine/datasets/catalog/landsat)
- **Used in 2 notebook(s):** 20251029.ipynb, 20251030.ipynb

#### Other Public Datasets

**`your_filtered_codf_polygons`**
- **Used in 1 notebook(s):** 20250917.ipynb

---

## Static/Reference Datasets

These are well-known reference datasets used for analysis.

### Hansen Global Forest Change

- **Description:** Global forest cover change dataset from University of Maryland
- **Spatial Resolution:** 30 meters
- **Temporal Coverage:** 2000-2024
- **Access:** Google Earth Engine, Global Forest Watch
- **Primary Use Case:** Baseline forest loss analysis, deforestation detection
- **Used in 6 notebook(s):** 20250916.ipynb, 20250917.ipynb, 20251006.ipynb, 20251027.ipynb, 20251028_2.ipynb
  and 1 more...

### Landsat

- **Description:** NASA/USGS Landsat satellite program
- **Spatial Resolution:** 30 meters
- **Temporal Coverage:** 1972-present
- **Access:** USGS Earth Explorer, Google Earth Engine
- **Primary Use Case:** Long-term time series analysis, CCDC algorithm input
- **Used in 6 notebook(s):** 20250910_2.ipynb, 20251006.ipynb, 20251024.ipynb, 20251029.ipynb, 20251030.ipynb
  and 1 more...

### MODIS

- **Description:** Moderate Resolution Imaging Spectroradiometer satellite data
- **Spatial Resolution:** 250m-1km
- **Temporal Coverage:** 2000-present
- **Access:** NASA Earth Data, Google Earth Engine
- **Primary Use Case:** Land cover classification, vegetation indices, GLANCE product basis
- **Used in 17 notebook(s):** 20250916.ipynb, 20250919.ipynb, 20251006.ipynb, 20251015.ipynb, 20251016.ipynb
  and 12 more...

### Sentinel

- **Description:** European Space Agency Sentinel satellite constellation
- **Spatial Resolution:** 10-60 meters
- **Temporal Coverage:** 2015-present (Sentinel-2)
- **Access:** Copernicus Open Access Hub, Planetary Computer
- **Primary Use Case:** High-resolution land cover mapping, change detection
- **Used in 4 notebook(s):** 20250910_2.ipynb, 20251006.ipynb, 20251014.ipynb, 20251018.ipynb

---

## Local Data Files

These are data files stored locally or on shared file systems.

### Vector Data Files (Shapefiles, GeoPackage, GeoJSON)

Vector datasets used for spatial analysis and boundaries.

**`*.shp`**
- **Description:** Vector dataset
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20251027.ipynb

**`/Users/samzhang/Downloads/mapbiomas_amazon_sample_100.shp`**
- **Description:** Amazon region boundary
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20251021.ipynb

**`/path/to/your_AOI.gpkg`**
- **Description:** Vector dataset
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20251006.ipynb

**`/project_data/CODF/CODF_Chishan.gpkg`**
- **Description:** CODF investment geometries
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20251006.ipynb

**`/projectnb/measures/datasets/Grids/FILTERED/json/GLANCE_V01_AS_PROJ_TILE_FILTERED.geojson`**
- **Description:** Vector dataset
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20251016.ipynb

**`/projectnb/modislc/users/chishan/data/MapBiomas/BR_Municipios_2021/BR_Municipios_2021.shp`**
- **Description:** Municipal boundaries
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20251027.ipynb

**`/projectnb/modislc/users/chishan/data/MapBiomas/Limites_Amazonia_Legal_2024_shp/Limites_Amazonia_Legal_2024.shp`**
- **Description:** Amazon region boundary
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20251021.ipynb

**`/usr2/postdoc/chishan/project_data/CODF/CODF_Chishan.gpkg`**
- **Description:** CODF investment geometries
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20250917.ipynb

**`/usr2/postdoc/chishan/project_data/MapBiomas/MAPBIOMAS/mapbiomas_amazon_sample_100.shp`**
- **Description:** Amazon region boundary
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20251014.ipynb

**`CODF_Chishan.gpkg`**
- **Description:** CODF investment geometries
- **Format:** Vector (Shapefile/GeoPackage/GeoJSON)
- **Used in:** 20251006.ipynb

*...and 1 more vector files*

### Raster Data Files (GeoTIFF)

Raster datasets for land cover, forest classification, and analysis.

#### GLANCE Land Cover Products

- **Description:** GLANCE (Global Land Cover) MODIS-based land cover classification
- **Resolution:** 500m
- **Format:** GeoTIFF (COG - Cloud Optimized GeoTIFF)
- **Location:** `/projectnb/measures/products/SA/v001/DAAC/LC/`
- **File Pattern:** `GLANCE.A{YEAR}0701.h*v*.001.*.SA.LC.tif`
- **Used in 7 file reference(s)**

#### TerraClass Amazon

- **Description:** Land use and land cover mapping for Brazilian Amazon (AMZ = Amazon Legal region)
- **Resolution:** 30 meters
- **Format:** GeoTIFF (COG available)
- **Temporal Coverage:** Multiple years (2008, 2010, 2012, 2014, 2016, 2018, etc.)
- **Source:** INPE (Brazilian National Institute for Space Research)
- **Primary Location:** `/projectnb/modislc/users/chishan/data/TerraClass/`
- **Also stored in:** `/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/` (note: despite folder name)
- **File Pattern:** `AMZ.{year}.M.tif` or `AMZ.{year}.M.cog.tif`
- **Note:** AMZ files are TerraClass data, not MapBiomas data
- **Used in 12 file reference(s)**

#### MapBiomas Collection

- **Description:** Annual land use and land cover maps for Brazil (primarily accessed via GEE)
- **Resolution:** 30 meters
- **Format:** Google Earth Engine ImageCollection
- **Temporal Coverage:** 1985-present (Collection 10)
- **GEE Asset:** `projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2`
- **Local Sample Data:** `/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/mapbiomas_85k_col2_1_points_english.csv`
- **Forest Classes:** 1, 3, 4, 5, 6, 49
- **Used primarily via GEE in multiple notebooks**

#### Other Raster Files

- `**/*Amazon*.tif` - Used in: 20251015.ipynb
- `**/*amazon*.tif` - Used in: 20251015.ipynb
- `**/*mapbiomas*.tif` - Used in: 20251015.ipynb
- `.M.tif` - Used in: 20251015.ipynb
- `.tif` - Used in: 20251015.ipynb
- *...and 6 more raster files*

### Tabular Data Files (CSV)

CSV files containing analysis results, sample points, and reference data.

#### MapBiomas Sample/Reference Data

- **`/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/mapbiomas_85k_col2_1_points_english.csv`**
  - Sample points or reference data for validation
  - Used in: 20251029.ipynb, 20251030.ipynb
- **`mapbiomas_85k_col2_1_points_english.csv`**
  - Sample points or reference data for validation
  - Used in: 20251028.ipynb
- **`mapbiomas_85k_col2_1_points_w_edge_and_edited_v1.csv.csv`**
  - Sample points or reference data for validation
  - Used in: 20251028.ipynb
- **`mapbiomas_ts_3000_pts_wide.csv`**
  - Sample points or reference data for validation
  - Used in: 20251030_2.ipynb
- **`municipality_loss_2009_2022_MapBiomas.csv`**
  - Sample points or reference data for validation
  - Used in: 20251028_2.ipynb, 20251029.ipynb

#### Analysis Results

- Forest loss analysis by municipality/region
- Accuracy assessment and confusion matrices
- Comparison results between different datasets
- **9 result file(s) referenced**

**Total CSV files referenced:** 24

---

## External Data Sources (URLs)

External data catalogs and APIs accessed by the notebooks.

### http://www.opengis.net/kml/2.2

- **Description:** OGC KML standard namespace
- **Usage:** Data access and retrieval
- **Used in 1 notebook(s):** 20250916.ipynb

### https://code.earthengine.google.com/tasks

- **Description:** Google Earth Engine task monitoring
- **Usage:** Data access and retrieval
- **Used in 1 notebook(s):** 20251018.ipynb

### https://data.inpe.br/bdc/stac/v1/

- **Description:** Brazil Data Cube STAC catalog (INPE)
- **Usage:** Data access and retrieval
- **Used in 1 notebook(s):** 20251015.ipynb

### https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x

- **Description:** Microsoft Planetary Computer STAC API - Access to Sentinel, Landsat, and other Earth observation data
- **Usage:** Data access and retrieval
- **Used in 1 notebook(s):** 20251014.ipynb

### https://planetarycomputer.microsoft.com/api/stac/v1

- **Description:** Microsoft Planetary Computer STAC API - Access to Sentinel, Landsat, and other Earth observation data
- **Usage:** Data access and retrieval
- **Used in 4 notebook(s):** 20250910_2.ipynb, 20251006.ipynb, 20251014.ipynb, 20251024.ipynb

---

## Summary Statistics

- **Total Notebooks Analyzed:** 24
- **Unique GEE Assets:** 18
- **Unique GEE Public Datasets:** 8
- **Static/Reference Datasets:** 4
- **Local Files Referenced:** 69
  - Vector files: 11
  - Raster files: 30
  - CSV files: 24
- **External URLs:** 5

---

## How to Use This Documentation


### Setting Up Data Access


#### Google Earth Engine

```python
import ee
ee.Initialize(project='ee-zcs')  # Required for CODF project assets
```

#### CODF Assets
```python
# Load CODF geometries
codf_polygons = ee.FeatureCollection('projects/ee-zcs/assets/CODF_polygons')
codf_points = ee.FeatureCollection('projects/ee-zcs/assets/CODF_points')
codf_lines = ee.FeatureCollection('projects/ee-zcs/assets/CODF_lines')
```

#### Hansen Forest Change
```python
# Load Hansen dataset
hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')
forest_loss = hansen.select('loss')
loss_year = hansen.select('lossyear')
```

### Local File Access

Most local files are stored on shared file systems:
- **CODF data:** `/usr2/postdoc/chishan/project_data/CODF/`
- **MapBiomas data:** `/projectnb/modislc/users/chishan/data/MapBiomas/`
  - Sample points: `mapbiomas_85k_col2_1_points_english.csv`
  - Boundaries: `BR_Municipios_2021/BR_Municipios_2021.shp`
  - **Note:** AMZ files in MAPBIOMAS folder are actually TerraClass data
- **GLANCE products:** `/projectnb/measures/products/SA/v001/DAAC/LC/`
  - Pattern: `GLANCE.A{YEAR}0701.h{tile_h}v{tile_v}.001.*.SA.LC.tif`
- **TerraClass data:** 
  - Primary: `/projectnb/modislc/users/chishan/data/TerraClass/`
  - Also in: `/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/`
  - Pattern: `AMZ.{year}.M.tif` or `AMZ.{year}.M.cog.tif`

#### Common File Patterns
```python
# TerraClass (AMZ = Amazon Legal region)
terraclass_file = f"/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/AMZ.{year}.M.tif"
# or
terraclass_cog = f"/projectnb/modislc/users/chishan/data/MapBiomas/COG/AMZ.{year}.M.cog.tif"

# GLANCE tiles
glance_pattern = "/projectnb/measures/products/SA/v001/DAAC/LC/GLANCE.A{year}0701.h*v*.001.*.SA.LC.tif"

# STAC catalog
glance_stac = "/projectnb/modislc/users/chishan/stac_glance_SA_fixed_m/catalog.json"
```

### External Data Access


#### Microsoft Planetary Computer
```python
from pystac_client import Client
import planetary_computer

catalog = Client.open(
    'https://planetarycomputer.microsoft.com/api/stac/v1',
    modifier=planetary_computer.sign_inplace
)
```

### Data Processing Patterns


#### Buffer Analysis
```python
# Standard buffer distances (km)
buffer_distances = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0]
```

#### Forest Threshold
```python
# Standard forest cover threshold
forest_threshold = 30  # percent tree cover
```

#### Time Periods
```python
# Common analysis periods
hansen_period = (2001, 2024)  # Forest loss years
mapbiomas_period = (1985, 2023)  # Full MapBiomas collection
codf_cutoff = 2015  # Pre-investment period
```

---

*Documentation generated from 24 notebooks*