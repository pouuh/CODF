"""
Hansen Global Forest Change Analysis with Protected Areas Overlay - Simplified Version

Purpose: Create separate feature collections for protected and non-protected portions,
then analyze forest loss for each separately and export to different CSV files.

Approach:
1. Create buffered CODF sites
2. Intersect with WCMC protected areas -> Protected portions  
3. Subtract protected areas from buffers -> Non-protected portions
4. Calculate forest loss metrics for each separately
5. Export as separate CSV files

Time window: 2001-2024 (Hansen forest loss), 2000 baseline forest cover
"""

import ee
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Initialize Google Earth Engine
ee.Initialize(project='ee-zcs')

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Load WCMC Protected Areas
wcmc_protected_areas = ee.FeatureCollection('WCMC/WDPA/current/polygons')

# Load Hansen Global Forest Change dataset
HANSEN = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

# Buffer distances (meters)
buffers = [5000, 10000, 15000, 20000, 25000, 50000]

# ============================================================================
# HANSEN DATA PROCESSING
# ============================================================================

# Extract Hansen components
treecover2000 = HANSEN.select('treecover2000')
lossyear = HANSEN.select('lossyear')
datamask = HANSEN.select('datamask')

# Define analysis parameters
analysis_start_year = 2001
analysis_end_year = 2024
forest_threshold = 30  # 30% tree cover threshold

# Create initial forest mask (year 2000, >=30% tree cover)
initial_forest_2000 = treecover2000.gte(forest_threshold).And(datamask.eq(1))

def create_hansen_annual_deforestation():
    """Create annual deforestation images (2001-2024)"""
    annual_deforestation_images = []
    
    for year in range(analysis_start_year, analysis_end_year + 1):
        # Hansen uses year codes (1 = 2001, 2 = 2002, etc.)
        hansen_year_code = year - 2000
        
        # Deforestation in this year = was forest in 2000 AND lost in this year
        annual_defor = initial_forest_2000.And(lossyear.eq(hansen_year_code))
        annual_deforestation_images.append(
            annual_defor.rename(f'hansen_defor_{year}')
        )
    
    return ee.Image.cat(annual_deforestation_images)

# Create the Hansen annual deforestation stack
hansen_annual_deforestation = create_hansen_annual_deforestation()

# ============================================================================
# SIMPLIFIED GEOMETRY PROCESSING
# ============================================================================

def create_protected_portions(filtered_codf_polygons, buffer_distances):
    """Create feature collections for PROTECTED portions only"""
    
    protected_collections = {}
    
    for buffer_dist in buffer_distances:
        print(f"Creating PROTECTED portions for {buffer_dist/1000}km buffer...")
        
        # Step 1: Create buffered features
        def create_buffer(feature):
            return (feature.buffer(buffer_dist)
                   .set('buffer_dist', buffer_dist)
                   .set('buffer_km', buffer_dist/1000)
                   .set('original_BU_ID', feature.get('BU_ID'))
                   .copyProperties(feature))
        
        buffered_features = filtered_codf_polygons.map(create_buffer)
        
        # Step 2: Intersect with protected areas to get PROTECTED portions
        def get_protected_portion(buffered_feature):
            buffered_geom = buffered_feature.geometry()
            
            # Find intersecting protected areas
            intersecting_pas = wcmc_protected_areas.filterBounds(buffered_geom)
            
            # Only process if there are intersecting PAs
            def create_intersection():
                pas_union = intersecting_pas.geometry().dissolve()
                protected_geom = buffered_geom.intersection(pas_union, ee.ErrorMargin(1))
                
                return buffered_feature.setGeometry(protected_geom).set({
                    'protection_status': 'protected',
                    'pa_count': intersecting_pas.size(),
                    'has_pa_overlap': True,
                    'valid_geometry': True
                })
            
            # Return feature only if there's actual overlap
            return ee.Algorithms.If(
                intersecting_pas.size().gt(0),
                create_intersection(),
                buffered_feature.set('valid_geometry', False)  # Return invalid feature instead of null
            )
        
        # Map and filter out invalid geometries
        protected_portions_raw = buffered_features.map(get_protected_portion)
        protected_portions = protected_portions_raw.filter(ee.Filter.eq('valid_geometry', True))
        
        protected_collections[buffer_dist] = protected_portions
        print(f"Protected portions created for {buffer_dist/1000}km buffer")
    
    return protected_collections

def create_non_protected_portions(filtered_codf_polygons, buffer_distances):
    """Create feature collections for NON-PROTECTED portions only"""
    
    non_protected_collections = {}
    
    for buffer_dist in buffer_distances:
        print(f"Creating NON-PROTECTED portions for {buffer_dist/1000}km buffer...")
        
        # Step 1: Create buffered features
        def create_buffer(feature):
            return (feature.buffer(buffer_dist)
                   .set('buffer_dist', buffer_dist)
                   .set('buffer_km', buffer_dist/1000)
                   .set('original_BU_ID', feature.get('BU_ID'))
                   .copyProperties(feature))
        
        buffered_features = filtered_codf_polygons.map(create_buffer)
        
        # Step 2: Subtract protected areas to get NON-PROTECTED portions
        def get_non_protected_portion(buffered_feature):
            buffered_geom = buffered_feature.geometry()
            
            # Find intersecting protected areas
            intersecting_pas = wcmc_protected_areas.filterBounds(buffered_geom)
            
            def create_difference():
                pas_union = intersecting_pas.geometry().dissolve()
                non_protected_geom = buffered_geom.difference(pas_union, ee.ErrorMargin(1))
                return buffered_feature.setGeometry(non_protected_geom)
            
            def keep_original():
                return buffered_feature
            
            # If there are PAs, subtract them; otherwise keep original buffer
            result_feature = ee.Algorithms.If(
                intersecting_pas.size().gt(0),
                create_difference(),
                keep_original()
            )
            
            # Add properties
            return ee.Feature(result_feature).set({
                'protection_status': 'non_protected', 
                'pa_count': intersecting_pas.size(),
                'has_pa_overlap': intersecting_pas.size().gt(0)
            })
        
        non_protected_portions = buffered_features.map(get_non_protected_portion)
        
        non_protected_collections[buffer_dist] = non_protected_portions
        print(f"Non-protected portions created for {buffer_dist/1000}km buffer")
    
    return non_protected_collections

# ============================================================================
# FOREST LOSS CALCULATION 
# ============================================================================

def calculate_forest_metrics(feature_collection, collection_name, buffer_dist):
    """Calculate forest loss metrics for a feature collection"""
    
    print(f"Calculating forest metrics for {collection_name} at {buffer_dist/1000}km buffer...")
    
    # Combine deforestation stack with initial forest
    defor_area_stack = hansen_annual_deforestation.multiply(ee.Image.pixelArea())
    initial_forest_area = initial_forest_2000.multiply(ee.Image.pixelArea())
    combined_stack = defor_area_stack.addBands(initial_forest_area.rename('initial_forest_2000'))
    
    def process_feature(f):
        # Check if feature has valid geometry and data
        initial_forest_raw = f.get('initial_forest_2000')
        
        # Return null if no valid data (this will be dropped by dropNulls=True)
        if_valid_data = ee.Algorithms.If(
            ee.Number(initial_forest_raw).gte(0),
            True,
            False
        )
        
        def calculate_metrics():
            # Calculate initial forest area
            initial_forest_ha = ee.Number(initial_forest_raw).divide(10000)
            
            # Get all the annual deforestation areas
            annual_stats = {}
            total_deforestation = ee.Number(0)
            
            for year in range(analysis_start_year, analysis_end_year + 1):
                band_name = f'hansen_defor_{year}'
                area_m2 = ee.Number(f.get(band_name))
                area_ha = area_m2.divide(10000)
                annual_stats[f'defor_{year}_ha'] = area_ha
                total_deforestation = total_deforestation.add(area_ha)
            
            # Calculate forest loss rate
            forest_loss_rate = ee.Number(
                ee.Algorithms.If(
                    initial_forest_ha.gt(0),
                    total_deforestation.divide(initial_forest_ha).multiply(100),
                    0
                )
            )
            
            # Calculate annual average loss rate  
            years_analyzed = analysis_end_year - analysis_start_year + 1
            annual_avg_loss_rate = forest_loss_rate.divide(years_analyzed)
            
            # Base properties
            base_properties = {
                'BU_ID': f.get('original_BU_ID'),
                'buffer_dist_m': buffer_dist,
                'buffer_km': buffer_dist/1000,
                'protection_status': f.get('protection_status'),
                'pa_count': f.get('pa_count'),
                'has_pa_overlap': f.get('has_pa_overlap'),
                'initial_forest_2000_ha': initial_forest_ha,
                'total_defor_2001_2024_ha': total_deforestation,
                'forest_loss_rate_percent': forest_loss_rate,
                'annual_avg_loss_rate_percent': annual_avg_loss_rate,
                'analysis_period': f'{analysis_start_year}-{analysis_end_year}',
                'forest_threshold_pct': forest_threshold,
                'dataset': 'Hansen_GFC_v1_12'
            }
            
            # Combine with annual stats
            all_properties = {**base_properties, **annual_stats}
            
            return ee.Feature(None, all_properties)
        
        # Return calculated feature or null (to be dropped)
        return ee.Algorithms.If(
            if_valid_data,
            calculate_metrics(),
            None
        )
    
    # Calculate statistics with dropNulls enabled
    stats = combined_stack.reduceRegions(
        collection=feature_collection,
        reducer=ee.Reducer.sum(),
        scale=30,
        crs='EPSG:4326'
    ).map(process_feature, dropNulls=True)  # Enable dropNulls
    
    return stats

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_forest_analysis_simplified(filtered_codf_polygons):
    """
    Simplified approach: Create and export protected vs non-protected portions separately
    """
    
    print("=== Starting Simplified Forest Analysis ===")
    print("Step 1: Creating protected portions...")
    
    # Create protected portions for all buffer distances
    protected_collections = create_protected_portions(filtered_codf_polygons, buffers)
    
    print("\nStep 2: Creating non-protected portions...")
    
    # Create non-protected portions for all buffer distances  
    non_protected_collections = create_non_protected_portions(filtered_codf_polygons, buffers)
    
    print("\nStep 3: Calculating forest metrics and exporting...")
    
    # Export protected portions
    for buffer_dist in buffers:
        if buffer_dist in protected_collections:
            protected_fc = protected_collections[buffer_dist]
            
            # Calculate forest metrics
            protected_stats = calculate_forest_metrics(
                protected_fc, 
                "protected", 
                buffer_dist
            )
            
            # Export
            task_protected = ee.batch.Export.table.toDrive(
                collection=protected_stats,
                description=f'forest_loss_PROTECTED_{buffer_dist//1000}km_2001_2024',
                fileFormat='CSV',
                folder='forestAnalysisSimple'
            )
            task_protected.start()
            print(f"✓ Export started: PROTECTED portions, {buffer_dist/1000}km buffer")
    
    # Export non-protected portions
    for buffer_dist in buffers:
        if buffer_dist in non_protected_collections:
            non_protected_fc = non_protected_collections[buffer_dist]
            
            # Calculate forest metrics
            non_protected_stats = calculate_forest_metrics(
                non_protected_fc,
                "non_protected", 
                buffer_dist
            )
            
            # Export
            task_non_protected = ee.batch.Export.table.toDrive(
                collection=non_protected_stats,
                description=f'forest_loss_NON_PROTECTED_{buffer_dist//1000}km_2001_2024',
                fileFormat='CSV',
                folder='forestAnalysisSimple'
            )
            task_non_protected.start()
            print(f"✓ Export started: NON-PROTECTED portions, {buffer_dist/1000}km buffer")
    
    print(f"\n=== Export Summary ===")
    print(f"Total export tasks: {len(buffers) * 2}")
    print(f"Check 'forestAnalysisSimple' folder in Google Drive")
    print(f"Files will be named:")
    print(f"  - forest_loss_PROTECTED_[X]km_2001_2024.csv")
    print(f"  - forest_loss_NON_PROTECTED_[X]km_2001_2024.csv")
    print(f"\nAfter download, you can combine them locally using pandas!")

# ============================================================================
# DEBUG AND TEST FUNCTIONS
# ============================================================================

def test_single_buffer(filtered_codf_polygons, buffer_dist=5000):
    """Test with a single buffer distance to debug issues"""
    
    print(f"=== Testing with {buffer_dist/1000}km buffer ===")
    
    # Test protected portions
    print("Testing protected portions...")
    try:
        protected_fc = create_protected_portions(filtered_codf_polygons, [buffer_dist])[buffer_dist]
        protected_count = protected_fc.size().getInfo()
        print(f"✓ Protected features created: {protected_count}")
        
        if protected_count > 0:
            # Test metrics calculation
            protected_stats = calculate_forest_metrics(protected_fc, "protected", buffer_dist)
            stats_count = protected_stats.size().getInfo()
            print(f"✓ Protected stats calculated: {stats_count}")
            
            # Test export
            task = ee.batch.Export.table.toDrive(
                collection=protected_stats,
                description=f'TEST_forest_loss_PROTECTED_{buffer_dist//1000}km',
                fileFormat='CSV',
                folder='forestAnalysisTest'
            )
            task.start()
            print(f"✓ Test export started for protected portions")
        else:
            print("⚠ No protected features found")
            
    except Exception as e:
        print(f"✗ Error with protected portions: {e}")
    
    # Test non-protected portions
    print("\nTesting non-protected portions...")
    try:
        non_protected_fc = create_non_protected_portions(filtered_codf_polygons, [buffer_dist])[buffer_dist]
        non_protected_count = non_protected_fc.size().getInfo()
        print(f"✓ Non-protected features created: {non_protected_count}")
        
        if non_protected_count > 0:
            # Test metrics calculation
            non_protected_stats = calculate_forest_metrics(non_protected_fc, "non_protected", buffer_dist)
            stats_count = non_protected_stats.size().getInfo()
            print(f"✓ Non-protected stats calculated: {stats_count}")
            
            # Test export
            task = ee.batch.Export.table.toDrive(
                collection=non_protected_stats,
                description=f'TEST_forest_loss_NON_PROTECTED_{buffer_dist//1000}km',
                fileFormat='CSV',
                folder='forestAnalysisTest'
            )
            task.start()
            print(f"✓ Test export started for non-protected portions")
        else:
            print("⚠ No non-protected features found")
            
    except Exception as e:
        print(f"✗ Error with non-protected portions: {e}")

# ============================================================================
# EXECUTION
# ============================================================================

print("=== Simplified Forest Analysis Script Loaded ===")
print("To run analysis, call:")
print("export_forest_analysis_simplified(your_filtered_CODF_polygons)")
print("\nThis approach:")
print("1. Creates separate protected/non-protected feature collections")  
print("2. Calculates forest loss metrics for each separately")
print("3. Exports as separate CSV files") 
print("4. Avoids complex geometry operations and error handling")
print("5. Allows easy local combination of results")

# Example of how to combine results locally (after download):
print("\n=== Local Combination Example ===")
print("""
# After downloading CSV files, combine them locally:
import pandas as pd
import glob

# Read all protected files
protected_files = glob.glob('forest_loss_PROTECTED_*km_2001_2024.csv')
protected_dfs = [pd.read_csv(f) for f in protected_files]
protected_combined = pd.concat(protected_dfs, ignore_index=True)

# Read all non-protected files  
non_protected_files = glob.glob('forest_loss_NON_PROTECTED_*km_2001_2024.csv')
non_protected_dfs = [pd.read_csv(f) for f in non_protected_files] 
non_protected_combined = pd.concat(non_protected_dfs, ignore_index=True)

# Combine everything
all_results = pd.concat([protected_combined, non_protected_combined], ignore_index=True)
all_results.to_csv('combined_forest_analysis_results.csv', index=False)
""")
# ============================================================================
# EXECUTION
# ============================================================================
# read the saved file
availability_threshold = 90  # Set your threshold here
output_file = f'/usr2/postdoc/chishan/project_data/CODF/tmf_suitable_sites_90percent.csv'

good_sites = pd.read_csv(output_file)

# Load all CODF datasets - points, lines, and polygons
HANSEN = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
CODF_polygons = ee.FeatureCollection("projects/ee-zcs/assets/CODF_polygons")
CODF_points = ee.FeatureCollection("projects/ee-zcs/assets/CODF_points")
CODF_lines = ee.FeatureCollection("projects/ee-zcs/assets/CODF_lines")

#combine all CODF datasets into a single FeatureCollection
CODF_combined = CODF_polygons.merge(CODF_lines)
print("Combined CODF FeatureCollection:", CODF_combined.size().getInfo())

#filter good sites based on the availability threshold
filtered_CODF_polygons = CODF_combined.filter(ee.Filter.inList('BU ID', good_sites['BU_ID'].tolist()))

# 运行分析 - 请确保您的 filtered_CODF_polygons 变量已定义
# export_forest_analysis_with_protection_status(filtered_CODF_polygons)
export_forest_analysis_simplified(filtered_CODF_polygons)

print("Forest analysis script loaded successfully!")
print("To run the analysis, call: export_forest_analysis_with_protection_status(your_filtered_CODF_polygons)")
print("\nThis script will:")
print("1. Split each CODF feature buffer into protected and non-protected portions")
print("2. Calculate annual forest loss (2001-2024) for each portion separately") 
print("3. Calculate forest loss rates and annual averages")
print("4. Export results to CSV files in 'hansenProtectionAnalysis' folder")
print("5. Include both separate files by protection status and a combined file")