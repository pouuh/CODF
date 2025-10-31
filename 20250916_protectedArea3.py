"""
Hansen Global Forest Change Analysis with Protected Areas Mask - Rasterized Approach

Purpose: Use rasterized WCMC protected areas mask to analyze forest loss,
avoiding complex geometry operations that cause computation errors.

Approach:
1. Rasterize WCMC protected areas into binary mask (1=protected, 0=non-protected)
2. Create protected forest images: hansen_data * protected_mask  
3. Create non-protected forest images: hansen_data * (1 - protected_mask)
4. Use simple buffered geometries for reduceRegions on masked images
5. Export separate CSV files for protected vs non-protected statistics

Benefits:
- No complex geometry boolean operations
- Scalable with tileScale parameter
- More reliable computation
- Consistent area calculations
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
# PROTECTED AREAS RASTERIZATION
# ============================================================================

def create_protected_areas_mask(scale=30):
    """
    Rasterize WCMC protected areas into binary mask
    
    Args:
        scale: Pixel size in meters (default 30m to match Hansen)
    
    Returns:
        Binary image where 1=protected, 0=non-protected
    """
    print(f"Creating protected areas mask at {scale}m resolution...")
    
    # Create a binary image from protected areas
    # Method 1: Paint the protected areas with value 1
    protected_mask = ee.Image(0).byte()  # Start with all zeros
    protected_mask = protected_mask.paint(wcmc_protected_areas, 1)  # Paint PAs as 1
    
    # Alternative method if needed:
    # protected_mask = wcmc_protected_areas.reduceToImage(['WDPAID'], ee.Reducer.first()).gt(0)
    
    return protected_mask.rename('protected_mask')

# Create the protected areas mask
protected_areas_mask = create_protected_areas_mask(scale=30)

# ============================================================================
# MASKED FOREST DATA CREATION
# ============================================================================

def create_masked_forest_data():
    """
    Create protected and non-protected versions of all forest datasets
    
    Returns:
        Dictionary containing masked forest data
    """
    print("Creating masked forest datasets...")
    
    # Create masks
    protected_mask = protected_areas_mask.eq(1)  # 1 where protected
    non_protected_mask = protected_areas_mask.eq(0)  # 1 where not protected
    
    # Mask initial forest area
    initial_forest_protected = initial_forest_2000.updateMask(protected_mask)
    initial_forest_non_protected = initial_forest_2000.updateMask(non_protected_mask)
    
    # Mask annual deforestation data
    hansen_defor_protected = hansen_annual_deforestation.updateMask(protected_mask)
    hansen_defor_non_protected = hansen_annual_deforestation.updateMask(non_protected_mask)
    
    # Convert to area images (multiply by pixel area)
    pixel_area = ee.Image.pixelArea()
    
    initial_forest_protected_area = initial_forest_protected.multiply(pixel_area)
    initial_forest_non_protected_area = initial_forest_non_protected.multiply(pixel_area)
    
    hansen_defor_protected_area = hansen_defor_protected.multiply(pixel_area)
    hansen_defor_non_protected_area = hansen_defor_non_protected.multiply(pixel_area)
    
    return {
        'protected': {
            'initial_forest': initial_forest_protected_area.rename('initial_forest_2000'),
            'deforestation': hansen_defor_protected_area
        },
        'non_protected': {
            'initial_forest': initial_forest_non_protected_area.rename('initial_forest_2000'),
            'deforestation': hansen_defor_non_protected_area
        }
    }

# Create masked datasets
masked_forest_data = create_masked_forest_data()

# ============================================================================
# SIMPLE BUFFERED GEOMETRIES (NO BOOLEAN OPERATIONS)
# ============================================================================

def create_simple_buffers(filtered_codf_polygons, buffer_distances):
    """
    Create simple buffered geometries without any boolean operations
    
    Args:
        filtered_codf_polygons: Input CODF features
        buffer_distances: List of buffer distances in meters
        
    Returns:
        Dictionary of buffered feature collections
    """
    
    buffered_collections = {}
    
    for buffer_dist in buffer_distances:
        print(f"Creating simple {buffer_dist/1000}km buffers...")
        
        def create_buffer(feature):
            return (feature.buffer(buffer_dist)
                   .set('buffer_dist_m', buffer_dist)
                   .set('buffer_km', buffer_dist/1000)
                   .set('original_BU_ID', feature.get('BU_ID'))  # Try 'BU_ID' first
                   .set('original_BU_ID_alt', feature.get('BU ID'))  # Also try 'BU ID' (with space)
                   .copyProperties(feature, ['BU_ID', 'BU ID']))  # Copy both possible field names
        
        buffered_fc = filtered_codf_polygons.map(create_buffer)
        buffered_collections[buffer_dist] = buffered_fc
        
    return buffered_collections

# ============================================================================
# FOREST STATISTICS CALCULATION WITH MASKS
# ============================================================================

def calculate_masked_forest_statistics(buffered_fc, protection_status, buffer_dist, tileScale=2):
    """
    Calculate forest statistics using masked imagery and simple geometries
    
    Args:
        buffered_fc: Buffered feature collection
        protection_status: 'protected' or 'non_protected'
        buffer_dist: Buffer distance in meters
        tileScale: Scale factor for computation (1=default, 2=half resolution, 4=quarter, etc.)
        
    Returns:
        Feature collection with statistics
    """
    
    print(f"Calculating {protection_status} forest statistics for {buffer_dist/1000}km buffer (tileScale={tileScale})...")
    
    # Get the appropriate masked datasets
    forest_data = masked_forest_data[protection_status]
    initial_forest_img = forest_data['initial_forest']
    deforestation_img = forest_data['deforestation']
    
    # Combine all images for single reduceRegions call
    combined_img = deforestation_img.addBands(initial_forest_img)
    
    def process_feature(f):
        # Handle potential null/undefined values
        def safe_get_number(property_name, default_value=0):
            return ee.Number(ee.Algorithms.If(
                ee.Algorithms.IsEqual(f.get(property_name), None),
                default_value,
                f.get(property_name)
            ))
        
        # Calculate initial forest area
        initial_forest_m2 = safe_get_number('initial_forest_2000')
        initial_forest_ha = initial_forest_m2.divide(10000)
        
        # Calculate annual deforestation areas and total
        annual_stats = {}
        total_deforestation = ee.Number(0)
        
        for year in range(analysis_start_year, analysis_end_year + 1):
            band_name = f'hansen_defor_{year}'
            area_m2 = safe_get_number(band_name)
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
        
        # Create properties dictionary - handle multiple possible BU_ID field names
        bu_id_value = ee.Algorithms.If(
            ee.Algorithms.IsEqual(f.get('original_BU_ID'), None),
            f.get('original_BU_ID_alt'),  # Try alternative field name
            f.get('original_BU_ID')
        )
        
        # Also try to get from copied properties
        bu_id_final = ee.Algorithms.If(
            ee.Algorithms.IsEqual(bu_id_value, None),
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(f.get('BU_ID'), None),
                f.get('BU ID'),  # Try 'BU ID' with space
                f.get('BU_ID')
            ),
            bu_id_value
        )
        
        base_properties = {
            'BU_ID': bu_id_final,
            'buffer_dist_m': buffer_dist,
            'buffer_km': buffer_dist/1000,
            'protection_status': protection_status,
            'initial_forest_2000_ha': initial_forest_ha,
            'total_defor_2001_2024_ha': total_deforestation,
            'forest_loss_rate_percent': forest_loss_rate,
            'annual_avg_loss_rate_percent': annual_avg_loss_rate,
            'analysis_period': f'{analysis_start_year}-{analysis_end_year}',
            'forest_threshold_pct': forest_threshold,
            'dataset': 'Hansen_GFC_v1_12_masked',
            'mask_method': 'rasterized_WCMC',
            'tileScale': tileScale
        }
        
        # Combine base properties with annual stats
        all_properties = {**base_properties, **annual_stats}
        
        return ee.Feature(None, all_properties)
    
    # Perform reduceRegions with tileScale parameter for memory management
    try:
        stats_fc = combined_img.reduceRegions(
            collection=buffered_fc,
            reducer=ee.Reducer.sum(),
            scale=30,
            crs='EPSG:4326',
            tileScale=tileScale
        )
        
        # Process the statistics
        processed_stats = stats_fc.map(process_feature)
        
        return processed_stats
        
    except Exception as e:
        print(f"Error in reduceRegions, trying with higher tileScale...")
        if tileScale < 16:  # Retry with higher tileScale
            return calculate_masked_forest_statistics(buffered_fc, protection_status, buffer_dist, tileScale*2)
        else:
            raise e

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_masked_forest_analysis(filtered_codf_polygons, tileScale=2):
    """
    Main function to export forest analysis using rasterized masks
    
    Args:
        filtered_codf_polygons: Your filtered CODF polygon collection
        tileScale: Computation scale factor (2=recommended, higher for memory issues)
    """
    
    print("=== Starting Masked Forest Analysis ===")
    print(f"Using tileScale: {tileScale}")
    
    # Step 1: Create simple buffered geometries
    print("Step 1: Creating simple buffered geometries...")
    buffered_collections = create_simple_buffers(filtered_codf_polygons, buffers)
    
    # Step 2: Calculate and export statistics
    print("Step 2: Calculating forest statistics and exporting...")
    
    export_tasks = []
    
    for buffer_dist in buffers:
        buffered_fc = buffered_collections[buffer_dist]
        
        # Export PROTECTED statistics
        print(f"\nProcessing PROTECTED areas for {buffer_dist/1000}km buffer...")
        try:
            protected_stats = calculate_masked_forest_statistics(
                buffered_fc, 
                'protected', 
                buffer_dist, 
                tileScale
            )
            
            task_protected = ee.batch.Export.table.toDrive(
                collection=protected_stats,
                description=f'forest_PROTECTED_mask_{buffer_dist//1000}km_2001_2024',
                fileFormat='CSV',
                folder='forestAnalysisMasked'
            )
            task_protected.start()
            export_tasks.append(f"PROTECTED {buffer_dist/1000}km")
            print(f"✓ Export started: PROTECTED areas, {buffer_dist/1000}km buffer")
            
        except Exception as e:
            print(f"✗ Error with PROTECTED areas {buffer_dist/1000}km: {e}")
        
        # Export NON-PROTECTED statistics  
        print(f"Processing NON-PROTECTED areas for {buffer_dist/1000}km buffer...")
        try:
            non_protected_stats = calculate_masked_forest_statistics(
                buffered_fc,
                'non_protected',
                buffer_dist,
                tileScale
            )
            
            task_non_protected = ee.batch.Export.table.toDrive(
                collection=non_protected_stats,
                description=f'forest_NON_PROTECTED_mask_{buffer_dist//1000}km_2001_2024',
                fileFormat='CSV',
                folder='forestAnalysisMasked'
            )
            task_non_protected.start()
            export_tasks.append(f"NON-PROTECTED {buffer_dist/1000}km")
            print(f"✓ Export started: NON-PROTECTED areas, {buffer_dist/1000}km buffer")
            
        except Exception as e:
            print(f"✗ Error with NON-PROTECTED areas {buffer_dist/1000}km: {e}")
    
    print(f"\n=== Export Summary ===")
    print(f"Export tasks started: {len(export_tasks)}")
    print("Check 'forestAnalysisMasked' folder in Google Drive")
    print("Files format:")
    print("  - forest_PROTECTED_mask_[X]km_2001_2024.csv")
    print("  - forest_NON_PROTECTED_mask_[X]km_2001_2024.csv")
    
    if len(export_tasks) < len(buffers) * 2:
        print(f"\n⚠ Warning: Some exports failed. Consider increasing tileScale parameter.")
        print(f"Current tileScale: {tileScale}")
        print(f"Try: export_masked_forest_analysis(your_data, tileScale=4)")

# ============================================================================
# TESTING AND DEBUG FUNCTIONS
# ============================================================================

def debug_field_names(filtered_codf_polygons):
    """Debug function to check field names in the original data"""
    
    print("=== Debugging Original Data Field Names ===")
    
    try:
        # Get first feature to check field names
        first_feature = ee.Feature(filtered_codf_polygons.first())
        properties = first_feature.propertyNames().getInfo()
        
        print("Available field names:")
        for prop in properties:
            print(f"  - '{prop}'")
        
        # Check specific BU_ID related fields
        first_feature_info = first_feature.getInfo()
        print(f"\nFirst feature properties:")
        
        for prop in properties:
            if 'BU' in prop.upper() or 'ID' in prop.upper():
                value = first_feature_info['properties'].get(prop)
                print(f"  {prop}: {value}")
        
        return properties
        
    except Exception as e:
        print(f"Error getting field names: {e}")
        return None

def test_bu_id_transfer(filtered_codf_polygons, buffer_dist=5000):
    """Test BU_ID transfer through buffer creation"""
    
    print("=== Testing BU_ID Transfer ===")
    
    # First check original field names
    original_fields = debug_field_names(filtered_codf_polygons)
    
    # Get first feature
    first_original = ee.Feature(filtered_codf_polygons.first())
    
    # Create buffer
    def create_test_buffer(feature):
        return (feature.buffer(buffer_dist)
               .set('buffer_dist_m', buffer_dist)
               .set('buffer_km', buffer_dist/1000)
               .set('original_BU_ID', feature.get('BU_ID'))
               .set('original_BU_ID_alt', feature.get('BU ID'))
               .set('debug_all_props', feature.toDictionary())  # Copy all properties for debugging
               .copyProperties(feature))
    
    buffered_first = create_test_buffer(first_original)
    buffered_info = buffered_first.getInfo()
    
    print(f"\nBuffered feature properties:")
    props = buffered_info['properties']
    for key in sorted(props.keys()):
        if 'BU' in key.upper() or 'ID' in key.upper() or 'original' in key.lower():
            print(f"  {key}: {props[key]}")
    
    return buffered_first

def test_protected_mask(geometry_bounds=None):
    """Test the protected areas mask creation"""
    
    print("=== Testing Protected Areas Mask ===")
    
    try:
        # Test mask creation
        mask = create_protected_areas_mask()
        print("✓ Protected areas mask created successfully")
        
        # Test mask statistics in a small area
        if geometry_bounds:
            test_area = ee.Geometry.Rectangle(geometry_bounds)
        else:
            # Use a small test area (adjust coordinates as needed)
            test_area = ee.Geometry.Rectangle([-10, -10, 10, 10])
        
        mask_stats = mask.reduceRegion(
            reducer=ee.Reducer.sum().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=test_area,
            scale=30,
            maxPixels=1e9
        )
        
        print(f"Test area mask statistics: {mask_stats.getInfo()}")
        
        return mask
        
    except Exception as e:
        print(f"✗ Error testing mask: {e}")
        return None

def test_single_site_masked(filtered_codf_polygons, site_index=0, buffer_dist=5000, tileScale=2):
    """Test masked analysis on a single site"""
    
    print(f"=== Testing Single Site (index {site_index}) ===")
    
    # Get single site
    site_list = filtered_codf_polygons.toList(site_index+1)
    single_site = ee.FeatureCollection([site_list.get(site_index)])
    
    # Create buffer
    buffered_site = create_simple_buffers(single_site, [buffer_dist])[buffer_dist]
    
    print(f"Testing {buffer_dist/1000}km buffer on single site...")
    
    # Test protected analysis
    try:
        protected_stats = calculate_masked_forest_statistics(
            buffered_site, 'protected', buffer_dist, tileScale
        )
        protected_result = protected_stats.first().getInfo()
        print(f"✓ Protected analysis successful")
        print(f"  Initial forest: {protected_result['properties'].get('initial_forest_2000_ha', 'N/A')} ha")
        
    except Exception as e:
        print(f"✗ Protected analysis failed: {e}")
    
    # Test non-protected analysis
    try:
        non_protected_stats = calculate_masked_forest_statistics(
            buffered_site, 'non_protected', buffer_dist, tileScale
        )
        non_protected_result = non_protected_stats.first().getInfo()
        print(f"✓ Non-protected analysis successful")
        print(f"  Initial forest: {non_protected_result['properties'].get('initial_forest_2000_ha', 'N/A')} ha")
        
    except Exception as e:
        print(f"✗ Non-protected analysis failed: {e}")

# ============================================================================
# EXECUTION
# ============================================================================

print("=== Masked Forest Analysis Script Loaded ===")
print("This approach uses rasterized WCMC masks to avoid geometry operations!")
print("\nRecommended debugging steps:")
print("1. Check field names: debug_field_names(your_data)")
print("2. Test BU_ID transfer: test_bu_id_transfer(your_data)")
print("3. Test mask: test_protected_mask()")
print("4. Test single site: test_single_site_masked(your_data)")
print("5. Run full analysis: export_masked_forest_analysis(your_data)")
print("\nTileScale parameters:")
print("- tileScale=1: Full resolution (may cause memory errors)")
print("- tileScale=2: Half resolution (recommended)")
print("- tileScale=4: Quarter resolution (for large areas)")
print("- tileScale=8+: Lower resolution (for memory issues)")

# Example local combination code
print("""\n=== Local Data Combination (after download) ===
import pandas as pd
import glob

# Read protected files
protected_files = glob.glob('forest_PROTECTED_mask_*km_2001_2024.csv')
protected_df = pd.concat([pd.read_csv(f) for f in protected_files], ignore_index=True)

# Read non-protected files  
non_protected_files = glob.glob('forest_NON_PROTECTED_mask_*km_2001_2024.csv')
non_protected_df = pd.concat([pd.read_csv(f) for f in non_protected_files], ignore_index=True)

# Combine all results
combined_df = pd.concat([protected_df, non_protected_df], ignore_index=True)
combined_df.to_csv('masked_forest_analysis_results.csv', index=False)
""")

# CODF 数据 (你需要替换为自己的资产路径)
CODF_polygons = ee.FeatureCollection("projects/ee-zcs/assets/CODF_polygons")
CODF_points = ee.FeatureCollection("projects/ee-zcs/assets/CODF_points")
CODF_lines = ee.FeatureCollection("projects/ee-zcs/assets/CODF_lines")
CODF_combined = CODF_polygons.merge(CODF_lines)

#combine all CODF datasets into a single FeatureCollection
CODF_combined = CODF_polygons.merge(CODF_lines)
print("Combined CODF FeatureCollection:", CODF_combined.size().getInfo())
#filter good sites based on the availability threshold
availability_threshold = 90  # Set your threshold here
output_file = f'/usr2/postdoc/chishan/project_data/CODF/tmf_suitable_sites_90percent.csv'
good_sites = pd.read_csv(output_file)
filtered_CODF_polygons = CODF_combined.filter(ee.Filter.inList('BU ID', good_sites['BU_ID'].tolist()))
export_masked_forest_analysis(filtered_CODF_polygons, tileScale=2)
