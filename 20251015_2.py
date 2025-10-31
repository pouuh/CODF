import xml.etree.ElementTree as ET
import xarray as xr
import numpy as np

def parse_qml_legend(qml_path):
    """
    Parse QGIS QML file to extract class values and labels
    
    Parameters:
    -----------
    qml_path : str
        Path to .qml file
    
    Returns:
    --------
    dict : Dictionary mapping pixel values to class names
    """
    tree = ET.parse(qml_path)
    root = tree.getroot()
    
    classes = {}
    
    # QML files typically have paletteEntry elements with value and label
    for entry in root.findall('.//paletteEntry'):
        value = entry.get('value')
        label = entry.get('label')
        if value and label:
            classes[int(value)] = label
    
    # Alternative: Look for item elements in categorizedrenderer
    if not classes:
        for item in root.findall('.//category'):
            value = item.get('value')
            label = item.get('label')
            if value and label:
                classes[int(value)] = label
    
    print(f"Parsed {len(classes)} classes from QML file:")
    for val, label in sorted(classes.items()):
        print(f"  {val}: {label}")
    
    return classes

# Example usage - update path to your QML file
folder_terra = '/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/'
REF_TIF = folder_terra + 'AMZ.2016.M.tif'

qml_file = folder_terra + 'AMZ.2016.M.qml'
classes = parse_qml_legend(qml_file)
print("QML parser ready. Update qml_file path to use.")

# Create MapBiomas-specific forest reclassification
def create_mapbiomas_forest_map(classes_dict):
    """
    Create forest/non-forest reclassification for MapBiomas Amazon classes
    
    MapBiomas Forest Classes:
    - 1: VEGETACAO NATURAL FLORESTAL PRIMARIA (Primary Forest)
    - 2: VEGETACAO NATURAL FLORESTAL SECUNDARIA (Secondary Forest)
    
    MapBiomas Non-Forest Classes:
    - 9: SILVICULTURA (Forestry/Plantation)
    - 10: PASTAGEM ARBUSTIVA/ARBOREA (Shrubby/Tree Pasture)
    - 11: PASTAGEM HERBACEA (Herbaceous Pasture)
    - 12: CULTURA AGRICOLA PERENE (Perennial Agriculture)
    - 13: CULTURA AGRICOLA SEMIPERENE (Semi-perennial Agriculture)
    - 14: CULTURA AGRICOLA TEMPORARIA DE 1 CICLO (1-cycle Agriculture)
    - 15: CULTURA AGRICOLA TEMPORARIA DE MAIS DE 1 CICLO (Multi-cycle Agriculture)
    - 16: MINERACAO (Mining)
    - 17: URBANIZADA (Urban)
    - 20: OUTROS USOS (Other Uses)
    - 22: DESFLORESTAMENTO NO ANO (Deforestation in year)
    - 23: CORPO DAGUA (Water Body)
    - 25: NAO OBSERVADO (Not Observed)
    - 51: NATURAL NAO FLORESTAL (Natural Non-Forest)
    
    Parameters:
    -----------
    classes_dict : dict
        Dictionary from parse_qml_legend()
    
    Returns:
    --------
    dict : Reclassification map (value -> 1=Forest, 0=Non-Forest)
    """
    # Define forest classes explicitly for MapBiomas
    # FOREST_CLASSES = [1, 2]  # Primary and Secondary Natural Forest only
    
    # Optional: Include plantation forest (silvicultura)
    FOREST_CLASSES = [1, 2, 9]  # Uncomment to include plantations
    
    reclass_map = {}
    for value in classes_dict.keys():
        reclass_map[value] = 1 if value in FOREST_CLASSES else 0
    
    # Print detailed classification
    print("\n" + "=" * 80)
    print("MAPBIOMAS AMAZON FOREST RECLASSIFICATION")
    print("=" * 80)
    
    print("\n🌳 FOREST Classes (mapped to 1):")
    for value, label in sorted(classes_dict.items()):
        if reclass_map[value] == 1:
            print(f"  {value:3d}: {label}")
    
    print("\n🏞️  NON-FOREST Classes (mapped to 0):")
    for value, label in sorted(classes_dict.items()):
        if reclass_map[value] == 0:
            print(f"  {value:3d}: {label}")
    
    print("\n" + "=" * 80)
    print(f"Total: {sum(1 for v in reclass_map.values() if v == 1)} Forest classes, "
          f"{sum(1 for v in reclass_map.values() if v == 0)} Non-forest classes")
    print("=" * 80)
    
    return reclass_map

# # Show the mapping
# print("\nReclassification mapping ready!")
# print("Note: Only classes 1 (Primary) and 2 (Secondary) Natural Forest are classified as Forest.")
# print("      Silvicultura (9, plantation) is treated as Non-Forest by default.")
# print("      Uncomment line in function to include plantations as forest if needed.")


def find_glance_tiles_by_year(base_folder, year, region='SA'):
    """
    Find all GLANCE tiles for a specific year and region
    
    Parameters:
    -----------
    base_folder : str
        Base directory containing GLANCE data (e.g., '/projectnb/measures/products/SA/v001/DAAC/LC/')
    year : int
        Target year (e.g., 2018)
    region : str
        Region code (default: 'SA' for South America)
    
    Returns:
    --------
    list : Sorted list of file paths matching the year and region
    """
    # GLANCE naming: GLANCE.A{year}{month}{day}.h{hh}v{vv}.001.{processing_date}.{region}.LC.tif
    # For annual products, typically July 1st (0701)
    pattern = os.path.join(base_folder, f'GLANCE.A{year}0701.h*v*.001.*.{region}.LC.tif')
    files = sorted(glob.glob(pattern))
    
    print(f"Found {len(files)} tiles for year {year}, region {region}")
    if files:
        print(f"Example: {os.path.basename(files[0])}")
    
    return files


def load_glance_tiles_lazy(file_list, chunks='auto'):
    """
    Load multiple GLANCE tiles as a lazy dask array mosaic
    
    Parameters:
    -----------
    file_list : list
        List of file paths to load
    chunks : str or dict
        Chunk size for dask array (default: 'auto')
    
    Returns:
    --------
    xarray.DataArray : Merged mosaic with dask backend
    """
    if not file_list:
        raise ValueError("Empty file list provided")
    
    # Open all tiles lazily with rioxarray
    datasets = []
    for fpath in file_list:
        try:
            ds = rioxarray.open_rasterio(fpath, chunks=chunks, lock=False)
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: Failed to open {os.path.basename(fpath)}: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets loaded")
    
    print(f"Loaded {len(datasets)} tiles lazily")
    
    # Merge tiles into a single mosaic
    # Method 1: Simple concatenation if tiles are aligned
    # Method 2: merge_arrays for overlapping tiles
    try:
        from rioxarray.merge import merge_arrays
        mosaic = merge_arrays(datasets, nodata=0)
        print(f"Merged mosaic shape: {mosaic.shape}")
        print(f"Merged mosaic dtype: {mosaic.dtype}")
        print(f"Chunk size: {mosaic.chunks}")
        return mosaic
    except Exception as e:
        print(f"Merge failed: {e}")
        print("Falling back to simple stack approach")
        # If merge fails, return first dataset as example
        return datasets[0]

def reclassify_glance_to_forest(glance_array, glance_forest_classes=None):
    """
    Reclassify GLANCE land cover to binary Forest/Non-Forest
    
    Parameters:
    -----------
    glance_array : xarray.DataArray
        GLANCE classification array
    glance_forest_classes : list, optional
        List of GLANCE class values considered as forest (default: 1-5)
    
    Returns:
    --------
    xarray.DataArray : Binary forest classification (1=Forest, 0=Non-Forest)
    """

    
    # Default forest classes: 5 (all forest types)
    if glance_forest_classes is None:
        glance_forest_classes = [5]  # All forest types
    
    print("GLANCE Forest Classification:")
    print(f"  Forest classes: {glance_forest_classes}")
    
    # Squeeze if needed
    arr = glance_array.squeeze() if len(glance_array.shape) > 2 else glance_array
    
    # Create binary mask
    forest_mask = np.isin(arr.values, glance_forest_classes)
    binary_output = forest_mask.astype(np.uint8)
    
    # Count pixels
    n_forest = forest_mask.sum()
    n_total = arr.size
    n_nonforest = (~forest_mask & (arr.values != 0)).sum()
    
    print(f"  Forest pixels: {n_forest:,} ({n_forest/n_total*100:.2f}%)")
    print(f"  Non-forest pixels: {n_nonforest:,}")
    
    # Create output xarray
    result = xr.DataArray(
        binary_output,
        coords=arr.coords,
        dims=arr.dims,
        attrs={
            **arr.attrs,
            'long_name': 'GLANCE Forest classification (1=Forest, 0=Non-Forest)',
            'forest_classes': str(glance_forest_classes)
        }
    )
    
    if hasattr(arr, 'rio'):
        result.rio.write_crs(arr.rio.crs, inplace=True)
        result.rio.write_nodata(0, inplace=True)
    
    return result

def load_and_reclassify_terraclass(terraclass_path, qml_path=None, reclass_map=None, chunks='auto'):
    """
    Complete workflow: Load TerraClass map and reclassify to Forest/Non-Forest
    
    Parameters:
    -----------
    terraclass_path : str
        Path to TerraClass Amazon GeoTIFF
    qml_path : str, optional
        Path to QML legend file (auto-search if None)
    reclass_map : dict, optional
        Custom reclassification map (auto-create if None)
    chunks : str or dict
        Chunk size for lazy loading
    
    Returns:
    --------
    tuple : (binary_forest_map, original_classes_dict, reclass_map)
    """
    print(f"Loading TerraClass map: {os.path.basename(terraclass_path)}")
    
    # Load TerraClass map (lazy)
    terraclass = rioxarray.open_rasterio(terraclass_path, chunks=chunks, lock=False)
    print(f"  Shape: {terraclass.shape}")
    print(f"  CRS: {terraclass.rio.crs}")
    print(f"  Dtype: {terraclass.dtype}")
    
    # Parse QML legend if provided
    if qml_path is None:
        # Try to find QML file in same directory
        qml_search = terraclass_path.replace('.tif', '.qml')
        if os.path.exists(qml_search):
            qml_path = qml_search
            print(f"  Found QML file: {os.path.basename(qml_path)}")
    
    classes_dict = {}
    if qml_path and os.path.exists(qml_path):
        classes_dict = parse_qml_legend(qml_path)
    else:
        print("  Warning: No QML file found, using default TerraClass mapping")
        # Default TerraClass Amazon classes (update based on your data)
        classes_dict = {
            1: 'Floresta',
            2: 'Desflorestamento',
            3: 'Agricultura',
            4: 'Pasto',
            5: 'Area Urbana',
            6: 'Outros'
        }
    
    # Create reclassification map if not provided
    if reclass_map is None:
        reclass_map = create_forest_reclassification_map(classes_dict)
    
    # Apply reclassification (lazy operation)
    print("\nApplying reclassification...")
    binary_forest = reclassify_terraclass_to_forest(terraclass, reclass_map)

        # Optional: Include plantation forest (silvicultura)
    FOREST_CLASSES = [1, 2, 9]  # Uncomment to include plantations
    
    reclass_map = {}
    for value in classes_dict.keys():
        reclass_map[value] = 1 if value in FOREST_CLASSES else 0
    
    # Print detailed classification
    print("\n" + "=" * 80)
    print("MAPBIOMAS AMAZON FOREST RECLASSIFICATION")
    print("=" * 80)
    
    print("\n🌳 FOREST Classes (mapped to 1):")
    for value, label in sorted(classes_dict.items()):
        if reclass_map[value] == 1:
            print(f"  {value:3d}: {label}")
    
    print("\n🏞️  NON-FOREST Classes (mapped to 0):")
    for value, label in sorted(classes_dict.items()):
        if reclass_map[value] == 0:
            print(f"  {value:3d}: {label}")
    
    print("\n" + "=" * 80)
    print(f"Total: {sum(1 for v in reclass_map.values() if v == 1)} Forest classes, "
          f"{sum(1 for v in reclass_map.values() if v == 0)} Non-forest classes")
    print("=" * 80)
    
    return reclass_map
    
# Load and reclassify MapBiomas to binary forest map
import rioxarray
print("Loading MapBiomas data...")
mapbiomas_ref = rioxarray.open_rasterio(REF_TIF, chunks={'band': 1, 'x': 2048, 'y': 2048}, lock=False)

print(f"  Shape: {mapbiomas_ref.shape}")
print(f"  CRS: {mapbiomas_ref.rio.crs}")
print(f"  Bounds: {mapbiomas_ref.rio.bounds()}")
print(f"  Data type: {mapbiomas_ref.dtype}")

# Check unique values in the data
print("\nChecking data values (this will sample the data)...")
sample = mapbiomas_ref.isel(x=slice(0, 1000), y=slice(0, 1000)).values.ravel()
unique_vals = np.unique(sample[sample != 0])
print(f"Unique values in sample: {unique_vals}")

# Apply reclassification
print("\nApplying forest/non-forest reclassification...")
mapbiomas_binary = reclassify_terraclass_to_forest(mapbiomas_ref, reclass_map, nodata_value=0)

print("\n✓ MapBiomas reclassified to binary forest map!")
print("  Values: 0 = Non-forest, 1 = Forest")



# ========================================
# COMPLETE WORKFLOW: GLANCE vs TerraClass Forest Comparison
# ========================================

# Step 1: Set up paths
folder_glance = '/projectnb/measures/products/SA/v001/DAAC/LC/'
# terraclass_path = '/path/to/terraclass_amazon_2016.tif'  # TODO: Update!
# qml_path = '/path/to/terraclass_legend.qml'  # TODO: Update! (or set to None for auto-search)

year_target = 2016
folder_terra = '/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/'
terraclass_path = folder_terra + 'AMZ.' + str(year_target) + '.M.tif'
qml_path = folder_terra + 'AMZ.' + str(year_target) + '.M.qml'

print("=" * 70)
print("FOREST CLASSIFICATION COMPARISON: GLANCE vs TerraClass Amazon")
print("=" * 70)

# Step 2: Find and load GLANCE tiles
print("\n[1/6] Finding GLANCE tiles...")
tiles_2018 = find_glance_tiles_by_year(folder_glance, year_target, 'SA')

print("\n[2/6] Loading GLANCE mosaic (lazy)...")
glance_2018 = load_glance_tiles_lazy(tiles_2018, chunks={'band': 1, 'x': 2048, 'y': 2048})

# Step 3: Reclassify GLANCE to Forest/Non-Forest
print("\n[3/6] Reclassifying GLANCE to Forest/Non-Forest...")
glance_forest = reclassify_glance_to_forest(glance_2018)

# Step 4: Load and reclassify TerraClass
print("\n[4/6] Loading and reclassifying TerraClass to Forest/Non-Forest...")
terraclass_forest, tc_classes, tc_reclass = load_and_reclassify_terraclass(
    terraclass_path, 
    qml_path=qml_path,
    chunks={'band': 1, 'x': 2048, 'y': 2048}
)

# Apply to your parsed classes
# reclass_map = create_mapbiomas_forest_map(classes)

# Step 5: Align and compare binary forest maps
print("\n[5/6] Aligning and comparing binary forest maps...")
# comparison_results = align_and_compare_maps(glance_forest, terraclass_forest)

# Step 6: Compute accuracy metrics (triggers computation)
print("\n[6/6] Computing confusion matrix for forest classification...")
# This will be a 2x2 matrix: Forest vs Non-Forest
# cm = compute_confusion_matrix_dask(
#     comparison_results['glance_aligned'],
#     comparison_results['reference_aligned'],
#     comparison_results['valid_mask']
# )

# Step 7: Calculate and display metrics
# metrics = compute_accuracy_metrics(cm)
# print_accuracy_report(metrics, class_names=['Non-Forest', 'Forest'])

# Step 8: Visualize
# fig_cm = plot_confusion_matrix(cm, class_names=['Non-Forest', 'Forest'], normalize=True)
# fig_vis = visualize_comparison(
#     comparison_results['glance_aligned'],
#     comparison_results['reference_aligned'],
#     comparison_results['agreement_map']
# )

# Step 9: Save results
# save_comparison_results(
#     comparison_results['agreement_map'],
#     metrics,
#     output_dir='/projectnb/modislc/users/chishan/data/forest_comparison_2018'
# )

print("\n" + "=" * 70)
print("Workflow ready! Update paths and uncomment steps 3-9 to run.")
print("=" * 70)