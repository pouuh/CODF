#!/usr/bin/env python3
"""
GLANCE vs MapBiomas Amazon Forest Classification Comparison
============================================================
High-performance comparison of GLANCE and MapBiomas forest classifications
using Dask for distributed processing.

Author: Generated for Amazon Forest Analysis
Date: 2025-10-15
"""

import os
import glob
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr
import rioxarray
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ============================================================================
# 1. QML LEGEND PARSING
# ============================================================================

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


# ============================================================================
# 2. MAPBIOMAS RECLASSIFICATION
# ============================================================================

def create_mapbiomas_forest_map(classes_dict, include_plantations=True):
    """
    Create forest/non-forest reclassification for MapBiomas Amazon classes
    
    MapBiomas Forest Classes:
    - 1: VEGETACAO NATURAL FLORESTAL PRIMARIA (Primary Forest)
    - 2: VEGETACAO NATURAL FLORESTAL SECUNDARIA (Secondary Forest)
    - 9: SILVICULTURA (Forestry/Plantation) - optional
    
    Parameters:
    -----------
    classes_dict : dict
        Dictionary from parse_qml_legend()
    include_plantations : bool
        Whether to include silvicultura (class 9) as forest (default: True)
    
    Returns:
    --------
    dict : Reclassification map (value -> 1=Forest, 0=Non-Forest)
    """
    # Define forest classes explicitly for MapBiomas
    if include_plantations:
        FOREST_CLASSES = [1, 2, 9]  # Include silvicultura (plantations)
    else:
        FOREST_CLASSES = [1, 2]  # Primary and Secondary Natural Forest only
    
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

def reclassify_to_binary_forest(raster_array, reclass_map, nodata_value=0):
    """
    Reclassify any land cover map to binary Forest/Non-Forest (lazy version)
    
    Parameters:
    -----------
    raster_array : xarray.DataArray or numpy.ndarray
        Classification array
    reclass_map : dict
        Reclassification mapping (value -> 1=Forest, 0=Non-Forest)
    nodata_value : int
        Value to use for nodata pixels (default: 0)
    
    Returns:
    --------
    xarray.DataArray : Binary classification (1=Forest, 0=Non-Forest, nodata=0)
    """
    # Work with xarray directly (lazy) - NO COPY!
     # Work with xarray directly (lazy) - NO COPY!
    if isinstance(raster_array, xr.DataArray):
        arr = raster_array
    else:
        raise ValueError("Input must be xarray.DataArray for lazy processing")
    
    # Squeeze band dimension if present
    if 'band' in arr.dims and len(arr.band) == 1:
        arr = arr.squeeze('band', drop=True)

    # Ensure lazy dask-backed storage to avoid allocating huge numpy arrays
    if not isinstance(arr.data, da.Array):
        chunk_map = {}
        for dim in arr.dims:
            if dim == 'band':
                chunk_map[dim] = 1
            else:
                chunk_map[dim] = min(4096, arr.sizes[dim])
        arr = arr.chunk(chunk_map)
    
    # Create output using xarray.where for lazy evaluation
    output = xr.full_like(arr, nodata_value, dtype=np.uint8)
    
    # Apply reclassification using lazy operations
    for old_value, new_value in reclass_map.items():
        output = xr.where(arr == old_value, new_value, output)
    
    # Handle nodata - mark pixels not in reclass_map as nodata
    valid_values = list(reclass_map.keys())
    is_valid = xr.zeros_like(arr, dtype=bool)
    for val in valid_values:
        is_valid = is_valid | (arr == val)
    
    output = xr.where(~is_valid, nodata_value, output)
    
    # Copy metadata (shallow copy of attrs dict is fine)
    output.attrs = {**arr.attrs, 'long_name': 'Forest classification (1=Forest, 0=Non-Forest)'}
    if hasattr(arr, 'rio'):
        output.rio.write_crs(arr.rio.crs, inplace=True)
        output.rio.write_nodata(nodata_value, inplace=True)
    
    # Print statistics (will compute only small chunks)
    print(f"Reclassification complete (lazy evaluation)")
    print(f"  Output shape: {output.shape}")
    print(f"  Output chunks: {output.chunks}")
    
    return output
# ============================================================================
# 3. GLANCE TILE LOADING
# ============================================================================

def find_glance_tiles_by_year(base_folder, year, region='SA'):
    """
    Find all GLANCE tiles for a specific year and region
    
    Parameters:
    -----------
    base_folder : str
        Base directory containing GLANCE data
    year : int
        Target year (e.g., 2016)
    region : str
        Region code (default: 'SA' for South America)
    
    Returns:
    --------
    list : Sorted list of file paths matching the year and region
    """
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
    try:
        from rioxarray.merge import merge_arrays
        mosaic = merge_arrays(datasets, nodata=0)
        print(f"Merged mosaic shape: {mosaic.shape}")
        print(f"Merged mosaic dtype: {mosaic.dtype}")
        print(f"Chunk size: {mosaic.chunks}")
        return mosaic
    except Exception as e:
        print(f"Merge failed: {e}")
        print("Falling back to first tile only")
        return datasets[0]

# ============================================================================
# 4. GLANCE RECLASSIFICATION
# ============================================================================

def reclassify_glance_to_forest(glance_array, glance_forest_classes=None):
    """
    Reclassify GLANCE land cover to binary Forest/Non-Forest
    
    GLANCE classes (typical):
    1-7: ['Water', 'Snow/Ice', 'Developed', 'Bare', 'Forest', 'Shrub', 'Herbaceous']
    
    Parameters:
    -----------
    glance_array : xarray.DataArray
        GLANCE classification array
    glance_forest_classes : list, optional
        List of GLANCE class values considered as forest (default: [5])
    
    Returns:
    --------
    xarray.DataArray : Binary forest classification (1=Forest, 0=Non-Forest)
    """
    # Default forest classes
    if glance_forest_classes is None:
        glance_forest_classes = [5]  # Mixed Forest (typical for Amazon)
        # For all forest types, use: [1, 2, 3, 4, 5]
    
    print("GLANCE Forest Classification:")
    print(f"  Forest classes: {glance_forest_classes}")
    
    # Create reclassification map
    # GLANCE has classes 1-7, we need to map them
    reclass_map = {i: (1 if i in glance_forest_classes else 0) for i in range(1, 8)}

    return reclassify_to_binary_forest(glance_array, reclass_map, nodata_value=0)


# ============================================================================
# 5. MAP ALIGNMENT AND COMPARISON
# ============================================================================

def align_and_compare_maps(glance_mosaic, reference_map, mask_nodata=True):
    """
    Align two maps to the same grid and compare classifications
    
    Parameters:
    -----------
    glance_mosaic : xarray.DataArray
        GLANCE classification mosaic
    reference_map : xarray.DataArray
        Reference classification map
    mask_nodata : bool
        Whether to mask nodata values (default: True)
    
    Returns:
    --------
    dict : Dictionary containing aligned data and agreement map
    """
    # Ensure both maps have same CRS
    if glance_mosaic.rio.crs != reference_map.rio.crs:
        print("Reprojecting reference map to match GLANCE CRS...")
        reference_map = reference_map.rio.reproject_match(glance_mosaic, resampling=5)
    
    # Get overlapping extent
    glance_bounds = glance_mosaic.rio.bounds()
    ref_bounds = reference_map.rio.bounds()
    
    overlap_bounds = (
        max(glance_bounds[0], ref_bounds[0]),  # left
        max(glance_bounds[1], ref_bounds[1]),  # bottom
        min(glance_bounds[2], ref_bounds[2]),  # right
        min(glance_bounds[3], ref_bounds[3])   # top
    )
    
    print(f"Overlap bounds: {overlap_bounds}")
    
    # Clip both maps to overlap extent
    glance_clip = glance_mosaic.rio.clip_box(*overlap_bounds)
    ref_clip = reference_map.rio.clip_box(*overlap_bounds)
    
    # Reproject reference to exactly match GLANCE grid
    ref_aligned = ref_clip.rio.reproject_match(glance_clip, resampling=5)
    
    print(f"Aligned GLANCE shape: {glance_clip.shape}")
    print(f"Aligned reference shape: {ref_aligned.shape}")
    
    # Squeeze to remove band dimension if present
    glance_arr = glance_clip.squeeze() if len(glance_clip.shape) > 2 else glance_clip
    ref_arr = ref_aligned.squeeze() if len(ref_aligned.shape) > 2 else ref_aligned
    
    # Create masks for valid data (use xarray methods to stay lazy)
    if mask_nodata:
        # ✅ Use xarray's .notnull() method instead of np.isnan()
        # This stays lazy and works with dask arrays
        valid_mask = (
            (glance_arr != 0) & 
            (ref_arr != 0) & 
            glance_arr.notnull() & 
            ref_arr.notnull()
        )
    else:
        valid_mask = xr.ones_like(glance_arr, dtype=bool)
    
    # Agreement map
    agreement_map = xr.where(valid_mask, (glance_arr == ref_arr).astype(np.int8), -1)
    
    # Overall accuracy (lazy - will compute when called)
    n_agree = (agreement_map == 1).sum()
    n_valid = valid_mask.sum()
    
    # print(f"Valid pixels for comparison: {n_valid.values:,}")
    
    return {
        'glance_aligned': glance_arr,
        'reference_aligned': ref_arr,
        'agreement_map': agreement_map,
        'valid_mask': valid_mask,
        'n_agree': n_agree,
        'n_valid': n_valid
    }


# ============================================================================
# 6. CONFUSION MATRIX AND ACCURACY METRICS
# ============================================================================

def compute_confusion_matrix_dask(glance_arr, ref_arr, valid_mask):
    """
    Compute confusion matrix using Dask for memory efficiency
    
    Parameters:
    -----------
    glance_arr : xarray.DataArray
        GLANCE classification array
    ref_arr : xarray.DataArray
        Reference classification array
    valid_mask : xarray.DataArray
        Boolean mask for valid pixels
    
    Returns:
    --------
    numpy.ndarray : Confusion matrix (rows=reference, cols=GLANCE)
    """
    print("Computing confusion matrix with chunked processing...")
    
    # METHOD 1: Use dask.array operations to compute confusion matrix efficiently
    # This avoids loading entire arrays into memory
    
    # Ensure all arrays are dask-backed
    if not hasattr(glance_arr.data, 'compute'):
        glance_arr = glance_arr.chunk({'x': 4096, 'y': 4096})
    if not hasattr(ref_arr.data, 'compute'):
        ref_arr = ref_arr.chunk({'x': 4096, 'y': 4096})
    if not hasattr(valid_mask.data, 'compute'):
        valid_mask = valid_mask.chunk({'x': 4096, 'y': 4096})
    
    # Flatten to 1D dask arrays (still lazy)
    glance_flat = glance_arr.data.ravel()
    ref_flat = ref_arr.data.ravel()
    mask_flat = valid_mask.data.ravel()
    
    # Apply mask using dask (still lazy)
    glance_masked = glance_flat[mask_flat]
    ref_masked = ref_flat[mask_flat]
    
    # Compute the total count first (small operation)
    n_valid = mask_flat.sum().compute()
    print(f"Computing confusion matrix for {n_valid:,} valid pixels...")
    print("  This may take several minutes for large datasets...")
    
    # For 2x2 confusion matrix, compute each cell separately to control memory
    # This is more memory-efficient than loading all data at once
    print("  Computing True Negatives (both=0)...")
    tn = ((ref_masked == 0) & (glance_masked == 0)).sum().compute()
    
    print("  Computing False Positives (ref=0, glance=1)...")
    fp = ((ref_masked == 0) & (glance_masked == 1)).sum().compute()
    
    print("  Computing False Negatives (ref=1, glance=0)...")
    fn = ((ref_masked == 1) & (glance_masked == 0)).sum().compute()
    
    print("  Computing True Positives (both=1)...")
    tp = ((ref_masked == 1) & (glance_masked == 1)).sum().compute()
    
    # Construct confusion matrix
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=np.int64)
    
    print(f"✓ Confusion matrix computed successfully")
    print(f"  Total pixels: {cm.sum():,}")
    
    return cm


def compute_forest_accuracy_metrics(cm):
    """
    Compute detailed forest-specific accuracy metrics from 2x2 confusion matrix
    
    Confusion matrix layout:
                      Predicted
                   Non-Forest  Forest
    Reference  
    Non-Forest      TN         FP
    Forest          FN         TP
    
    Parameters:
    -----------
    cm : numpy.ndarray
        2x2 confusion matrix (rows=reference, cols=predicted)
    
    Returns:
    --------
    dict : Comprehensive forest accuracy metrics
    """
    # Extract confusion matrix components
    tn = cm[0, 0]  # True Negative
    fp = cm[0, 1]  # False Positive
    fn = cm[1, 0]  # False Negative
    tp = cm[1, 1]  # True Positive
    
    total = cm.sum()
    
    # Overall accuracy
    overall_acc = (tp + tn) / total
    
    # Producer's accuracy (Recall/Sensitivity)
    forest_producers = tp / (tp + fn) if (tp + fn) > 0 else 0
    nonforest_producers = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # User's accuracy (Precision)
    forest_users = tp / (tp + fp) if (tp + fp) > 0 else 0
    nonforest_users = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # F1 Score for forest class
    forest_f1 = 2 * (forest_producers * forest_users) / (forest_producers + forest_users) \
                if (forest_producers + forest_users) > 0 else 0
    
    # Kappa coefficient
    p_o = overall_acc
    p_e = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total ** 2)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0
    
    # Commission and Omission errors
    commission_error = fp / (tp + fp) if (tp + fp) > 0 else 0
    omission_error = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics = {
        'confusion_matrix': cm,
        'overall_accuracy': overall_acc,
        'kappa': kappa,
        'forest_producers_accuracy': forest_producers,
        'forest_users_accuracy': forest_users,
        'forest_f1_score': forest_f1,
        'forest_commission_error': commission_error,
        'forest_omission_error': omission_error,
        'nonforest_producers_accuracy': nonforest_producers,
        'nonforest_users_accuracy': nonforest_users,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'total_pixels': int(total)
    }
    
    return metrics


def print_forest_accuracy_report(metrics):
    """
    Print formatted forest accuracy assessment report
    """
    print("\n" + "=" * 80)
    print("FOREST CLASSIFICATION ACCURACY ASSESSMENT")
    print("=" * 80)
    
    print(f"\nTotal pixels analyzed: {metrics['total_pixels']:,}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"Kappa Coefficient: {metrics['kappa']:.4f}")
    
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX:")
    print("-" * 80)
    print("                    Predicted Non-Forest    Predicted Forest")
    print(f"Reference Non-Forest     {metrics['true_negative']:12,}     {metrics['false_positive']:12,}")
    print(f"Reference Forest         {metrics['false_negative']:12,}     {metrics['true_positive']:12,}")
    
    print("\n" + "-" * 80)
    print("FOREST CLASS METRICS:")
    print("-" * 80)
    print(f"Producer's Accuracy (Recall):  {metrics['forest_producers_accuracy']:.4f} ({metrics['forest_producers_accuracy']*100:.2f}%)")
    print(f"  → {metrics['forest_producers_accuracy']*100:.2f}% of reference forest correctly detected")
    print(f"User's Accuracy (Precision):   {metrics['forest_users_accuracy']:.4f} ({metrics['forest_users_accuracy']*100:.2f}%)")
    print(f"  → {metrics['forest_users_accuracy']*100:.2f}% of predicted forest is actually forest")
    print(f"F1 Score:                      {metrics['forest_f1_score']:.4f}")
    print(f"\nCommission Error (False Positives): {metrics['forest_commission_error']:.4f} ({metrics['forest_commission_error']*100:.2f}%)")
    print(f"  → {metrics['false_positive']:,} non-forest pixels misclassified as forest")
    print(f"Omission Error (False Negatives):   {metrics['forest_omission_error']:.4f} ({metrics['forest_omission_error']*100:.2f}%)")
    print(f"  → {metrics['false_negative']:,} forest pixels missed")
    
    print("\n" + "-" * 80)
    print("NON-FOREST CLASS METRICS:")
    print("-" * 80)
    print(f"Producer's Accuracy: {metrics['nonforest_producers_accuracy']:.4f} ({metrics['nonforest_producers_accuracy']*100:.2f}%)")
    print(f"User's Accuracy:     {metrics['nonforest_users_accuracy']:.4f} ({metrics['nonforest_users_accuracy']*100:.2f}%)")
    
    print("\n" + "=" * 80)


# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, class_names=None, normalize=False, figsize=(8, 6)):
    """
    Plot confusion matrix as heatmap
    """
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_norm
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('GLANCE Classification')
    ax.set_ylabel('MapBiomas Reference')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def save_results(comparison, metrics, output_dir, year):
    """
    Save comparison results to disk
    """
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save agreement map with chunked writing for memory efficiency
    agreement_path = os.path.join(output_dir, f'agreement_map_{year}_{timestamp}.tif')
    print(f"Saving agreement map to {agreement_path}...")
    print("  ⏳ Writing in tiles to manage memory...")
    # ✅ Add tiled=True and blocksize for chunked writing
    comparison['agreement_map'].rio.to_raster(
        agreement_path, 
        compress='LZW',
        tiled=True,
        blockxsize=512,
        blockysize=512
    )
    print("  ✓ Agreement map saved")
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, f'confusion_matrix_{year}_{timestamp}.csv')
    np.savetxt(cm_path, metrics['confusion_matrix'], delimiter=',', fmt='%d')
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'accuracy_metrics_{year}_{timestamp}.json')
    metrics_export = {
        'year': year,
        'overall_accuracy': float(metrics['overall_accuracy']),
        'kappa': float(metrics['kappa']),
        'forest_producers_accuracy': float(metrics['forest_producers_accuracy']),
        'forest_users_accuracy': float(metrics['forest_users_accuracy']),
        'forest_f1_score': float(metrics['forest_f1_score']),
        'forest_commission_error': float(metrics['forest_commission_error']),
        'forest_omission_error': float(metrics['forest_omission_error']),
        'true_positive': metrics['true_positive'],
        'true_negative': metrics['true_negative'],
        'false_positive': metrics['false_positive'],
        'false_negative': metrics['false_negative'],
        'total_pixels': metrics['total_pixels'],
        'timestamp': timestamp
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_export, f, indent=2)
    
    print(f"✓ Results saved to: {output_dir}")


# ============================================================================
# 8. MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow for GLANCE vs MapBiomas forest comparison
    """
    # Configuration
    year_target = 2016
    folder_glance = '/projectnb/measures/products/SA/v001/DAAC/LC/'
    folder_mapbiomas = '/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/'
    output_dir = f'/projectnb/modislc/users/chishan/data/forest_comparison_{year_target}'
    
    # File paths
    mapbiomas_tif = os.path.join(folder_mapbiomas, f'AMZ.{year_target}.M.tif')
    qml_file = os.path.join(folder_mapbiomas, f'AMZ.{year_target}.M.qml')
    
    print("=" * 80)
    print(f"GLANCE vs MapBiomas FOREST COMPARISON - {year_target} Amazon")
    print("=" * 80)
    
    # Step 1: Parse MapBiomas legend
    print("\n[1/7] Parsing MapBiomas legend...")
    classes = parse_qml_legend(qml_file)
    
    # Step 2: Create reclassification map
    print("\n[2/7] Creating MapBiomas forest reclassification...")
    reclass_map = create_mapbiomas_forest_map(classes, include_plantations=True)
    
    # Step 3: Load and reclassify MapBiomas
    print("\n[3/7] Loading and reclassifying MapBiomas...")
    mapbiomas_ref = rioxarray.open_rasterio(mapbiomas_tif, chunks={'band': 1, 'x': 4096, 'y': 4096}, lock=False)
    print(f"  MapBiomas shape: {mapbiomas_ref.shape}")
    print(f"  MapBiomas CRS: {mapbiomas_ref.rio.crs}")
    mapbiomas_binary = reclassify_to_binary_forest(mapbiomas_ref, reclass_map, nodata_value=0)
    
    # Step 4: Find and load GLANCE tiles
    print("\n[4/7] Finding and loading GLANCE tiles...")
    tiles = find_glance_tiles_by_year(folder_glance, year_target, 'SA')
    glance_mosaic = load_glance_tiles_lazy(tiles, chunks={'band': 1, 'x': 4096, 'y': 4096})
    
    # Step 5: Reclassify GLANCE
    print("\n[5/7] Reclassifying GLANCE to forest/non-forest...")
    glance_binary = reclassify_glance_to_forest(glance_mosaic, glance_forest_classes=[5])
    
    # Step 6: Align and compare
    print("\n[6/7] Aligning and comparing datasets...")
    comparison = align_and_compare_maps(glance_binary, mapbiomas_binary, mask_nodata=True)
    
    # Step 7: Compute accuracy metrics
    print("\n[7/7] Computing confusion matrix and accuracy metrics...")
    print("  ⏳ This may take several minutes...")
    cm = compute_confusion_matrix_dask(
        comparison['glance_aligned'],
        comparison['reference_aligned'],
        comparison['valid_mask']
    )
    
    metrics = compute_forest_accuracy_metrics(cm)
    print_forest_accuracy_report(metrics)
    
    # Visualization
    print("\nCreating visualizations...")
    fig = plot_confusion_matrix(cm, class_names=['Non-Forest', 'Forest'], normalize=True)
    fig_path = os.path.join(output_dir, f'confusion_matrix_{year_target}.png')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved confusion matrix plot to {fig_path}")
    
    # Save results
    print("\nSaving results...")
    save_results(comparison, metrics, output_dir, year_target)
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    
    return metrics, comparison


if __name__ == "__main__":
    # Initialize Dask (optional but recommended for large datasets)
    print("Initializing Dask cluster...")
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=16,
        memory_limit='128GB'
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")
    
    # Run main workflow
    metrics, comparison = main()
    
    # Close Dask client
    client.close()
    cluster.close()
