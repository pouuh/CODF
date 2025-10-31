#!/usr/bin/env python3
"""
Simple wrapper for running forest comparison with custom parameters

Usage:
    python run_comparison.py

Configuration is done by editing the CONFIGURATION section below.
"""

# Import directly - make sure 20251015_forest_comparison.py is in same directory
import sys
import os

# Workaround for module name starting with number
import importlib.util
spec = importlib.util.spec_from_file_location(
    "forest_comparison",
    os.path.join(os.path.dirname(__file__), "20251015_forest_comparison.py")
)
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)

# Import specific functions
parse_qml_legend = fc.parse_qml_legend
create_mapbiomas_forest_map = fc.create_mapbiomas_forest_map
reclassify_to_binary_forest = fc.reclassify_to_binary_forest
find_glance_tiles_by_year = fc.find_glance_tiles_by_year
load_glance_tiles_lazy = fc.load_glance_tiles_lazy
reclassify_glance_to_forest = fc.reclassify_glance_to_forest
align_and_compare_maps = fc.align_and_compare_maps
compute_confusion_matrix_dask = fc.compute_confusion_matrix_dask
compute_forest_accuracy_metrics = fc.compute_forest_accuracy_metrics
print_forest_accuracy_report = fc.print_forest_accuracy_report
plot_confusion_matrix = fc.plot_confusion_matrix
save_results = fc.save_results

# Import required libraries
import rioxarray
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster

# ===========================================================================
# CONFIGURATION - Edit these parameters
# ===========================================================================

# Years to analyze
YEARS = [2016]  # Add more years: [2006, 2010, 2014, 2016, 2018]

# Paths
FOLDER_GLANCE = '/projectnb/measures/products/SA/v001/DAAC/LC/'
FOLDER_MAPBIOMAS = '/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/'
OUTPUT_BASE_DIR = '/projectnb/modislc/users/chishan/data/forest_comparison_results'

# Forest class definitions
INCLUDE_PLANTATIONS = True  # Include MapBiomas class 9 (Silvicultura) as forest?
GLANCE_FOREST_CLASSES = [5]  # GLANCE classes to consider as forest
# Options:
#   [5] - Only Mixed Forest (typical for Amazon)
#   [1, 2, 3, 4, 5] - All forest types
#   [2, 5] - Evergreen Broadleaf and Mixed

# Dask configuration
N_WORKERS = 8
THREADS_PER_WORKER = 2
MEMORY_LIMIT = '4GB'

# Chunk sizes for lazy loading
CHUNK_SIZE = {'band': 1, 'x': 2048, 'y': 2048}

# ===========================================================================
# RUN ANALYSIS
# ===========================================================================

def run_analysis_for_year(year):
    """
    Run forest comparison analysis for a specific year
    """
    print("\n" + "="*80)
    print(f"ANALYZING YEAR: {year}")
    print("="*80)
    
    # Setup paths
    mapbiomas_tif = os.path.join(FOLDER_MAPBIOMAS, f'AMZ.{year}.M.tif')
    qml_file = os.path.join(FOLDER_MAPBIOMAS, f'AMZ.{year}.M.qml')
    output_dir = os.path.join(OUTPUT_BASE_DIR, f'year_{year}')
    
    # Check if files exist
    if not os.path.exists(mapbiomas_tif):
        print(f"❌ ERROR: MapBiomas file not found: {mapbiomas_tif}")
        return None
    if not os.path.exists(qml_file):
        print(f"⚠️  WARNING: QML file not found: {qml_file}")
    
    try:
        # Step 1: Parse legend
        print(f"\n[1/7] Parsing MapBiomas legend for {year}...")
        classes = parse_qml_legend(qml_file)
        
        # Step 2: Create reclassification map
        print(f"\n[2/7] Creating forest reclassification...")
        reclass_map = create_mapbiomas_forest_map(classes, include_plantations=INCLUDE_PLANTATIONS)
        
        # Step 3: Load MapBiomas
        print(f"\n[3/7] Loading MapBiomas {year}...")
        mapbiomas_ref = rioxarray.open_rasterio(mapbiomas_tif, chunks=CHUNK_SIZE, lock=False)
        print(f"  Shape: {mapbiomas_ref.shape}, CRS: {mapbiomas_ref.rio.crs}")
        mapbiomas_binary = reclassify_to_binary_forest(mapbiomas_ref, reclass_map)
        
        # Step 4: Load GLANCE
        print(f"\n[4/7] Loading GLANCE {year}...")
        tiles = find_glance_tiles_by_year(FOLDER_GLANCE, year, 'SA')
        if not tiles:
            print(f"❌ ERROR: No GLANCE tiles found for year {year}")
            return None
        glance_mosaic = load_glance_tiles_lazy(tiles, chunks=CHUNK_SIZE)
        
        # Step 5: Reclassify GLANCE
        print(f"\n[5/7] Reclassifying GLANCE...")
        glance_binary = reclassify_glance_to_forest(glance_mosaic, 
                                                     glance_forest_classes=GLANCE_FOREST_CLASSES)
        
        # Step 6: Align and compare
        print(f"\n[6/7] Aligning and comparing...")
        comparison = align_and_compare_maps(glance_binary, mapbiomas_binary)
        
        # Step 7: Compute metrics
        print(f"\n[7/7] Computing accuracy metrics...")
        cm = compute_confusion_matrix_dask(
            comparison['glance_aligned'],
            comparison['reference_aligned'],
            comparison['valid_mask']
        )
        
        metrics = compute_forest_accuracy_metrics(cm)
        print_forest_accuracy_report(metrics)
        
        # Visualization
        print(f"\nCreating visualizations...")
        fig = plot_confusion_matrix(cm, class_names=['Non-Forest', 'Forest'], normalize=True)
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'confusion_matrix_{year}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save results
        print(f"\nSaving results...")
        save_results(comparison, metrics, output_dir, year)
        
        print(f"\n✅ Year {year} analysis complete!")
        print(f"   Results saved to: {output_dir}")
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ ERROR analyzing year {year}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main entry point
    """
    print("="*80)
    print("GLANCE vs MapBiomas Forest Comparison")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Years: {YEARS}")
    print(f"  Include plantations: {INCLUDE_PLANTATIONS}")
    print(f"  GLANCE forest classes: {GLANCE_FOREST_CLASSES}")
    print(f"  Output directory: {OUTPUT_BASE_DIR}")
    print("="*80)
    
    # Initialize Dask
    print("\nInitializing Dask cluster...")
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=MEMORY_LIMIT
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")
    
    # Run analysis for each year
    results = {}
    for year in YEARS:
        metrics = run_analysis_for_year(year)
        if metrics:
            results[year] = metrics
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    for year, metrics in results.items():
        print(f"\nYear {year}:")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Kappa: {metrics['kappa']:.4f}")
        print(f"  Forest Producer's Accuracy: {metrics['forest_producers_accuracy']:.4f}")
        print(f"  Forest User's Accuracy: {metrics['forest_users_accuracy']:.4f}")
    
    # Cleanup
    client.close()
    cluster.close()
    
    print("\n" + "="*80)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
