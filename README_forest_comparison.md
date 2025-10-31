# GLANCE vs MapBiomas Forest Comparison Tool

## 📋 Overview

High-performance Python script for comparing GLANCE and MapBiomas Amazon forest classifications using Dask for distributed processing.

## 🚀 Quick Start

### 1. Activate your environment and install dependencies

```bash
# If needed, install required packages
pip install numpy xarray rioxarray dask distributed matplotlib seaborn scikit-learn
```

### 2. Run the script

```bash
# Simple run
python 20251015_forest_comparison.py

# Or submit as HPC job
qsub run_forest_comparison.sh
```

## 📂 File Structure

```
20251015_forest_comparison.py  # Main analysis script
README_forest_comparison.md    # This file
```

## ⚙️ Configuration

Edit these variables in the `main()` function:

```python
year_target = 2016  # Target year for comparison
folder_glance = '/projectnb/measures/products/SA/v001/DAAC/LC/'
folder_mapbiomas = '/projectnb/modislc/users/chishan/data/MapBiomas/MAPBIOMAS/'
output_dir = f'/projectnb/modislc/users/chishan/data/forest_comparison_{year_target}'
```

## 🌳 Forest Definitions

### MapBiomas (Reference)
- **Class 1**: Primary Natural Forest
- **Class 2**: Secondary Natural Forest
- **Class 9**: Silvicultura (Plantations) - *included by default*

To exclude plantations, set `include_plantations=False` in:
```python
reclass_map = create_mapbiomas_forest_map(classes, include_plantations=False)
```

### GLANCE (Predicted)
- **Class 5**: Mixed Forest (default for Amazon)

To include all forest types (classes 1-5), modify:
```python
glance_binary = reclassify_glance_to_forest(glance_mosaic, glance_forest_classes=[1,2,3,4,5])
```

## 📊 Outputs

All outputs are saved to the `output_dir` directory:

1. **agreement_map_YYYY_timestamp.tif**
   - GeoTIFF showing pixel-by-pixel agreement
   - Values: 1=agree, 0=disagree, -1=nodata

2. **confusion_matrix_YYYY_timestamp.csv**
   - 2x2 confusion matrix (rows=reference, cols=predicted)

3. **accuracy_metrics_YYYY_timestamp.json**
   - Detailed accuracy metrics in JSON format
   - Includes overall accuracy, kappa, producer's/user's accuracy, F1 score, etc.

4. **confusion_matrix_YYYY.png**
   - Normalized confusion matrix heatmap visualization

## 📈 Key Metrics Explained

| Metric | Description |
|--------|-------------|
| **Overall Accuracy** | Percentage of pixels correctly classified |
| **Kappa Coefficient** | Agreement beyond chance (0-1, >0.8 excellent) |
| **Producer's Accuracy** | Of all MapBiomas forest, % detected by GLANCE |
| **User's Accuracy** | Of all GLANCE forest predictions, % correct |
| **F1 Score** | Harmonic mean of precision and recall |
| **Commission Error** | False positives (over-prediction) |
| **Omission Error** | False negatives (under-prediction) |

## 🔧 Advanced Usage

### Run for multiple years

```python
for year in [2006, 2010, 2014, 2016, 2018]:
    year_target = year
    # ... run analysis
```

### Adjust Dask cluster

```python
cluster = LocalCluster(
    n_workers=16,           # Increase for more parallelism
    threads_per_worker=4,   # Adjust based on cores
    memory_limit='8GB'      # Adjust based on RAM
)
```

### Subset to specific region

After alignment, subset before computing confusion matrix:
```python
# Example: Clip to specific bounding box
subset = comparison['glance_aligned'].rio.clip_box(-70, -10, -50, 5)
```

## 🐛 Troubleshooting

### Memory errors
- Reduce `n_workers` or `memory_limit` in LocalCluster
- Decrease chunk sizes: `chunks={'band': 1, 'x': 1024, 'y': 1024}`

### No GLANCE tiles found
- Check year is available in GLANCE dataset
- Verify `folder_glance` path is correct
- Some years may use different naming conventions

### CRS mismatch warnings
- Script automatically handles reprojection
- Uses nearest neighbor resampling to preserve categorical data

## 📝 Script Structure

1. **QML Legend Parsing** - Extract MapBiomas class definitions
2. **MapBiomas Reclassification** - Binary forest map creation
3. **GLANCE Tile Loading** - Lazy loading and merging of tiles
4. **GLANCE Reclassification** - Binary forest map creation
5. **Map Alignment** - CRS and grid alignment
6. **Confusion Matrix** - Pixel-by-pixel comparison
7. **Metrics Calculation** - Accuracy assessment
8. **Visualization & Export** - Results saving

## 🔬 Scientific Notes

- **Resampling**: Nearest neighbor (preserves categorical values)
- **NoData handling**: Pixels with 0 or NaN are masked
- **Spatial alignment**: Reference (MapBiomas) reprojected to match GLANCE grid
- **Confusion matrix**: Rows=Reference, Columns=Predicted (standard convention)

## 📚 References

- GLANCE: Global Land Cover with Fine Classification System
- MapBiomas: Annual land cover mapping for Brazil and Amazon
- Confusion Matrix: Standard accuracy assessment for categorical maps

## 💡 Tips

1. **Start small**: Test with one year before batch processing
2. **Monitor Dask dashboard**: Check progress and resource usage
3. **Validate results**: Check if accuracy metrics make sense
4. **Document changes**: Keep track of forest class definitions used

## 👥 Contact

For questions or issues, contact the project team.

---
**Last Updated**: 2025-10-15
**Version**: 1.0
