# CCDC Analysis Module

A Python module for running CCDC (Continuous Change Detection and Classification) analysis on Landsat NDVI time series with integrated visualization.

## Overview

This module provides a streamlined interface for:
- Preprocessing Landsat Collection 2 Level-2 Surface Reflectance data (Landsat 5, 7, 8, 9)
- Running CCDC temporal segmentation algorithm
- Extracting harmonic model coefficients and fit values
- Visualizing CCDC segments with Landsat imagery

## Installation

### Prerequisites

```bash
pip install earthengine-api pandas numpy matplotlib pillow requests
```

### Setup

1. Place `ccdc_analysis.py` in your project directory or Python path
2. Initialize Earth Engine authentication:

```python
import ee
ee.Authenticate()  # First time only
ee.Initialize()
```

## Quick Start

### Basic Usage

```python
from ccdc_analysis import run_ccdc_analysis

# Run analysis for a single point
result = run_ccdc_analysis(
    lat=-10.5,           # Latitude
    lon=-50.2,           # Longitude
    start_year=2000,     # Start year
    end_year=2022,       # End year
    target_year=2010     # Year to highlight (optional)
)

# Display the plot
import matplotlib.pyplot as plt
plt.show()
```

### Using with CSV Data

```python
import pandas as pd
from ccdc_analysis import run_ccdc_analysis

# Load your point data
df_points = pd.read_csv('your_points.csv')
point = df_points[df_points['TARGETID'] == 37464].iloc[0]

# Run analysis
result = run_ccdc_analysis(
    lat=point['LAT'],
    lon=point['LON'],
    start_year=2000,
    end_year=2022,
    target_year=2006,
    point_id=point['TARGETID'],
    biome=point.get('BIOMA_250K', 'Unknown'),
    use_false_color=True  # False color (NIR-R-G)
)
```

## Main Function: `run_ccdc_analysis()`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lat` | float | **required** | Latitude of the point location |
| `lon` | float | **required** | Longitude of the point location |
| `start_year` | int | **required** | Start year for analysis (e.g., 2000) |
| `end_year` | int | **required** | End year for analysis (e.g., 2022) |
| `target_year` | int | None | Year to highlight in visualization |
| `point_id` | int/str | None | Point identifier for labeling |
| `biome` | str | None | Biome name for labeling |
| `use_false_color` | bool | False | True for false color (NIR-R-G), False for true color (RGB) |
| `buffer_distance` | float | 1500 | Buffer distance in meters for imagery ROI |
| `cloud_threshold` | float | 30 | Maximum cloud cover percentage for imagery |
| `ccdc_lambda` | float | 0.002 | CCDC regularization parameter |
| `ccdc_max_iter` | int | 10000 | Maximum CCDC iterations |
| `min_observations` | int | 6 | Minimum observations per segment |
| `chi_square_prob` | float | 0.99 | Chi-square probability threshold |
| `verbose` | bool | True | Print progress messages |

### Returns

Dictionary containing:
- `'df_ccdc'`: pandas DataFrame with NDVI time series and fit values
- `'segments'`: List of tuples (segment_index, start_date, end_date)
- `'fits'`: Earth Engine dictionary with CCDC coefficients
- `'fig'`: Matplotlib figure object

### DataFrame Structure

The returned `df_ccdc` DataFrame has the following columns:
- `date`: datetime - Observation date
- `ndvi`: float - Observed NDVI value
- `fit`: float - CCDC harmonic model fit value
- `segment`: int - Segment index (0, 1, 2, ...)

## Usage Examples

### Example 1: Simple Analysis

```python
result = run_ccdc_analysis(
    lat=-10.5,
    lon=-50.2,
    start_year=2000,
    end_year=2022,
    target_year=2010
)

# Access results
df = result['df_ccdc']
segments = result['segments']
fig = result['fig']

# Save figure
fig.savefig('ccdc_analysis.png', dpi=300, bbox_inches='tight')
```

### Example 2: Batch Processing

```python
import pandas as pd
from ccdc_analysis import run_ccdc_analysis, export_ccdc_results

# Load points
df_points = pd.read_csv('validation_points.csv')

# Process multiple points
output_dir = '/path/to/output'

for idx, row in df_points.iterrows():
    result = run_ccdc_analysis(
        lat=row['LAT'],
        lon=row['LON'],
        start_year=2000,
        end_year=2022,
        target_year=2010,
        point_id=row['TARGETID'],
        verbose=True
    )
    
    # Export results
    export_ccdc_results(result, output_dir, row['TARGETID'])
    
    # Close figure to free memory
    import matplotlib.pyplot as plt
    plt.close(result['fig'])
```

### Example 3: Custom Analysis

```python
result = run_ccdc_analysis(
    lat=-10.5,
    lon=-50.2,
    start_year=2000,
    end_year=2022
)

# Access time series data
df = result['df_ccdc']

# Calculate statistics by year
yearly_stats = df.groupby(df['date'].dt.year).agg({
    'ndvi': ['mean', 'std', 'count'],
    'fit': 'mean'
})

# Calculate RMSE
rmse = ((df['ndvi'] - df['fit']) ** 2).mean() ** 0.5
print(f"Overall RMSE: {rmse:.3f}")

# Export to CSV
df.to_csv('my_ccdc_results.csv', index=False)
```

### Example 4: False Color Imagery

```python
# Use false color (NIR-Red-Green) for vegetation analysis
result = run_ccdc_analysis(
    lat=-10.5,
    lon=-50.2,
    start_year=2000,
    end_year=2022,
    target_year=2010,
    use_false_color=True,  # NIR-R-G composite
    buffer_distance=2000,  # Larger imagery area
    cloud_threshold=20     # Stricter cloud filtering
)
```

## Utility Functions

### `export_ccdc_results(result, output_path, point_id)`

Export CCDC analysis results to files.

**Exports:**
- `ccdc_point_{point_id}.csv` - Time series data
- `ccdc_point_{point_id}.png` - Visualization
- `ccdc_segments_{point_id}.txt` - Segment statistics

```python
from ccdc_analysis import export_ccdc_results

export_ccdc_results(result, '/path/to/output', point_id=12345)
```

### `plot_ccdc_from_dataframe(df, point_id, lat, lon, biome=None, figsize=(16, 6))`

Create a simplified CCDC plot from pre-computed DataFrame.

```python
from ccdc_analysis import plot_ccdc_from_dataframe

fig = plot_ccdc_from_dataframe(
    df=df_ccdc,
    point_id=12345,
    lat=-10.5,
    lon=-50.2,
    biome='Amazon'
)
```

## Algorithm Details

### Preprocessing

The module applies Collection 2 Level-2 preprocessing:

**Scale Factors:**
- Optical bands: `DN * 0.0000275 - 0.2`
- Thermal bands: `DN * 0.00341802 + 149.0`

**Quality Masks:**
- QA_PIXEL: Clear land pixels only
- QA_RADSAT: No radiometric saturation
- Valid reflectance range: 0-1
- Atmospheric opacity/aerosol filtering

### CCDC Algorithm

**Default Parameters:**
- Lambda (λ): 0.002 (regularization parameter)
- Max iterations: 10000
- Min observations: 6 per segment
- Chi-square probability: 0.99
- Date format: Fractional years

**Harmonic Model:**

NDVI(t) = c₀ + c₁·t + Σᵢ₌₁³ [c₂ᵢ·cos(ωᵢt) + c₂ᵢ₊₁·sin(ωᵢt)]

Where:
- t = fractional year
- ω = 2π (annual frequency)
- 8 coefficients per segment

## Data Requirements

### Landsat Collections Used

- **Landsat 5 TM**: `LANDSAT/LT05/C02/T1_L2`
- **Landsat 7 ETM+**: `LANDSAT/LE07/C02/T1_L2`
- **Landsat 8 OLI**: `LANDSAT/LC08/C02/T1_L2`
- **Landsat 9 OLI-2**: `LANDSAT/LC09/C02/T1_L2`

### Band Mapping

| Standard | L5/L7 | L8/L9 |
|----------|-------|-------|
| BLUE     | SR_B1 | SR_B2 |
| GREEN    | SR_B2 | SR_B3 |
| RED      | SR_B3 | SR_B4 |
| NIR      | SR_B4 | SR_B5 |
| SWIR1    | SR_B5 | SR_B6 |
| SWIR2    | SR_B7 | SR_B7 |

## Visualization

The module creates a 2-row figure:

**Top Row: CCDC Time Series**
- Scatter plot: Observed NDVI values
- Line plots: Harmonic fit curves per segment
- Color-coded segments with background shading
- Optional target year highlighting

**Bottom Row: Landsat Imagery**
- 5 images centered on target year (±2 years)
- True color (RGB) or false color (NIR-R-G)
- Center point marker
- Image count and year labels

## Troubleshooting

### No images available
- Check date range covers Landsat missions
- Verify location has Landsat coverage
- Try relaxing cloud_threshold parameter

### Memory errors
- Reduce time range (fewer years)
- Increase buffer_distance for smaller imagery
- Process points individually instead of batch

### Authentication errors
```python
import ee
ee.Authenticate()  # Re-authenticate
ee.Initialize()
```

## Performance Tips

1. **Batch processing**: Process points sequentially and close figures
2. **Time range**: Limit to 15-20 years for faster processing
3. **Cloud filtering**: Use 20-40% threshold for balance
4. **Verbose output**: Set `verbose=False` for batch jobs

## Citation

If you use this module, please cite the CCDC algorithm:

> Zhu, Z., & Woodcock, C. E. (2014). Continuous change detection and classification of land cover using all available Landsat data. Remote Sensing of Environment, 144, 152-171.

## License

This module is provided for research purposes. Please ensure you comply with Google Earth Engine Terms of Service.

## Contact

For issues or questions, please contact the CODF project team.

---

**Version**: 1.0  
**Last Updated**: 2025-10-30
