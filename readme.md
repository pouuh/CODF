---
applyTo: "**"
---

# AI Coding Guidelines for DailyNotes

Purpose: enable clear, reproducible geospatial analysis for environmental monitoring and CODF impact assessment. Favor simple, linear code that’s easy to review and rerun.

## Project snapshot
- Platform: Google Earth Engine (GEE) for large-scale geospatial processing
- Data: Hansen Global Forest Change, CORINE Land Cover, CODF investment database
- Geometry: Points, lines, polygons representing investment sites
- Analyses: Multi-distance buffer summaries, temporal change detection, site/sector aggregation

## Environment and authentication (required)
```python
import ee
ee.Initialize(project='ee-zcs')  # Always use this project ID
```

## Guiding principles
- Optimize for readability and traceability over cleverness or abstraction.
- Prefer a linear, step-by-step flow (esp. in notebooks and scripts).
- Name things by scientific meaning (e.g., loss_ha_2015_2020, buffer_5km, sector_code).
- Comment intent (why, what) more than mechanics (how); keep comments short but informative.
- Avoid try/except unless it guards a critical execution boundary (e.g., export submission).
- Accept small duplication when it improves clarity and reduces cognitive load.

## Coding patterns to prefer
- Keep analysis scripts self-contained and top-down; put helpers at the bottom if needed.
- Use explicit constants for parameters (years, buffer distances, scales, reducers).
- Keep function signatures simple; pass plain values (year_start, year_end, distances_km).
- Log key milestones with brief prints (e.g., print("Submitting export: ...")).
- Save intermediate results only when they’re reused or are critical checkpoints.

## Earth Engine best practices (important)
- Be explicit about scale and projection in reducers; align to source data where possible.
- Avoid client-side materialization (getInfo, toList on large collections). Stay server-side.
- Use map/reduce patterns instead of iterative client loops; break work into tiles/years if needed.
- For heavy reducers, consider tileScale to mitigate memory/timeouts; measure impact.
- Export results (tables/images) with clear names and metadata; use a consistent destination per workflow.
- Guard exports with minimal checks; retry sensibly on transient failures.

## Data handling and semantics
- Distinguish area vs. rate vs. count explicitly in names (loss_ha vs. loss_rate vs. pixels_n).
- Keep buffer distances in kilometers and reflect units in names (e.g., buffer_0_1_1_5_25_km).
- Document temporal windows (e.g., pre_5yr, post_5yr) and ensure alignment across layers.
- When aggregating by sector/site, include stable IDs and provenance fields in outputs.

## Minimum documentation per script/notebook
- Header: purpose, inputs (datasets, geometry), outputs (tables/images), time window.
- Parameter block: main knobs (years, distances, reducers) with a short rationale.
- One usage example (how to run or how to set AOI) if not obvious.

## Quick scaffold (buffer-based change summary)
```python
import ee
ee.Initialize(project='ee-zcs')

# Parameters
years = (2015, 2020)
buffer_km = [0.1, 1, 5, 25]
scale = 30  # meters; match source data when possible
buffers_km = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0]
```

## File Naming Conventions
Follow this pattern for output files:
```
{dataset}_{analysis_type}_{geometry_type}_buffer_{distance}km_{time_period}_{threshold}.csv
```
Example: `hansen_annual_defor_all_types_buffer_1.0km_2001_2019_30pct.csv`

## Earth Engine Assets
- CODF Polygons: `projects/ee-zcs/assets/CODF_polygons`
- CODF Points: `projects/ee-zcs/assets/CODF_points`
- CODF Lines: `projects/ee-zcs/assets/CODF_lines`
- Hansen Data: `UMD/hansen/global_forest_change_2024_v1_12`

## Analysis Parameters
- **Forest Threshold**: 30% tree cover (Hansen data)
- **Analysis Period**: 2001-2019 (forest loss)
- **CODF Filtering**: Loan year < 2015, Level of precision = 1, valid_data_percentage > 0.9

## Export Pattern
```python
task = ee.batch.Export.table.toDrive({
    'collection': feature_collection,
    'description': 'descriptive_name',
    'folder': 'EarthEngineExports',
    'fileNamePrefix': 'filename_prefix',
    'fileFormat': 'CSV'
})
task.start()
```

## Data Processing Workflow
1. Load CODF features from Earth Engine assets
2. Filter by quality criteria (precision, loan year, data availability)
3. Create buffer zones at standard distances
4. Process satellite data within buffers
5. Export results to Google Drive
6. Post-process CSVs with pandas for statistics and visualization

## Key Directories
- `/CODF_output/`: Processed CODF shapefiles and metadata
- `/papers/`: Extracted PDF content for literature review
- Root level: Analysis results CSVs, notebooks, and scripts

## Common Libraries
- `ee`: Google Earth Engine
- `pandas`, `geopandas`: Data processing
- `matplotlib`, `seaborn`: Visualization
- `scipy`: Statistical analysis
- `fitz` (PyMuPDF): PDF processing

## Quality Assurance
- Always combine all geometry types (points, lines, polygons) for comprehensive analysis
- Include geometry_type property in all feature collections
- Validate data availability before analysis
- Use consistent forest thresholds and time periods

## Debugging Tips
- Check Earth Engine task status in Google Earth Engine Code Editor
- Verify buffer creation with small test areas first
- Monitor Google Drive for export completion
- Use `ee.FeatureCollection.size().getInfo()` to verify data loading</content>
