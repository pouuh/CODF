"""
CCDC Analysis Module for Landsat NDVI Time Series

This module provides functions for running CCDC (Continuous Change Detection and 
Classification) analysis on Landsat data and visualizing results with imagery.

Author: Generated for CODF Project
Date: 2025-10-30
"""

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import math
import requests
from PIL import Image
from io import BytesIO


# ======================= PREPROCESSING FUNCTIONS =======================

def prepareL4L5L7Col2(image):
    """
    Preprocess Landsat 4/5/7 Collection 2 Level-2 Surface Reflectance.
    
    Applies scale factors, quality masks, and band renaming for standardization.
    
    Args:
        image: ee.Image from Landsat 4/5/7 Collection 2
        
    Returns:
        Preprocessed ee.Image with standardized band names
    """
    bandList = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6']
    nameList = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP']
    subBand = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
    
    # Apply Collection 2 scale factors
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0)
    scaled = opticalBands.addBands(thermalBand, None, True).select(bandList).rename(nameList)
    
    # Quality masks
    validQA = [5440, 5504]  # Clear land pixels
    mask1 = image.select(['QA_PIXEL']).remap(validQA, ee.List.repeat(1, len(validQA)), 0)
    mask2 = image.select('QA_RADSAT').eq(0)  # No radiometric saturation
    mask3 = scaled.select(subBand).reduce(ee.Reducer.min()).gt(0)  # Valid reflectance range
    mask4 = scaled.select(subBand).reduce(ee.Reducer.max()).lt(1)
    mask5 = image.select("SR_ATMOS_OPACITY").unmask(-1).lt(300)  # Atmospheric opacity
    
    return image.addBands(scaled).updateMask(
        mask1.And(mask2).And(mask3).And(mask4).And(mask5))


def prepareL8Col2(image):
    """
    Preprocess Landsat 8/9 Collection 2 Level-2 Surface Reflectance.
    
    Applies scale factors, quality masks, and band renaming for standardization.
    
    Args:
        image: ee.Image from Landsat 8/9 Collection 2
        
    Returns:
        Preprocessed ee.Image with standardized band names
    """
    bandList = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10']
    nameList = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP']
    subBand = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
    
    # Apply Collection 2 scale factors
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B10').multiply(0.00341802).add(149.0)
    scaled = opticalBands.addBands(thermalBand, None, True).select(bandList).rename(nameList)
    
    # Quality masks
    validTOA = [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164, 64, 128]
    validQA = [21824, 21888, 21952]  # Clear land pixels
    
    mask1 = image.select(['QA_PIXEL']).remap(validQA, ee.List.repeat(1, len(validQA)), 0)
    mask2 = image.select('QA_RADSAT').eq(0)  # No radiometric saturation
    mask3 = scaled.select(subBand).reduce(ee.Reducer.min()).gt(0)  # Valid reflectance range
    mask4 = scaled.select(subBand).reduce(ee.Reducer.max()).lt(1)
    mask5 = image.select(['SR_QA_AEROSOL']).remap(
        validTOA, ee.List.repeat(1, len(validTOA)), 0)  # Aerosol quality
    
    return image.addBands(scaled).updateMask(
        mask1.And(mask2).And(mask3).And(mask4).And(mask5))


def add_ndvi_band(image):
    """Add NDVI band to image."""
    ndvi = image.normalizedDifference(['NIR', 'RED']).rename('NDVI')
    return image.addBands(ndvi)


# ======================= CCDC HELPER FUNCTIONS =======================

def convert_date_to_fractional_year(date):
    """Convert Earth Engine date to fractional year format."""
    year = ee.Number(date.get('year'))
    fractional = date.difference(ee.Date.fromYMD(year, 1, 1), 'year')
    return year.add(fractional)


def date_to_segment(t, fits):
    """Find which CCDC segment contains time t."""
    t_start = ee.Array(fits.get('tStart'))
    t_end = ee.Array(fits.get('tEnd'))
    mask = t_start.lte(t).And(t_end.gte(t))
    return ee.List(mask.toList()).indexOf(1)


def harmonic_fit(t, coef):
    """Calculate harmonic model value at time t using CCDC coefficients."""
    pi2 = 2.0 * math.pi
    omega = pi2
    return (
        coef.get([0])
        .add(coef.get([1]).multiply(t))
        .add(coef.get([2]).multiply(t.multiply(omega).cos()))
        .add(coef.get([3]).multiply(t.multiply(omega).sin()))
        .add(coef.get([4]).multiply(t.multiply(omega * 2).cos()))
        .add(coef.get([5]).multiply(t.multiply(omega * 2).sin()))
        .add(coef.get([6]).multiply(t.multiply(omega * 3).cos()))
        .add(coef.get([7]).multiply(t.multiply(omega * 3).sin()))
    )


def fractional_year_to_datetime(value):
    """Convert fractional year to Python datetime object."""
    if value is None:
        return None
    year = int(math.floor(value))
    frac = value - year
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return start + (end - start) * frac


# ======================= MAIN ANALYSIS FUNCTION =======================

def run_ccdc_analysis(lat, lon, start_year, end_year, target_year=None, 
                      point_id=None, biome=None, use_false_color=False,
                      buffer_distance=1500, cloud_threshold=30,
                      ccdc_lambda=20/10000.0, ccdc_max_iter=10000,
                      min_observations=6, chi_square_prob=0.99,
                      verbose=True,
                      imagery_combo='RGB',            # 'RGB' | 'CIR' | 'SWIR'
                      season_months=None,             # e.g., (6, 10) for Jun–Oct
                      composite_method='median',      # 'median' | 'medoid'
                      vis_percentiles=None,           # e.g., (2, 98); None -> fixed min/max
                      thumb_dimensions=100):
    """
    Run CCDC analysis on Landsat NDVI time series and visualize results.
    
    This function performs the complete CCDC workflow:
    1. Prepares Landsat collections (5, 7, 8, 9) with quality masking
    2. Runs CCDC temporal segmentation algorithm
    3. Extracts harmonic model coefficients and fit values
    4. Visualizes CCDC segments and Landsat imagery
    
    Args:
        lat (float): Latitude of the point location
        lon (float): Longitude of the point location
        start_year (int): Start year for analysis (e.g., 2000)
        end_year (int): End year for analysis (e.g., 2022)
        target_year (int, optional): Year to highlight in visualization
        point_id (int/str, optional): Point identifier for labeling
        biome (str, optional): Biome name for labeling
        use_false_color (bool): If True, use false color (NIR-R-G); if False, use true color (RGB)
        buffer_distance (float): Buffer distance in meters for imagery ROI (default: 1500)
        cloud_threshold (float): Maximum cloud cover percentage for imagery (default: 30)
        ccdc_lambda (float): CCDC regularization parameter (default: 0.002)
        ccdc_max_iter (int): Maximum CCDC iterations (default: 10000)
        min_observations (int): Minimum observations per segment (default: 6)
        chi_square_prob (float): Chi-square probability threshold (default: 0.99)
        verbose (bool): If True, print progress messages
        
    Returns:
        dict: Dictionary containing:
            - 'df_ccdc': DataFrame with NDVI time series and fit values
            - 'segments': List of tuples (segment_index, start_date, end_date)
            - 'fits': Earth Engine dictionary with CCDC coefficients
            - 'fig': Matplotlib figure object
            
    Example:
        >>> import ee
        >>> ee.Initialize()
        >>> result = run_ccdc_analysis(
        ...     lat=-10.5, lon=-50.2,
        ...     start_year=2000, end_year=2022,
        ...     target_year=2010, point_id=12345,
        ...     use_false_color=True
        ... )
        >>> result['fig'].savefig('ccdc_analysis.png', dpi=300, bbox_inches='tight')
    """
    
    # CCDC band configuration
    BREAKPOINT_BANDS = ['GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
    TMASK_BANDS = ['GREEN', 'SWIR2']
    CCDC_DATE_FORMAT = 1  # fractional years
    
    if verbose:
        print(f"{'='*80}")
        print(f"CCDC Analysis + Landsat Imagery")
        if point_id:
            print(f"Point ID: {point_id}")
        print(f"Location: Lat {lat:.4f}, Lon {lon:.4f}")
        if biome:
            print(f"Biome: {biome}")
        print(f"Time range: {start_year}-{end_year}")
        print(f"{'='*80}\n")
    
    # Create Earth Engine point geometry
    ee_point = ee.Geometry.Point([lon, lat])
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    
    # Prepare Landsat collections
    if verbose:
        print(f"Preparing Landsat collections ({start_year}-{end_year})...")
    
    standard_bands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'NDVI']
    
    l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
        .filterDate(start_date, end_date) \
        .filterBounds(ee_point) \
        .map(prepareL4L5L7Col2) \
        .map(add_ndvi_band) \
        .select(standard_bands)
    
    l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
        .filterDate(start_date, end_date) \
        .filterBounds(ee_point) \
        .map(prepareL4L5L7Col2) \
        .map(add_ndvi_band) \
        .select(standard_bands)
    
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterDate(start_date, end_date) \
        .filterBounds(ee_point) \
        .map(prepareL8Col2) \
        .map(add_ndvi_band) \
        .select(standard_bands)
    
    collection = l5.merge(l7).merge(l8)
    n_imgs = collection.size().getInfo()
    
    if verbose:
        print(f"Total images: {n_imgs}")
    
    if n_imgs == 0:
        print("ERROR: No images available for analysis")
        return None
    
    # Run CCDC algorithm
    if verbose:
        print("Running CCDC algorithm...")
    
    ccdc_params = {
        'collection': collection,
        'breakpointBands': BREAKPOINT_BANDS,
        'tmaskBands': TMASK_BANDS,
        'dateFormat': CCDC_DATE_FORMAT,
        'lambda': ccdc_lambda,
        'maxIterations': ccdc_max_iter,
        'minObservations': min_observations,
        'chiSquareProbability': chi_square_prob,
        'minNumOfYearsScaler': 1.33
    }
    ccdc_result = ee.Algorithms.TemporalSegmentation.Ccdc(**ccdc_params)
    
    projection = ee.Projection('EPSG:4326').atScale(30)
    fits = ee.Dictionary(ccdc_result.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=ee_point,
        crs=projection
    ))
    
    # Augment collection with fit values
    def augment_with_fit(img):
        img = ee.Image(img)
        time = convert_date_to_fractional_year(img.date())
        segment = date_to_segment(time, fits)
        value = img.select('NDVI').reduceRegion(
            ee.Reducer.first(), geometry=ee_point, crs=projection).get('NDVI')
        coef = ee.Algorithms.If(
            ee.Number(segment).add(1),
            ee.Array(fits.getArray('NDVI_coefs')).slice(0, segment, ee.Number(segment).add(1)).project([1]),
            ee.Array([0, 0, 0, 0, 0, 0, 0, 0])
        )
        fit = harmonic_fit(time, ee.Array(coef))
        return img.set({
            'value': value, 'fitTime': time, 'fit': fit,
            'segment': segment, 'dateString': img.date().format('YYYY-MM-dd')
        })
    
    series = collection.sort('system:time_start').map(augment_with_fit)
    
    if verbose:
        print("Extracting time series data...")
    
    reducer = ee.Reducer.toList(4, 4)
    table = ee.List(series.reduceColumns(reducer, ['dateString', 'value', 'fit', 'segment']).get('list')).getInfo()
    
    df_ccdc = pd.DataFrame(table, columns=['date', 'ndvi', 'fit', 'segment'])
    df_ccdc['date'] = pd.to_datetime(df_ccdc['date'])
    df_ccdc = df_ccdc.sort_values('date')
    
    # Extract segments
    t_start = ee.Array(fits.get('tStart')).toList().getInfo()
    t_end = ee.Array(fits.get('tEnd')).toList().getInfo()
    segments = []
    for i, (start_val, end_val) in enumerate(zip(t_start, t_end)):
        if start_val is not None and end_val is not None:
            start_dt = fractional_year_to_datetime(start_val)
            end_dt = fractional_year_to_datetime(end_val)
            if start_dt and end_dt and end_dt > start_dt:
                segments.append((i, start_dt, end_dt))
    
    if verbose:
        print(f"Found {len(segments)} CCDC segments\n")
    
    # ==================== FETCH GLANCE DATA ====================
    glance_data = {}
    if verbose:
        print("Fetching GLANCE land cover data...")
    
    try:
        GLANCE_IC = ee.ImageCollection('projects/GLANCE/DATASETS/V001')
        glance_class_names = {
            1: 'Water', 2: 'Snow/Ice', 3: 'Developed', 4: 'Barren', 
            5: 'Forest', 6: 'Shrubland', 7: 'Herbaceous'
        }
        
        glance_years = []
        glance_classes = []
        
        for year in range(start_year, end_year + 1):
            try:
                img = GLANCE_IC.filterDate(f'{year}-01-01', f'{year}-12-31').select('LC').mosaic()
                value = img.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=ee_point,
                    scale=30
                ).getInfo()
                lc_value = value.get('LC', None)
                
                if lc_value is not None:
                    glance_years.append(year)
                    glance_classes.append(glance_class_names.get(lc_value, 'Unknown'))
            except Exception as e:
                if verbose:
                    print(f"  Warning: GLANCE {year} - {e}")
        
        glance_data = {'years': glance_years, 'classes': glance_classes}
        if verbose:
            print(f"  Retrieved {len(glance_years)} years of GLANCE data")
    except Exception as e:
        if verbose:
            print(f"  Error fetching GLANCE data: {e}")
        glance_data = {'years': [], 'classes': []}
    
    # ==================== CREATE VISUALIZATION ====================
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.2, 0.6, 1], hspace=0.35, wspace=0.25)
    
    # CCDC Plot (top row, spanning all columns)
    ax = fig.add_subplot(gs[0, :])
    
    # Plot observed NDVI
    ax.scatter(df_ccdc['date'], df_ccdc['ndvi'], s=15, color='darkgreen', 
               alpha=0.5, label='Observed NDVI', zorder=2)
    
    # Plot harmonic curves for each segment
    segment_colors_line = ['#0066cc', '#cc0066', '#00cc66', '#cc6600', '#6600cc', '#cc9900']
    segment_colors_bg = ['#e6f2ff', '#ffe6f2', '#e6ffe6', '#fff2e6', '#f2e6ff', '#fff9e6']
    
    for seg_idx, start_dt, end_dt in segments:
        # Background color
        color_bg = segment_colors_bg[seg_idx % len(segment_colors_bg)]
        ax.axvspan(start_dt, end_dt, alpha=0.2, color=color_bg, zorder=1)
        
        # Generate dense time points for smooth curve (200 points per segment)
        n_points = 200
        time_range = pd.date_range(start=start_dt, end=end_dt, periods=n_points)
        
        # Convert to fractional years
        fractional_times = []
        for t in time_range:
            year = t.year
            year_start = datetime(year, 1, 1)
            year_end = datetime(year + 1, 1, 1)
            frac = (t - year_start) / (year_end - year_start)
            fractional_times.append(year + frac)
        
        # Get coefficients and calculate harmonic fit
        try:
            coefs_array = ee.Array(fits.getArray('NDVI_coefs')).slice(0, seg_idx, seg_idx + 1).project([1]).getInfo()
            pi2 = 2.0 * math.pi
            omega = pi2
            
            fit_values = []
            for t in fractional_times:
                fit_val = (
                    coefs_array[0] + coefs_array[1] * t +
                    coefs_array[2] * np.cos(omega * t) +
                    coefs_array[3] * np.sin(omega * t) +
                    coefs_array[4] * np.cos(2 * omega * t) +
                    coefs_array[5] * np.sin(2 * omega * t) +
                    coefs_array[6] * np.cos(3 * omega * t) +
                    coefs_array[7] * np.sin(3 * omega * t)
                )
                fit_values.append(fit_val)
            
            color = segment_colors_line[seg_idx % len(segment_colors_line)]
            ax.plot(time_range, fit_values, color=color, linewidth=2.5, 
                   label=f'Segment {seg_idx + 1}' if seg_idx < 6 else None, zorder=3)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not plot segment {seg_idx + 1}: {e}")
    
    # Highlight target year if provided
    if target_year:
        target_start_dt = datetime(target_year, 1, 1)
        target_end_dt = datetime(target_year, 12, 31)
        ax.axvspan(target_start_dt, target_end_dt, alpha=0.15, color='orange', 
                  edgecolor='red', linewidth=2, linestyle='--',
                  label=f'Target Year {target_year}', zorder=4)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('NDVI', fontsize=12, fontweight='bold')
    
    title = f'CCDC NDVI Time Series ({start_year}-{end_year})\n'
    title += f'Location: ({lat:.4f}, {lon:.4f})'
    if biome:
        title += f', Biome: {biome}'
    if point_id:
        title = f'Point {point_id} - ' + title
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-0.1, 1.0])
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ==================== GLANCE LAND COVER PLOT (middle row) ====================
    ax_glance = fig.add_subplot(gs[1, :])
    
    if len(glance_data['years']) > 0:
        # Convert years to datetime for consistency with CCDC plot
        glance_dates = [datetime(year, 7, 1) for year in glance_data['years']]  # Mid-year dates
        
        # Create numeric mapping for plotting
        unique_classes = sorted(set(glance_data['classes']))
        class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
        glance_numeric = [class_to_num[cls] for cls in glance_data['classes']]
        
        # Define colors for each land cover class
        glance_colors = {
            'Water': '#0066cc', 'Snow/Ice': '#ccffff', 'Developed': '#cc0066',
            'Barren': '#cc9966', 'Forest': '#006633', 'Shrubland': '#ffcc66',
            'Herbaceous': '#99cc66', 'Unknown': '#999999', 'No Data': '#cccccc'
        }
        
        # Plot with color-coded markers
        for i, (date, cls) in enumerate(zip(glance_dates, glance_data['classes'])):
            color = glance_colors.get(cls, '#999999')
            ax_glance.plot(date, glance_numeric[i], 'o', color=color, 
                          markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        
        # Connect points with line
        ax_glance.plot(glance_dates, glance_numeric, '-', color='gray', 
                      linewidth=1, alpha=0.5, zorder=1)
        
        ax_glance.set_yticks(range(len(unique_classes)))
        ax_glance.set_yticklabels(unique_classes, fontsize=10)
        ax_glance.set_ylabel('GLANCE Land Cover', fontsize=11, fontweight='bold')
        ax_glance.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax_glance.set_title('GLANCE Land Cover Classification Time Series', 
                           fontsize=12, fontweight='bold')
        ax_glance.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax_glance.set_ylim(-0.5, len(unique_classes) - 0.5)
        
        # Highlight target year if provided
        if target_year:
            target_start_dt = datetime(target_year, 1, 1)
            target_end_dt = datetime(target_year, 12, 31)
            ax_glance.axvspan(target_start_dt, target_end_dt, alpha=0.15, color='orange', 
                            edgecolor='red', linewidth=2, linestyle='--', zorder=0)
        
        ax_glance.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_glance.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax_glance.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax_glance.text(0.5, 0.5, 'GLANCE data not available', 
                      ha='center', va='center', fontsize=12, 
                      transform=ax_glance.transAxes)
        ax_glance.set_title('GLANCE Land Cover Classification Time Series', 
                           fontsize=12, fontweight='bold')
        ax_glance.axis('off')
    
    # Landsat imagery (bottom row)
    if target_year:
        # Backward compatibility: if use_false_color is True and imagery_combo wasn't set to something else
        # switch to CIR (NIR-Red-Green). Otherwise honor imagery_combo.
        if use_false_color and imagery_combo == 'RGB':
            imagery_combo = 'CIR'
        if verbose:
            print(f"Fetching Landsat thumbnails ({imagery_combo})...")

        roi = ee_point.buffer(buffer_distance).bounds()
        imagery_years = [target_year - 2, target_year - 1, target_year, target_year + 1, target_year + 2]

        # Helper: choose collection and bands by sensor/imagery_combo
        def year_collection(year):
            if year < 2013:
                base = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').merge(
                       ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'))
                rgb  = ['SR_B3','SR_B2','SR_B1']          # R,G,B
                cir  = ['SR_B4','SR_B3','SR_B2']          # NIR,R,G
                swir = ['SR_B5','SR_B4','SR_B3']          # SWIR1,NIR,R
            else:
                base = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').merge(
                       ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
                rgb  = ['SR_B4','SR_B3','SR_B2']          # R,G,B
                cir  = ['SR_B5','SR_B4','SR_B3']          # NIR,R,G
                swir = ['SR_B6','SR_B5','SR_B4']          # SWIR1,NIR,R
            if imagery_combo == 'CIR':
                bands = cir
            elif imagery_combo == 'SWIR':
                bands = swir
            else:
                bands = rgb
            ic = (base
                  .filterDate(f'{year}-01-01', f'{year}-12-31')
                  .filterBounds(ee_point)
                  .filter(ee.Filter.lt('CLOUD_COVER', cloud_threshold)))
            # Optional seasonal filter
            if season_months and len(season_months) == 2:
                ic = ic.filter(ee.Filter.calendarRange(season_months[0], season_months[1], 'month'))
            return ic, bands

        # Helper: medoid composite on scaled bands
        def medoid_composite(ic, bands):
            median = ic.select(bands).median()
            def add_dist(img):
                d = (img.select(bands).subtract(median).pow(2)
                     .reduce(ee.Reducer.sum()).sqrt().rename('dist'))
                return ee.Image(img).addBands(d)
            with_d = ic.map(add_dist)
            return ee.Image(with_d.sort('dist').first()).select(bands)

        # Helper: percentile-based stretch (client-side values)
        def viz_params(img, roi_geom, bands):
            if not vis_percentiles:
                return None
            p_low, p_high = vis_percentiles
            stats = img.select(bands).reduceRegion(
                reducer=ee.Reducer.percentile([p_low, p_high]),
                geometry=roi_geom, scale=30, maxPixels=1e8)
            stats_info = stats.getInfo() or {}
            mins = []
            maxs = []
            for b in bands:
                mins.append(stats_info.get(f"{b}_p{p_low}", 0))
                maxs.append(stats_info.get(f"{b}_p{p_high}", 0.4))
            return {'min': mins, 'max': maxs}

        for idx, year in enumerate(imagery_years):
            ax_img = fig.add_subplot(gs[2, idx])
            try:
                ic, bands = year_collection(year)
                n_imgs = ic.size().getInfo()
                if n_imgs > 0:
                    # Scale collection to reflectance for the chosen bands
                    # Apply scale factor to the selected bands, then select them
                    def scale_img(img):
                        return img.select(bands).multiply(0.0000275).add(-0.2)
                    ic_scaled = ic.map(scale_img)
                    
                    if composite_method == 'medoid':
                        img = medoid_composite(ic_scaled, bands)
                    else:
                        img = ic_scaled.median()

                    params = viz_params(img, roi, bands)
                    if params is None:
                        # Fallback fixed stretch similar to previous behavior
                        # Slightly higher max for CIR/SWIR to avoid saturation
                        vis_max = 0.4 if imagery_combo in ('CIR','SWIR') else 0.3
                        params = {'min': 0, 'max': vis_max}

                    url = img.getThumbURL({
                        'region': roi.getInfo()['coordinates'],
                        'dimensions': thumb_dimensions,
                        'format': 'png',
                        'min': params['min'],
                        'max': params['max']
                    })

                    response = requests.get(url)
                    img_data = Image.open(BytesIO(response.content))
                    ax_img.imshow(img_data)

                    # Add center point marker (fix width/height usage)
                    width, height = img_data.size
                    center_x, center_y = width / 2, height / 2
                    ax_img.plot(center_x, center_y, 'ro', markersize=8,
                                markeredgecolor='white', markeredgewidth=1.5, zorder=10)

                    if year == target_year:
                        ax_img.set_title(f'{year} ★\n({n_imgs} images)',
                                         fontsize=12, fontweight='bold', color='red')
                        for spine in ax_img.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(3)
                    else:
                        ax_img.set_title(f'{year}\n({n_imgs} images)', fontsize=11, fontweight='bold')
                    ax_img.axis('off')
                    if verbose:
                        print(f"  {year}: ✓ ({n_imgs} images)")
                else:
                    ax_img.text(0.5, 0.5, f'No imagery\n({year})',
                                 ha='center', va='center', fontsize=10, transform=ax_img.transAxes)
                    ax_img.set_title(f'{year}', fontsize=11, fontweight='bold')
                    ax_img.axis('off')
                    if verbose:
                        print(f"  {year}: No imagery")
            except Exception as e:
                ax_img.text(0.5, 0.5, f'Error\n{year}',
                             ha='center', va='center', fontsize=10, transform=ax_img.transAxes)
                ax_img.set_title(f'{year}', fontsize=11, fontweight='bold')
                ax_img.axis('off')
                if verbose:
                    print(f"  {year}: Error - {e}")
    
    suptitle = 'CCDC Analysis & Landsat Imagery'
    if point_id:
        suptitle += f' - Point {point_id}'
    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    
    # Print statistics
    if verbose:
        print(f"\n{'='*80}")
        print("CCDC Segment Statistics:")
        print(f"{'='*80}")
        for idx, start_dt, end_dt in segments:
            duration_days = (end_dt - start_dt).days
            segment_data = df_ccdc[(df_ccdc['date'] >= start_dt) & (df_ccdc['date'] <= end_dt)]
            if len(segment_data) > 0:
                mean_ndvi = segment_data['ndvi'].mean()
                rmse = np.sqrt(((segment_data['ndvi'] - segment_data['fit']) ** 2).mean())
                print(f"Segment {idx + 1}: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} "
                      f"({duration_days} days, {len(segment_data)} obs, mean NDVI: {mean_ndvi:.3f}, RMSE: {rmse:.3f})")
        
        if target_year:
            target_data = df_ccdc[df_ccdc['date'].dt.year == target_year]
            if len(target_data) > 0:
                print(f"\nTarget Year {target_year}:")
                print(f"  Observations: {len(target_data)}")
                print(f"  Mean NDVI (observed): {target_data['ndvi'].mean():.3f}")
                print(f"  Mean NDVI (fitted): {target_data['fit'].mean():.3f}")
                print(f"  RMSE: {np.sqrt(((target_data['ndvi'] - target_data['fit']) ** 2).mean()):.3f}")
        print(f"{'='*80}")
    
    # Return results
    return {
        'df_ccdc': df_ccdc,
        'segments': segments,
        'fits': fits,
        'glance_data': glance_data,
        'fig': fig
    }


# ======================= CONVENIENCE FUNCTIONS =======================

def plot_ccdc_from_dataframe(df, point_id, lat, lon, biome=None, figsize=(16, 6)):
    """
    Create a simplified CCDC plot from a pre-computed DataFrame.
    
    Args:
        df: DataFrame with columns ['date', 'ndvi', 'fit', 'segment']
        point_id: Point identifier for labeling
        lat: Latitude
        lon: Longitude
        biome: Biome name (optional)
        figsize: Figure size tuple (default: (16, 6))
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot observed NDVI
    ax.scatter(df['date'], df['ndvi'], s=15, color='darkgreen', 
               alpha=0.5, label='Observed NDVI', zorder=2)
    
    # Plot fitted NDVI
    ax.plot(df['date'], df['fit'], color='red', linewidth=2, 
            label='CCDC Fit', alpha=0.7, zorder=3)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('NDVI', fontsize=12, fontweight='bold')
    
    title = f'CCDC NDVI Time Series - Point {point_id}\n'
    title += f'Location: ({lat:.4f}, {lon:.4f})'
    if biome:
        title += f', Biome: {biome}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-0.1, 1.0])
    ax.legend(loc='best', fontsize=10)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def export_ccdc_results(result, output_path, point_id):
    """
    Export CCDC analysis results to CSV and PNG files.
    
    Args:
        result: Dictionary returned from run_ccdc_analysis()
        output_path: Directory path for output files
        point_id: Point identifier for file naming
    """
    import os
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save DataFrame
    csv_file = os.path.join(output_path, f'ccdc_point_{point_id}.csv')
    result['df_ccdc'].to_csv(csv_file, index=False)
    print(f"Saved time series data to: {csv_file}")
    
    # Save figure
    png_file = os.path.join(output_path, f'ccdc_point_{point_id}.png')
    result['fig'].savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {png_file}")
    
    # Save segment statistics
    stats_file = os.path.join(output_path, f'ccdc_segments_{point_id}.txt')
    with open(stats_file, 'w') as f:
        f.write(f"CCDC Segment Statistics - Point {point_id}\n")
        f.write(f"{'='*80}\n")
        for idx, start_dt, end_dt in result['segments']:
            duration_days = (end_dt - start_dt).days
            df = result['df_ccdc']
            segment_data = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            if len(segment_data) > 0:
                mean_ndvi = segment_data['ndvi'].mean()
                rmse = np.sqrt(((segment_data['ndvi'] - segment_data['fit']) ** 2).mean())
                f.write(f"Segment {idx + 1}: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} "
                       f"({duration_days} days, {len(segment_data)} obs, mean NDVI: {mean_ndvi:.3f}, RMSE: {rmse:.3f})\n")
    print(f"Saved segment statistics to: {stats_file}")
