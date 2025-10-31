"""
Forest Deforestation Rate Analysis and Visualization by Sector and Protection Status

Purpose: 
- Load and combine exported forest analysis CSV files
- Match with sector information using BU ID
- Analyze annual deforestation rates by sector and protection status
- Create visualizations comparing protected vs non-protected areas across sectors

Author: Forest Analysis Team
Date: 2025-01-XX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_forest_analysis_data(data_folder_path):
    """
    Load and combine all forest analysis CSV files
    
    Args:
        data_folder_path: Path to folder containing CSV files
    
    Returns:
        Combined DataFrame with all forest analysis results
    """
    
    print("Loading forest analysis data...")
    
    # Find all CSV files
    protected_files = glob.glob(f"{data_folder_path}/forest_PROTECTED_*km_2001_2024.csv")
    non_protected_files = glob.glob(f"{data_folder_path}/forest_NON_PROTECTED_*km_2001_2024.csv")
    
    print(f"Found {len(protected_files)} protected files")
    print(f"Found {len(non_protected_files)} non-protected files")
    
    all_dfs = []
    
    # Load protected files
    for file_path in protected_files:
        df = pd.read_csv(file_path)
        buffer_km = Path(file_path).stem.split('_')[3].replace('km', '')
        df['buffer_distance_km'] = int(buffer_km)
        all_dfs.append(df)
        print(f"Loaded: {Path(file_path).name} - {len(df)} records")
    
    # Load non-protected files
    for file_path in non_protected_files:
        df = pd.read_csv(file_path)
        buffer_km = Path(file_path).stem.split('_')[4].replace('km', '')
        df['buffer_distance_km'] = int(buffer_km)
        all_dfs.append(df)
        print(f"Loaded: {Path(file_path).name} - {len(df)} records")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\nCombined dataset: {len(combined_df)} records")
    print(f"Unique BU_IDs: {combined_df['BU_ID'].nunique()}")
    print(f"Buffer distances: {sorted(combined_df['buffer_distance_km'].unique())}")
    print(f"Protection statuses: {combined_df['protection_status'].unique()}")
    
    return combined_df

def load_sector_data(shapefile_path):
    """
    Load sector information from shapefile
    
    Args:
        shapefile_path: Path to shapefile with sector information
    
    Returns:
        GeoDataFrame with BU ID, Sector, and Loan Sign Year columns
    """
    
    print(f"\nLoading sector data from: {shapefile_path}")
    
    # Load the specific layer
    gdf_sector = gpd.read_file(shapefile_path, layer='Dataset-2025-EN')
    
    print(f"Sector data loaded: {len(gdf_sector)} records")
    print(f"Unique sectors: {gdf_sector['Sector'].nunique()}")
    print(f"Sector distribution:\n{gdf_sector['Sector'].value_counts()}")
    
    # Show loan year distribution
    print(f"\nLoan Sign Year statistics:")
    print(f"  Range: {gdf_sector['Loan Sign Year'].min():.0f}-{gdf_sector['Loan Sign Year'].max():.0f}")
    print(f"  Mean: {gdf_sector['Loan Sign Year'].mean():.1f}")
    print(f"  Missing values: {gdf_sector['Loan Sign Year'].isna().sum()}")
    
    # Keep only necessary columns
    sector_df = gdf_sector[['BU ID', 'Sector', 'Loan Sign Year']].copy()
    sector_df.columns = ['BU_ID', 'Sector', 'Loan_Sign_Year']  # Standardize column names
    
    return sector_df

def combine_forest_sector_data(forest_df, sector_df):
    """
    Combine forest analysis data with sector information
    
    Args:
        forest_df: Forest analysis DataFrame
        sector_df: Sector information DataFrame
    
    Returns:
        Combined DataFrame with sector and loan sign year information
    """
    
    print("\nCombining forest and sector data...")
    
    # Merge on BU_ID
    combined_df = forest_df.merge(sector_df, on='BU_ID', how='left')
    
    # Check merge success
    matched_records = combined_df['Sector'].notna().sum()
    total_records = len(combined_df)
    
    print(f"Merge results:")
    print(f"  Total forest records: {total_records}")
    print(f"  Successfully matched: {matched_records} ({matched_records/total_records*100:.1f}%)")
    print(f"  Unmatched records: {total_records - matched_records}")
    
    # Show sector distribution in matched data
    sector_counts = combined_df['Sector'].value_counts()
    print(f"\nSector distribution in combined data:")
    print(sector_counts)
    
    # Show loan sign year distribution
    loan_year_stats = combined_df['Loan_Sign_Year'].describe()
    print(f"\nLoan Sign Year distribution in combined data:")
    print(loan_year_stats)
    
    return combined_df

# ============================================================================
# DATA PROCESSING FOR ANNUAL ANALYSIS
# ============================================================================

def reshape_annual_data(df):
    """
    Reshape data to have annual deforestation in long format
    
    Args:
        df: Combined forest-sector DataFrame
    
    Returns:
        DataFrame in long format with annual deforestation data
    """
    
    print("\nReshaping data for annual analysis...")
    
    # Identify annual deforestation columns (defor_YYYY_ha)
    annual_cols = [col for col in df.columns if col.startswith('defor_') and col.endswith('_ha')]
    annual_cols = sorted(annual_cols)  # Sort by year
    
    print(f"Found {len(annual_cols)} annual deforestation columns: {annual_cols[:3]}...{annual_cols[-3:]}")
    
    # Create ID columns - now including Loan_Sign_Year
    id_cols = ['BU_ID', 'buffer_distance_km', 'protection_status', 'Sector', 
               'Loan_Sign_Year', 'initial_forest_2000_ha']
    
    # Melt the dataframe
    df_long = df[id_cols + annual_cols].melt(
        id_vars=id_cols,
        value_vars=annual_cols,
        var_name='year_col',
        value_name='deforestation_ha'
    )
    
    # Extract year from column name
    df_long['year'] = df_long['year_col'].str.extract('(\\d{4})').astype(int)
    
    # Calculate annual deforestation rate (% of initial forest)
    df_long['annual_defor_rate_pct'] = np.where(
        df_long['initial_forest_2000_ha'] > 0,
        (df_long['deforestation_ha'] / df_long['initial_forest_2000_ha']) * 100,
        0
    )
    
    # Remove unnecessary column
    df_long = df_long.drop('year_col', axis=1)
    
    print(f"Reshaped data: {len(df_long)} records")
    print(f"Year range: {df_long['year'].min()}-{df_long['year'].max()}")
    
    return df_long

def add_loan_period_classification(df_long, fixed_years=None):
    """
    Add classification for pre-loan and post-loan periods
    
    Args:
        df_long: Long format DataFrame with annual data
        fixed_years: If specified, use fixed number of years before/after loan.
                    If None, use all available years.
    
    Returns:
        DataFrame with loan period classification added
    """
    
    print(f"\nClassifying data into pre-loan and post-loan periods...")
    if fixed_years is not None:
        print(f"Using fixed {fixed_years} years before and after loan sign year")
    else:
        print("Using all available years before and after loan sign year")
    
    # Create a copy to avoid modifying original
    df_classified = df_long.copy()
    
    if fixed_years is not None:
        # Use fixed number of years before and after loan
        df_classified['loan_period'] = np.where(
            (df_classified['year'] >= df_classified['Loan_Sign_Year'] - fixed_years) & 
            (df_classified['year'] < df_classified['Loan_Sign_Year']), 
            'pre_loan',
            np.where(
                (df_classified['year'] > df_classified['Loan_Sign_Year']) & 
                (df_classified['year'] <= df_classified['Loan_Sign_Year'] + fixed_years), 
                'post_loan',
                'outside_range'
            )
        )
    else:
        # Use all available years
        df_classified['loan_period'] = np.where(
            df_classified['year'] < df_classified['Loan_Sign_Year'], 
            'pre_loan',
            'post_loan'
        )
    
    # Add years relative to loan sign year
    df_classified['years_from_loan'] = df_classified['year'] - df_classified['Loan_Sign_Year']
    
    # Filter to only include data that has at least some pre and post loan years
    # (i.e., loan sign year should be within the data range)
    valid_loan_years = df_classified[
        (df_classified['Loan_Sign_Year'] >= df_classified['year'].min()) & 
        (df_classified['Loan_Sign_Year'] <= df_classified['year'].max())
    ]['Loan_Sign_Year'].unique()
    
    df_classified = df_classified[df_classified['Loan_Sign_Year'].isin(valid_loan_years)]
    
    # Filter out data outside the range if using fixed years
    if fixed_years is not None:
        df_classified = df_classified[df_classified['loan_period'] != 'outside_range']
    
    print(f"Valid loan years (within data range): {len(valid_loan_years)}")
    print(f"Records after filtering: {len(df_classified)}")
    
    # Show distribution
    period_dist = df_classified['loan_period'].value_counts()
    print(f"Loan period distribution:")
    print(period_dist)
    
    return df_classified

# ============================================================================
# BEFORE/AFTER LOAN ANALYSIS FUNCTIONS
# ============================================================================

def calculate_pre_post_loan_averages(df_classified, buffer_km=10, fixed_years=None):
    """
    Calculate average annual deforestation rates before and after loan sign year
    
    Args:
        df_classified: DataFrame with loan period classification
        buffer_km: Buffer distance to analyze
        fixed_years: If None, use all years; if int, use fixed window around loan year
    
    Returns:
        DataFrame with pre/post loan averages by protection status and sector
    """
    
    year_window_text = "All Years" if fixed_years is None else f"Fixed {fixed_years}-Year Window"
    print(f"\nCalculating pre/post loan averages for {buffer_km}km buffer ({year_window_text})...")
    
    # Filter data for specific buffer distance
    df_analysis = df_classified[df_classified['buffer_distance_km'] == buffer_km].copy()
    
    if df_analysis.empty:
        print(f"No data found for {buffer_km}km buffer")
        return pd.DataFrame()
    
    # Calculate averages by BU_ID, protection status, and loan period
    site_averages = df_analysis.groupby([
        'BU_ID', 'Sector', 'protection_status', 'loan_period', 'Loan_Sign_Year'
    ])['annual_defor_rate_pct'].mean().reset_index()
    
    # Pivot to get pre and post loan rates side by side
    pivot_data = site_averages.pivot_table(
        index=['BU_ID', 'Sector', 'protection_status', 'Loan_Sign_Year'],
        columns='loan_period',
        values='annual_defor_rate_pct',
        fill_value=np.nan
    ).reset_index()
    
    # Flatten column names
    pivot_data.columns.name = None
    
    # Calculate difference (post - pre)
    if 'pre_loan' in pivot_data.columns and 'post_loan' in pivot_data.columns:
        pivot_data['loan_effect'] = pivot_data['post_loan'] - pivot_data['pre_loan']
        pivot_data['loan_effect_pct'] = np.where(
            pivot_data['pre_loan'] > 0,
            (pivot_data['loan_effect'] / pivot_data['pre_loan']) * 100,
            np.nan
        )
    
    # Remove rows where both pre and post are missing
    if 'pre_loan' in pivot_data.columns and 'post_loan' in pivot_data.columns:
        pivot_data = pivot_data.dropna(subset=['pre_loan', 'post_loan'], how='all')
    
    print(f"Sites with valid pre/post loan data: {len(pivot_data)}")
    
    # Show summary by protection status
    if not pivot_data.empty and 'pre_loan' in pivot_data.columns and 'post_loan' in pivot_data.columns:
        protection_summary = pivot_data.groupby('protection_status')[
            ['pre_loan', 'post_loan', 'loan_effect']
        ].agg(['count', 'mean', 'std']).round(4)
        
        print(f"\nSummary by protection status:")
        print(protection_summary)
    
    return pivot_data

def calculate_sector_protection_summary(df_long):
    """
    Calculate summary statistics by sector and protection status
    
    Args:
        df_long: Long format DataFrame with annual data
    
    Returns:
        Summary DataFrame
    """
    
    print("\nCalculating sector-protection summary statistics...")
    
    summary = df_long.groupby(['Sector', 'protection_status', 'buffer_distance_km']).agg({
        'BU_ID': 'nunique',
        'initial_forest_2000_ha': 'sum',
        'deforestation_ha': 'sum',
        'annual_defor_rate_pct': 'mean'
    }).reset_index()
    
    summary.columns = ['Sector', 'protection_status', 'buffer_distance_km', 
                      'site_count', 'total_initial_forest_ha', 'total_deforestation_ha', 
                      'mean_annual_defor_rate_pct']
    
    # Calculate overall deforestation rate
    summary['overall_defor_rate_pct'] = np.where(
        summary['total_initial_forest_ha'] > 0,
        (summary['total_deforestation_ha'] / summary['total_initial_forest_ha']) * 100,
        0
    )
    
    return summary

def plot_combined_protection_comparison(pre_post_data, buffer_km=10, save_path=None, fixed_years=None, sectors_filter=None):
    """
    Plot before vs after loan deforestation rates by sector with paired t-tests
    for both protected and non-protected areas
    
    Args:
        pre_post_data: DataFrame with pre/post loan averages
        buffer_km: Buffer distance being analyzed
        save_path: Path to save the plot (optional)
        fixed_years: If None, use all years; if int, use fixed window around loan year
        sectors_filter: List of sectors to include, if None include all
    """
    from scipy import stats
    
    # Remove rows with missing data
    plot_data = pre_post_data.dropna(subset=['pre_loan', 'post_loan']).copy()
    
    if plot_data.empty:
        print("No valid pre/post loan data found")
        return
    
    # Filter sectors if specified
    if sectors_filter is not None:
        plot_data = plot_data[plot_data['Sector'].isin(sectors_filter)].copy()
        if plot_data.empty:
            print(f"No data found for specified sectors: {sectors_filter}")
            return
    
    # Get unique sectors
    sectors = sorted(plot_data['Sector'].dropna().unique())
    n_sectors = len(sectors)
    
    if n_sectors == 0:
        print("No sectors found in data")
        return
    
    # Calculate subplot layout
    n_cols = min(3, n_sectors)  # Max 3 columns
    n_rows = (n_sectors + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Handle single subplot case
    if n_sectors == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Colors for protection status and time periods
    colors = {
        'protected_pre': '#2E8B57',     # Dark green for protected pre-loan
        'protected_post': '#90EE90',    # Light green for protected post-loan
        'non_protected_pre': '#CD853F', # Dark brown for non-protected pre-loan
        'non_protected_post': '#F4A460' # Light brown for non-protected post-loan
    }
    
    # Store t-test results
    ttest_results = []
    
    for i, sector in enumerate(sectors):
        if n_rows == 1:
            ax = axes[i]  # For single row, axes is flattened
        else:
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
        
        # Filter data for this sector
        sector_data = plot_data[plot_data['Sector'] == sector].copy()
        
        if sector_data.empty:
            ax.text(0.5, 0.5, f'No data for\\n{sector}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{sector}')
            continue
        
        # Calculate means for each protection status
        sector_summary = sector_data.groupby('protection_status')[['pre_loan', 'post_loan']].agg(['mean', 'std', 'count'])
        
        # Prepare data for bar plot
        protection_statuses = ['protected', 'non_protected']
        x_positions = np.array([0, 1])  # Positions for protected and non-protected
        bar_width = 0.35
        
        pre_means = []
        post_means = []
        pre_stds = []
        post_stds = []
        
        for status in protection_statuses:
            if status in sector_summary.index:
                pre_means.append(sector_summary.loc[status, ('pre_loan', 'mean')])
                post_means.append(sector_summary.loc[status, ('post_loan', 'mean')])
                pre_stds.append(sector_summary.loc[status, ('pre_loan', 'std')])
                post_stds.append(sector_summary.loc[status, ('post_loan', 'std')])
            else:
                pre_means.append(0)
                post_means.append(0)
                pre_stds.append(0)
                post_stds.append(0)
        
        # Create bar plot
        bars1 = ax.bar(x_positions - bar_width/2, pre_means, bar_width, 
                      yerr=pre_stds, capsize=5,
                      color=[colors['protected_pre'], colors['non_protected_pre']], 
                      alpha=0.8, label='Pre-Loan')
        bars2 = ax.bar(x_positions + bar_width/2, post_means, bar_width,
                      yerr=post_stds, capsize=5,
                      color=[colors['protected_post'], colors['non_protected_post']], 
                      alpha=0.8, label='Post-Loan')
        
        # Add value labels on bars
        for j, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            if height1 > 0:
                ax.text(bar1.get_x() + bar1.get_width()/2., height1 + pre_stds[j]*0.1,
                       f'{height1:.3f}', ha='center', va='bottom', fontsize=9)
            if height2 > 0:
                ax.text(bar2.get_x() + bar2.get_width()/2., height2 + post_stds[j]*0.1,
                       f'{height2:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Perform paired t-tests for each protection status
        ttest_text = []
        
        for j, status in enumerate(protection_statuses):
            status_data = sector_data[sector_data['protection_status'] == status]
            
            if len(status_data) >= 3:  # Need at least 3 samples for meaningful t-test
                pre_values = status_data['pre_loan'].dropna()
                post_values = status_data['post_loan'].dropna()
                
                # Ensure we have paired data
                if len(pre_values) == len(post_values) and len(pre_values) >= 3:
                    # Perform paired t-test (post vs pre)
                    t_stat, p_value = stats.ttest_rel(post_values, pre_values)
                    
                    # Format p-value with significance
                    if p_value < 0.001:
                        p_text = "p<0.001***"
                    elif p_value < 0.01:
                        p_text = f"p={p_value:.3f}**"
                    elif p_value < 0.05:
                        p_text = f"p={p_value:.3f}*"
                    else:
                        p_text = f"p={p_value:.3f}"
                    
                    status_label = status.replace('_', ' ').title()
                    ttest_text.append(f'{status_label}: t={t_stat:.2f}, {p_text}')
                    
                    # Store results
                    ttest_results.append({
                        'Sector': sector,
                        'Protection_Status': status,
                        'n_samples': len(pre_values),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'mean_pre': np.mean(pre_values),
                        'mean_post': np.mean(post_values),
                        'mean_difference': np.mean(post_values) - np.mean(pre_values)
                    })
        
        # Add t-test results to plot
        if ttest_text:
            combined_text = '\\n'.join(ttest_text)
            ax.text(0.02, 0.98, combined_text, 
                   transform=ax.transAxes, ha='left', va='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                   fontsize=8)
        
        # Set labels and formatting
        ax.set_xlabel('Protection Status')
        ax.set_ylabel('Average Deforestation Rate (%)')
        ax.set_title(f'{sector}', fontweight='bold', fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['Protected', 'Non-Protected'])
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    if n_sectors < n_rows * n_cols:
        for i in range(n_sectors, n_rows * n_cols):
            if n_rows == 1:
                axes[i].set_visible(False)  # For flattened array
            else:
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
    
    # Overall title
    year_window_text = "All Years" if fixed_years is None else f"Fixed {fixed_years}-Year Window"
    fig.suptitle(f'Before vs After Loan Deforestation Rates by Sector\\n({buffer_km}km Buffer, {year_window_text})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print detailed statistical results
    print(f"\\n=== Paired T-Test Results (Before vs After Loan, {year_window_text}) ===")
    if ttest_results:
        print(f"{'Sector':<20} {'Protection':<15} {'n':<4} {'t_stat':<8} {'p_value':<10} {'Pre-Loan':<10} {'Post-Loan':<10} {'Difference':<12} {'Sig':<5}")
        print("-" * 110)
        for result in ttest_results:
            significance = ""
            if result['p_value'] < 0.001:
                significance = "***"
            elif result['p_value'] < 0.01:
                significance = "**"
            elif result['p_value'] < 0.05:
                significance = "*"
            
            print(f"{result['Sector']:<20} {result['Protection_Status']:<15} {result['n_samples']:<4} "
                  f"{result['t_statistic']:<8.3f} {result['p_value']:<10.3f} "
                  f"{result['mean_pre']:<10.4f} {result['mean_post']:<10.4f} "
                  f"{result['mean_difference']:<12.4f} {significance:<5}")
    
    # Print summary by protection status across all sectors
    print(f"\\n=== Overall Summary by Protection Status ({year_window_text}) ===")
    for status in ['protected', 'non_protected']:
        status_data = plot_data[plot_data['protection_status'] == status]
        if not status_data.empty:
            print(f"\\n{status.replace('_', ' ').title()} Areas:")
            print(f"  Total sites: {len(status_data)}")
            print(f"  Mean pre-loan rate: {status_data['pre_loan'].mean():.4f}% ± {status_data['pre_loan'].std():.4f}%")
            print(f"  Mean post-loan rate: {status_data['post_loan'].mean():.4f}% ± {status_data['post_loan'].std():.4f}%")
            print(f"  Mean difference (post-pre): {status_data['loan_effect'].mean():.4f}%")
            
            # Overall t-test for this protection status
            if len(status_data) >= 3:
                t_stat, p_value = stats.ttest_rel(status_data['post_loan'], status_data['pre_loan'])
                print(f"  Overall paired t-test: t={t_stat:.3f}, p={p_value:.3f}")
    
    return ttest_results
# ============================================================================
# UPDATED ANALYSIS FUNCTIONS
# ============================================================================

def create_summary_table(df_long, buffer_km=10):
    """
    Create summary statistics table
    
    Args:
        df_long: Long format DataFrame
        buffer_km: Buffer distance to analyze
    
    Returns:
        Summary DataFrame
    """
    
    df_summary = df_long[df_long['buffer_distance_km'] == buffer_km].copy()
    
    summary_stats = df_summary.groupby(['Sector', 'protection_status']).agg({
        'BU_ID': 'nunique',
        'initial_forest_2000_ha': 'sum', 
        'deforestation_ha': 'sum',
        'annual_defor_rate_pct': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    print(f"\\n=== Summary Statistics ({buffer_km}km Buffer) ===")
    print(summary_stats.to_string(index=False))
    
    return summary_stats

def plot_multi_buffer_sector_comparison(df_long, buffer_distances=[5, 10, 15, 20, 25, 50], sectors_filter=['Energy', 'Transportation'], save_path=None):
    """
    Compare Energy and Transportation sectors across different buffer distances
    for before/after loan analysis, separated by protection status
    
    Args:
        df_long: Long format DataFrame with annual data
        buffer_distances: List of buffer distances to compare (in km)
        sectors_filter: List of sectors to analyze (default: Energy and Transportation)
        save_path: Path to save the plot (optional)
    
    Returns:
        DataFrame with comparison results across buffers
    """
    from scipy import stats
    
    print(f"\nAnalyzing {sectors_filter} sectors across {buffer_distances}km buffers...")
    
    # Store results for each buffer
    all_buffer_results = []
    statistical_summary = []
    
    for buffer_km in buffer_distances:
        print(f"\nProcessing {buffer_km}km buffer...")
        
        # Filter data for this buffer distance
        df_buffer = df_long[df_long['buffer_distance_km'] == buffer_km].copy()
        
        if df_buffer.empty:
            print(f"No data found for {buffer_km}km buffer")
            continue
        
        # Add loan period classification for all years
        df_classified = add_loan_period_classification(df_buffer, fixed_years=None)
        
        if df_classified.empty:
            print(f"No classified data for {buffer_km}km buffer")
            continue
        
        # Calculate pre/post loan averages
        pre_post_data = calculate_pre_post_loan_averages(df_classified, buffer_km, fixed_years=None)
        
        if pre_post_data.empty:
            print(f"No pre/post loan data for {buffer_km}km buffer")
            continue
        
        # Filter for specified sectors
        sector_data = pre_post_data[pre_post_data['Sector'].isin(sectors_filter)].copy()
        
        if sector_data.empty:
            print(f"No data for sectors {sectors_filter} in {buffer_km}km buffer")
            continue
        
        # Remove rows with missing data
        sector_data = sector_data.dropna(subset=['pre_loan', 'post_loan'])
        
        # Add buffer distance for tracking
        sector_data['buffer_distance_km'] = buffer_km
        
        # Store results
        all_buffer_results.append(sector_data)
        
        # Perform statistical tests for each sector and protection status
        for sector in sectors_filter:
            for protection in ['protected', 'non_protected']:
                subset = sector_data[
                    (sector_data['Sector'] == sector) & 
                    (sector_data['protection_status'] == protection)
                ]
                
                if len(subset) >= 3:  # Need at least 3 samples
                    try:
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(subset['post_loan'], subset['pre_loan'])
                        
                        # Calculate effect size (Cohen's d for paired samples)
                        differences = subset['post_loan'] - subset['pre_loan']
                        cohens_d = differences.mean() / differences.std() if differences.std() > 0 else 0
                        
                        # Determine significance level
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = "ns"
                        
                        statistical_summary.append({
                            'buffer_km': buffer_km,
                            'sector': sector,
                            'protection_status': protection,
                            'n_sites': len(subset),
                            'pre_loan_mean': subset['pre_loan'].mean(),
                            'post_loan_mean': subset['post_loan'].mean(),
                            'loan_effect_mean': subset['loan_effect'].mean(),
                            'loan_effect_std': subset['loan_effect'].std(),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'significance': significance,
                            'direction': 'increase' if subset['loan_effect'].mean() > 0 else 'decrease'
                        })
                        
                    except Exception as e:
                        print(f"Statistical test failed for {sector} {protection} {buffer_km}km: {e}")
        
        print(f"  {len(sector_data)} sites with valid pre/post loan data")
    
    if not all_buffer_results:
        print("No valid data found across all buffers")
        return pd.DataFrame(), pd.DataFrame()
    
    # Combine all buffer results
    combined_results = pd.concat(all_buffer_results, ignore_index=True)
    statistical_df = pd.DataFrame(statistical_summary)
    
    # Create comprehensive visualization
    # fig = plt.figure(figsize=(24, 18))
    
    # # Create a grid layout for subplots
    # gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Colors for sectors and protection status
    sector_colors = {'Energy': '#FF6B6B', 'Transportation': '#4ECDC4'}
    protection_colors = {'protected': '#2E8B57', 'non_protected': '#CD853F'}
    
    # Plot 2: Significance heatmap
    fig = plt.figure(figsize=(12, 10))
    ax2 = fig.add_subplot(111)  # Create subplot for the heatmap
    
    if not statistical_df.empty:
        # Prepare data for heatmap
        heatmap_data = statistical_df.pivot_table(
            index=['sector', 'protection_status'], 
            columns='buffer_km', 
            values='p_value',
            fill_value=1.0
        )
        
        # Create significance level matrix with different thresholds
        # 3: p < 0.001 (highly significant)
        # 2: p < 0.01 (very significant) 
        # 1: p < 0.05 (significant)
        # 0: p >= 0.05 (not significant)
        sig_level_matrix = np.zeros_like(heatmap_data.values, dtype=int)
        sig_level_matrix[heatmap_data < 0.001] = 3
        sig_level_matrix[(heatmap_data >= 0.001) & (heatmap_data < 0.01)] = 2
        sig_level_matrix[(heatmap_data >= 0.01) & (heatmap_data < 0.05)] = 1
        
        # Create direction matrix (-1 for decrease, 1 for increase, 0 for not significant)
        direction_matrix = statistical_df.pivot_table(
            index=['sector', 'protection_status'], 
            columns='buffer_km', 
            values='loan_effect_mean',
            fill_value=0
        )
        
        # Combine significance level and direction
        final_matrix = sig_level_matrix * np.sign(direction_matrix.values)
        
        # Plot heatmap with expanded color range
        im = plt.imshow(final_matrix, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
        
        # Set ticks and labels
        ax2.set_xticks(range(final_matrix.shape[1]))
        ax2.set_xticklabels([f'{col}km' for col in direction_matrix.columns])
        ax2.set_yticks(range(final_matrix.shape[0]))
        ax2.set_yticklabels([f'{idx[0]}\\n{idx[1].replace("_", " ").title()}' for idx in direction_matrix.index])
        
        # Add text annotations with different significance levels
        for i in range(final_matrix.shape[0]):
            for j in range(final_matrix.shape[1]):
                value = final_matrix[i, j]
                if value == 3:
                    text = "↑***"  # Highly significant increase
                    color = 'white'
                elif value == 2:
                    text = "↑**"   # Very significant increase
                    color = 'white'
                elif value == 1:
                    text = "↑*"    # Significant increase
                    color = 'white'
                elif value == -3:
                    text = "↓***"  # Highly significant decrease
                    color = 'white'
                elif value == -2:
                    text = "↓**"   # Very significant decrease
                    color = 'white'
                elif value == -1:
                    text = "↓*"    # Significant decrease
                    color = 'white'
                else:
                    text = "ns"    # Not significant
                    color = 'black'
                ax2.text(j, i, text, ha="center", va="center", color=color, fontweight='bold', fontsize=10)
        
        ax2.set_title('Significance and Direction Heatmap\\n(↑=Increase, ↓=Decrease, *=p<0.05, **=p<0.01, ***=p<0.001, ns=Not Significant)')
        
        # Add colorbar with expanded levels
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        cbar.set_ticklabels(['High Sig\nDecrease', 'Very Sig\nDecrease', 'Sig\nDecrease', 'Not\nSignificant', 'Sig\nIncrease', 'Very Sig\nIncrease', 'High Sig\nIncrease'])
    
   
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"Multi-buffer comparison plot saved to: {save_path}")
    
    plt.show()
    
    # Print detailed statistical summary
    print(f"\n=== COMPREHENSIVE STATISTICAL SUMMARY ===")
    print(f"{'Buffer':<8} {'Sector':<15} {'Protection':<15} {'n':<4} {'Pre-Loan':<10} {'Post-Loan':<10} {'Effect':<10} {'t-stat':<8} {'p-value':<10} {'Sig':<5} {'Direction':<10}")
    print("-" * 120)
    
    for _, row in statistical_df.iterrows():
        print(f"{row['buffer_km']:<8} {row['sector']:<15} {row['protection_status']:<15} {row['n_sites']:<4} "
              f"{row['pre_loan_mean']:<10.4f} {row['post_loan_mean']:<10.4f} {row['loan_effect_mean']:<10.4f} "
              f"{row['t_statistic']:<8.3f} {row['p_value']:<10.3f} {row['significance']:<5} {row['direction']:<10}")
    
    # Summary by protection status
    print(f"\n=== SUMMARY BY PROTECTION STATUS ===")
    for protection in ['protected', 'non_protected']:
        prot_data = statistical_df[statistical_df['protection_status'] == protection]
        if not prot_data.empty:
            n_significant = len(prot_data[prot_data['significance'].isin(['*', '**', '***'])])
            n_increase = len(prot_data[prot_data['direction'] == 'increase'])
            n_decrease = len(prot_data[prot_data['direction'] == 'decrease'])
            
            print(f"\\n{protection.replace('_', ' ').title()} Areas:")
            print(f"  Total comparisons: {len(prot_data)}")
            print(f"  Significant results: {n_significant}/{len(prot_data)} ({n_significant/len(prot_data)*100:.1f}%)")
            print(f"  Increases after loan: {n_increase}/{len(prot_data)} ({n_increase/len(prot_data)*100:.1f}%)")
            print(f"  Decreases after loan: {n_decrease}/{len(prot_data)} ({n_decrease/len(prot_data)*100:.1f}%)")
    
    return combined_results, statistical_df

# ============================================================================
# MAIN ANALYSIS WORKFLOW
# ============================================================================

def run_complete_analysis(data_folder_path, shapefile_path, buffer_km=10, output_folder='./plots/'):
    """
    Run complete forest analysis workflow including before/after loan comparison
    
    Args:
        data_folder_path: Path to folder with CSV files
        shapefile_path: Path to shapefile with sector data
        buffer_km: Buffer distance to focus analysis on
        output_folder: Folder to save plots
    """
    
    print("=== STARTING COMPLETE FOREST ANALYSIS ===\\n")
    
    # Create output folder
    Path(output_folder).mkdir(exist_ok=True)
    
    # Step 1: Load data
    forest_df = load_forest_analysis_data(data_folder_path)
    sector_df = load_sector_data(shapefile_path)
    
    # Step 2: Combine data
    combined_df = combine_forest_sector_data(forest_df, sector_df)
    
    # Step 3: Reshape for annual analysis
    df_long = reshape_annual_data(combined_df)
    
    # Filter for loan sign year < 2018
    df_long = df_long[df_long['Loan_Sign_Year'] < 2018].copy()
    print(f"Filtered data: {len(df_long)} records after excluding loan sign year >= 2018")

    # Step 4: Add loan period classification for all years
    df_classified_all_years = add_loan_period_classification(df_long, fixed_years=None)
    
    # Step 5: Add loan period classification for fixed 5-year window
    df_classified_fixed_5_years = add_loan_period_classification(df_long, fixed_years=5)
    
    # Step 6: Create original summary statistics
    summary_stats = create_summary_table(df_long, buffer_km)
    
    # Step 7: Calculate before/after loan analysis for all years
    pre_post_data_all_years = calculate_pre_post_loan_averages(df_classified_all_years, buffer_km, fixed_years=None)
    
    # Step 8: Calculate before/after loan analysis for fixed 5-year window
    pre_post_data_fixed_5_years = calculate_pre_post_loan_averages(df_classified_fixed_5_years, buffer_km, fixed_years=5)

    # Step 9: Create visualizations
    print(f"\nCreating visualizations for {buffer_km}km buffer...")

    # NEW: Before/after loan analysis plots
    if not pre_post_data_all_years.empty:
        print("\\nCreating before/after loan analysis plots...")
        
        # Combined protection comparison - All Years
        plot_combined_protection_comparison(
            pre_post_data_all_years,
            buffer_km=buffer_km,
            save_path=f"{output_folder}/loan_impact_protection_comparison_all_years_{buffer_km}km.png",
            fixed_years=None,
            sectors_filter=['Transportation', 'Energy']
        )
        
        # Combined protection comparison - Fixed 5-Year Window
        plot_combined_protection_comparison(
            pre_post_data_fixed_5_years,
            buffer_km=buffer_km,
            save_path=f"{output_folder}/loan_impact_protection_comparison_fixed_5_years_{buffer_km}km.png",
            fixed_years=5,
            sectors_filter=['Transportation', 'Energy']
        )
    else:
        print("\\nNo valid before/after loan data available for plotting.")
    
    # Step 10: Multi-buffer sector comparison for Energy and Transportation
    print("\\n=== STEP 10: MULTI-BUFFER SECTOR COMPARISON ===")
    print("Comparing Energy and Transportation sectors across different buffer distances...")
    
    multi_buffer_results, statistical_summary = plot_multi_buffer_sector_comparison(
        df_long,
        buffer_distances=[5, 10, 15, 20, 25, 50],  # All buffer distances
        sectors_filter=['Energy', 'Transportation'],
        save_path=f"{output_folder}/multi_buffer_energy_transportation_comparison.png"
    )
    
    print("\\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to: {output_folder}")
    print("\\nGenerated plots:")
    if not pre_post_data_all_years.empty:
        print(f"  - Combined protection comparison (All Years)")
        print(f"  - Combined protection comparison (Fixed 5-Year Window)")
    if not multi_buffer_results.empty:
        print(f"  - Multi-buffer Energy vs Transportation comparison")
        print(f"  - Statistical summary across all buffer distances")
    
    return df_long, df_classified_all_years, summary_stats, pre_post_data_all_years, multi_buffer_results, statistical_summary

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Forest Analysis and Visualization Script")
    print("="*50)
    print("Usage example:")

    # Set your paths
    data_folder = "/usr2/postdoc/chishan/project_data/CODF/forestAnalysisMasked"  # Folder with CSV files
    shapefile_path = "/usr2/postdoc/chishan/project_data/CODF/CODF_Chishan.gpkg"  # Shapefile with sector data
    
    # Run complete analysis
    df_long, df_classified, summary, pre_post_results, multi_buffer_results, stats_summary = run_complete_analysis(
        data_folder_path=data_folder,
        shapefile_path=shapefile_path,
        buffer_km=25,  # Focus on 25km buffer
        output_folder='./analysis_results/')