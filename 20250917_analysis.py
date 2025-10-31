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

def plot_pre_post_loan_comparison(pre_post_data, protection_status_filter, buffer_km=10, save_path=None):
    """
    Plot comparison of pre-loan vs post-loan deforestation rates
    
    Args:
        pre_post_data: DataFrame with pre/post loan averages
        protection_status_filter: 'protected' or 'non_protected'
        buffer_km: Buffer distance being analyzed
        save_path: Path to save the plot (optional)
    """
    
    # Filter data for specific protection status
    plot_data = pre_post_data[pre_post_data['protection_status'] == protection_status_filter].copy()
    
    if plot_data.empty:
        print(f"No data found for {protection_status_filter} areas")
        return
    
    # Remove rows with missing data
    plot_data = plot_data.dropna(subset=['pre_loan', 'post_loan'])
    
    if plot_data.empty:
        print(f"No valid pre/post loan data for {protection_status_filter} areas")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors by sector
    sectors = plot_data['Sector'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))
    sector_colors = dict(zip(sectors, colors))
    
    # Plot 1: Scatter plot comparing pre vs post
    for sector in sectors:
        sector_data = plot_data[plot_data['Sector'] == sector]
        ax1.scatter(sector_data['pre_loan'], sector_data['post_loan'], 
                   c=[sector_colors[sector]], label=sector, alpha=0.7, s=50)
    
    # Add diagonal line (no change)
    max_val = max(plot_data['pre_loan'].max(), plot_data['post_loan'].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No Change')
    ax1.set_xlabel('Pre-Loan Deforestation Rate (%)')
    ax1.set_ylabel('Post-Loan Deforestation Rate (%)')
    ax1.set_title('Pre-Loan vs Post-Loan Deforestation Rates')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of loan effects by sector
    loan_effects = []
    sector_labels = []
    for sector in sectors:
        sector_data = plot_data[plot_data['Sector'] == sector]
        if not sector_data.empty:
            loan_effects.append(sector_data['loan_effect'].dropna().values)
            sector_labels.append(sector)
    
    if loan_effects:
        bp = ax2.boxplot(loan_effects, labels=sector_labels, patch_artist=True)
        for patch, sector in zip(bp['boxes'], sector_labels):
            patch.set_facecolor(sector_colors[sector])
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Effect')
        ax2.set_ylabel('Loan Effect (Post - Pre) %')
        ax2.set_title('Distribution of Loan Effects by Sector')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of loan effects
    ax3.hist(plot_data['loan_effect'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Effect')
    ax3.axvline(x=plot_data['loan_effect'].mean(), color='orange', linestyle='-', 
               label=f'Mean: {plot_data["loan_effect"].mean():.3f}%')
    ax3.set_xlabel('Loan Effect (Post - Pre) %')
    ax3.set_ylabel('Number of Sites')
    ax3.set_title('Distribution of Loan Effects')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time series by loan sign year
    loan_year_summary = plot_data.groupby('Loan_Sign_Year')[['pre_loan', 'post_loan']].mean()
    ax4.plot(loan_year_summary.index, loan_year_summary['pre_loan'], 
            marker='o', label='Pre-Loan Average', linewidth=2)
    ax4.plot(loan_year_summary.index, loan_year_summary['post_loan'], 
            marker='s', label='Post-Loan Average', linewidth=2)
    ax4.set_xlabel('Loan Sign Year')
    ax4.set_ylabel('Average Deforestation Rate (%)')
    ax4.set_title('Deforestation Rates by Loan Sign Year')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    protection_title = protection_status_filter.replace('_', ' ').title()
    fig.suptitle(f'Pre-Loan vs Post-Loan Deforestation Analysis\\n{protection_title} Areas ({buffer_km}km Buffer)', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\\n=== {protection_title} Areas Summary ===")
    print(f"Number of sites: {len(plot_data)}")
    print(f"Mean pre-loan rate: {plot_data['pre_loan'].mean():.4f}%")
    print(f"Mean post-loan rate: {plot_data['post_loan'].mean():.4f}%")
    print(f"Mean loan effect: {plot_data['loan_effect'].mean():.4f}%")
    print(f"Sites with increased deforestation: {(plot_data['loan_effect'] > 0).sum()}")
    print(f"Sites with decreased deforestation: {(plot_data['loan_effect'] < 0).sum()}")

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

def plot_annual_deforestation_by_sector(df_long, buffer_km=10, save_path=None):
    """
    Plot annual deforestation rates by sector and protection status
    
    Args:
        df_long: Long format DataFrame
        buffer_km: Buffer distance to analyze
        save_path: Path to save the plot (optional)
    """
    
    # Filter data for specific buffer distance
    df_plot = df_long[df_long['buffer_distance_km'] == buffer_km].copy()
    
    if df_plot.empty:
        print(f"No data found for {buffer_km}km buffer")
        return
    
    # Calculate annual means by sector and protection status
    annual_means = df_plot.groupby(['year', 'Sector', 'protection_status'])['annual_defor_rate_pct'].mean().reset_index()
    
    # Get unique sectors
    sectors = sorted(df_plot['Sector'].dropna().unique())
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = {'protected': '#2E8B57', 'non_protected': '#CD853F'}
    
    for i, sector in enumerate(sectors[:4]):  # Show first 4 sectors
        ax = axes[i]
        sector_data = annual_means[annual_means['Sector'] == sector]
        
        for protection_status in ['protected', 'non_protected']:
            status_data = sector_data[sector_data['protection_status'] == protection_status]
            if not status_data.empty:
                ax.plot(status_data['year'], status_data['annual_defor_rate_pct'], 
                       marker='o', linewidth=2, markersize=4,
                       color=colors[protection_status],
                       label=f'{protection_status.replace("_", " ").title()}')
        
        ax.set_title(f'{sector} Sector', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual Deforestation Rate (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2000.5, 2024.5)
    
    # Hide unused subplots
    for i in range(len(sectors), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Annual Deforestation Rates by Sector and Protection Status\\n({buffer_km}km Buffer)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_protection_effect_comparison(df_long, buffer_km=10, save_path=None):
    """
    Plot comparison of protection effect across sectors
    
    Args:
        df_long: Long format DataFrame
        buffer_km: Buffer distance to analyze
        save_path: Path to save the plot (optional)
    """
    
    # Filter data
    df_plot = df_long[df_long['buffer_distance_km'] == buffer_km].copy()
    
    # Calculate mean rates by sector and protection status
    sector_means = df_plot.groupby(['Sector', 'protection_status'])['annual_defor_rate_pct'].mean().reset_index()
    
    # Pivot to get protected vs non-protected comparison
    comparison = sector_means.pivot(index='Sector', columns='protection_status', values='annual_defor_rate_pct')
    comparison = comparison.fillna(0)
    
    # Calculate difference (non-protected - protected)
    if 'protected' in comparison.columns and 'non_protected' in comparison.columns:
        comparison['difference'] = comparison['non_protected'] - comparison['protected']
        comparison['protection_effect_pct'] = np.where(
            comparison['non_protected'] > 0,
            (comparison['difference'] / comparison['non_protected']) * 100,
            0
        )
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Side-by-side comparison
    if 'protected' in comparison.columns and 'non_protected' in comparison.columns:
        comparison[['protected', 'non_protected']].plot(kind='bar', ax=ax1, 
                                                        color=['#2E8B57', '#CD853F'],
                                                        width=0.8)
        ax1.set_title('Average Annual Deforestation Rate by Sector\\nProtected vs Non-Protected Areas')
        ax1.set_ylabel('Annual Deforestation Rate (%)')
        ax1.set_xlabel('Sector')
        ax1.legend(['Protected Areas', 'Non-Protected Areas'])
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Protection effect
    if 'protection_effect_pct' in comparison.columns:
        comparison['protection_effect_pct'].plot(kind='bar', ax=ax2, color='#4682B4')
        ax2.set_title('Protection Effect by Sector\\n(% Reduction in Deforestation Rate)')
        ax2.set_ylabel('Protection Effect (%)')
        ax2.set_xlabel('Sector')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Forest Protection Effectiveness Analysis ({buffer_km}km Buffer)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return comparison

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
    
    # Step 7: Create visualizations
    print(f"\\nCreating visualizations for {buffer_km}km buffer...")
    
    # Original annual trends plot
    plot_annual_deforestation_by_sector(
        df_long, 
        buffer_km=buffer_km,
        save_path=f"{output_folder}/annual_deforestation_by_sector_{buffer_km}km.png"
    )
    
    # Original protection effect comparison
    comparison_results = plot_protection_effect_comparison(
        df_long,
        buffer_km=buffer_km, 
        save_path=f"{output_folder}/protection_effect_comparison_{buffer_km}km.png"
    )
    
    # NEW: Before/after loan analysis plots
    if not pre_post_data_all_years.empty:
        print("\\nCreating before/after loan analysis plots...")
        
        # Protected areas loan impact
        plot_pre_post_loan_comparison(
            pre_post_data_all_years,
            protection_status_filter='protected',
            buffer_km=buffer_km,
            save_path=f"{output_folder}/loan_impact_protected_areas_{buffer_km}km.png"
        )
        
        # Non-protected areas loan impact  
        plot_pre_post_loan_comparison(
            pre_post_data_all_years,
            protection_status_filter='non_protected',
            buffer_km=buffer_km,
            save_path=f"{output_folder}/loan_impact_non_protected_areas_{buffer_km}km.png"
        )
        
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
    
    print("\\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to: {output_folder}")
    print("\\nGenerated plots:")
    print(f"  - Annual deforestation by sector")
    print(f"  - Protection effect comparison")
    if not pre_post_data_all_years.empty:
        print(f"  - Loan impact in protected areas")
        print(f"  - Loan impact in non-protected areas") 
        print(f"  - Combined protection comparison (All Years)")
        print(f"  - Combined protection comparison (Fixed 5-Year Window)")
    
    return df_long, df_classified_all_years, summary_stats, comparison_results, pre_post_data_all_years

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Forest Analysis and Visualization Script")
    print("="*50)
    print("Usage example:")
    print("""
    # Set your paths
    data_folder = "./forestAnalysisSimple/"  # Folder with CSV files
    shapefile_path = "path/to/your/shapefile.shp"  # Shapefile with sector data
    
    # Run complete analysis
    df_long, df_classified, summary, comparison, pre_post_results = run_complete_analysis(
        data_folder_path=data_folder,
        shapefile_path=shapefile_path,
        buffer_km=10,  # Focus on 10km buffer
        output_folder='./analysis_results/'
    )
    
    # The script will:
    # 1. Load and combine all CSV files
    # 2. Match with sector information and loan sign year using BU ID
    # 3. Calculate annual deforestation rates
    # 4. Classify data into pre-loan and post-loan periods
    # 5. Create comparison plots showing protection effectiveness
    # 6. Generate before/after loan analysis plots for both protected and non-protected areas
    # 7. Show loan impact comparison across protection status
    """)

    # Set your paths
    data_folder = "/usr2/postdoc/chishan/project_data/CODF/forestAnalysisMasked"  # Folder with CSV files
    shapefile_path = "/usr2/postdoc/chishan/project_data/CODF/CODF_Chishan.gpkg"  # Shapefile with sector data
    
    # Run complete analysis
    df_long, df_classified, summary, comparison, pre_post_results = run_complete_analysis(
        data_folder_path=data_folder,
        shapefile_path=shapefile_path,
        buffer_km=25,  # Focus on 25km buffer
        output_folder='./analysis_results/')