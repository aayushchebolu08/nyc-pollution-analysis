import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from scipy import stats
import seaborn as sns

# Set up premium visualization style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial']
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['xtick.color'] = '#333333'
mpl.rcParams['ytick.color'] = '#333333'

# Color palettes for different pollutants
COLORS = {
    'ozone': ['#1A5276', '#2874A6', '#3498DB', '#5DADE2', '#85C1E9'],
    'no2': ['#641E16', '#922B21', '#C0392B', '#E74C3C', '#F1948A'],
    'pm25': ['#145A32', '#196F3D', '#229954', '#27AE60', '#58D68D'],
    'health': ['#6C3483', '#7D3C98', '#8E44AD', '#A569BD', '#BB8FCE'],
    'background': '#F8F9F9'
}


def calculate_yearly_statistics(df, pollutant_name, summer_only=False, confidence_level=0.95):
    """
    Calculate yearly statistics including mean, std, confidence intervals, and sample sizes
    
    Args:
        df: DataFrame with air quality data
        pollutant_name: Name of the pollutant to analyze
        summer_only: If True, filter for summer months only (June-August)
        confidence_level: Confidence level for intervals (default 0.95)
    
    Returns:
        DataFrame with yearly statistics
    """
    # Filter for specific pollutant data
    pollutant_data = df[df['Name'] == pollutant_name].copy()
    
    # Convert dates and extract year and month
    pollutant_data['Start_Date'] = pd.to_datetime(pollutant_data['Start_Date'])
    pollutant_data['Year'] = pollutant_data['Start_Date'].dt.year
    pollutant_data['Month'] = pollutant_data['Start_Date'].dt.month
    
    # Filter for summer months if requested
    if summer_only:
        pollutant_data = pollutant_data[(pollutant_data['Month'] >= 6) & (pollutant_data['Month'] <= 8)]
    
    # Calculate yearly statistics
    yearly_stats = pollutant_data.groupby('Year')['Data Value'].agg([
        'mean',
        'std',
        'count',
        'sem'  # Standard error of the mean
    ]).reset_index()
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    yearly_stats['degrees_freedom'] = yearly_stats['count'] - 1
    yearly_stats['t_critical'] = yearly_stats['degrees_freedom'].apply(
        lambda df: stats.t.ppf(1 - alpha/2, df) if df > 0 else np.nan
    )
    
    # Margin of error
    yearly_stats['margin_error'] = yearly_stats['t_critical'] * yearly_stats['sem']
    
    # Confidence interval bounds
    yearly_stats['ci_lower'] = yearly_stats['mean'] - yearly_stats['margin_error']
    yearly_stats['ci_upper'] = yearly_stats['mean'] + yearly_stats['margin_error']
    
    # For single observations, use std as uncertainty estimate
    single_obs_mask = yearly_stats['count'] == 1
    if single_obs_mask.any():
        overall_std = pollutant_data['Data Value'].std()
        yearly_stats.loc[single_obs_mask, 'std'] = overall_std
        yearly_stats.loc[single_obs_mask, 'ci_lower'] = yearly_stats.loc[single_obs_mask, 'mean'] - overall_std
        yearly_stats.loc[single_obs_mask, 'ci_upper'] = yearly_stats.loc[single_obs_mask, 'mean'] + overall_std
    
    return yearly_stats


def plot_yearly_data_with_uncertainty(df, pollutant_name, color_key, filename, title_prefix=None, summer_only=False, confidence_level=0.95):
    """
    Create a premium line chart showing yearly average pollutant levels with uncertainty bands
    """
    # Get yearly statistics
    yearly_stats = calculate_yearly_statistics(df, pollutant_name, summer_only, confidence_level)
    
    # Create custom color gradient for the line
    cmap = LinearSegmentedColormap.from_list(f"{color_key}_gradient", COLORS[color_key])
    
    # Create the figure with a specific aspect ratio
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Custom background with very subtle gradient
    gradient = np.linspace(0, 1, 100).reshape(-1, 1)
    ax.imshow(gradient, aspect='auto', 
              extent=[yearly_stats['Year'].min() - 0.5, yearly_stats['Year'].max() + 0.5,
                     yearly_stats['ci_lower'].min() * 0.95, yearly_stats['ci_upper'].max() * 1.05],
              alpha=0.05, cmap=LinearSegmentedColormap.from_list("bg_gradient", [COLORS['background'], '#ECF0F1']))
    
    # Plot confidence interval band
    ax.fill_between(yearly_stats['Year'], 
                    yearly_stats['ci_lower'], 
                    yearly_stats['ci_upper'],
                    alpha=0.2, 
                    color=COLORS[color_key][2],
                    label=f'{int(confidence_level*100)}% Confidence Interval')
    
    # Plot standard deviation band (lighter)
    ax.fill_between(yearly_stats['Year'], 
                    yearly_stats['mean'] - yearly_stats['std'], 
                    yearly_stats['mean'] + yearly_stats['std'],
                    alpha=0.1, 
                    color=COLORS[color_key][1],
                    label='±1 Standard Deviation')
    
    # Plot line with gradient effect
    points = np.array([yearly_stats['Year'], yearly_stats['mean']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(yearly_stats['Year'].min(), yearly_stats['Year'].max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3.5, alpha=0.9)
    lc.set_array(yearly_stats['Year'])
    line = ax.add_collection(lc)
    
    # Add data points with error bars
    errorbar = ax.errorbar(yearly_stats['Year'], yearly_stats['mean'],
                          yerr=yearly_stats['margin_error'],
                          fmt='o',
                          markersize=8,
                          color=COLORS[color_key][2],
                          markerfacecolor=COLORS[color_key][2],
                          markeredgecolor='white',
                          markeredgewidth=1.5,
                          ecolor=COLORS[color_key][1],
                          elinewidth=2,
                          capsize=4,
                          capthick=2,
                          alpha=0.8,
                          zorder=5,
                          label='Annual Mean ± CI')
    
    # Add value labels with uncertainty
    for _, row in yearly_stats.iterrows():
        label_text = f"{row['mean']:.2f}±{row['margin_error']:.2f}"
        if row['count'] > 1:
            label_text += f"\n(n={int(row['count'])})"
        
        label = ax.annotate(label_text,
                           (row['Year'], row['mean']),
                           xytext=(0, 15),
                           textcoords='offset points',
                           ha='center',
                           va='bottom',
                           fontsize=8,
                           color='#333333',
                           fontweight='bold')
        label.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
    
    # Add trend line
    z = np.polyfit(yearly_stats['Year'], yearly_stats['mean'], 1)
    p = np.poly1d(z)
    trend = ax.plot(yearly_stats['Year'], p(yearly_stats['Year']),
                   linestyle='--',
                   linewidth=2,
                   color=COLORS[color_key][0],
                   alpha=0.7,
                   zorder=2,
                   label='Linear Trend')
    
    # Calculate trend significance
    slope, intercept, r_value, p_value, std_err = stats.linregress(yearly_stats['Year'], yearly_stats['mean'])
    
    # Add trend annotation with significance
    trend_direction = "increasing" if slope > 0 else "decreasing"
    significance = "significant" if p_value < 0.05 else "not significant"
    trend_color = '#7B241C' if slope > 0 else '#145A32'
    
    trend_text = f"Trend: {trend_direction} at {abs(slope):.3f} units/year\n"
    trend_text += f"R² = {r_value**2:.3f}, p = {p_value:.3f} ({significance})"
    
    ax.text(0.02, 0.02,
            trend_text,
            transform=ax.transAxes,
            fontsize=9,
            color=trend_color,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Set x-axis to show all years
    ax.set_xticks(yearly_stats['Year'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Styling y-axis
    ax.yaxis.grid(True, linestyle='-', alpha=0.15, linewidth=1.5)
    ax.xaxis.grid(False)
    
    # Extract nice pollutant name for display
    pollutant_short = pollutant_name.split('(')[0].strip()
    season_text = "Summer " if summer_only else "Annual "
    
    # Override title if provided
    if title_prefix:
        main_title = title_prefix
    else:
        main_title = f"NYC {season_text}{pollutant_short} Trends with Uncertainty"
    
    # Add labels and title with premium styling
    title = ax.set_title(main_title,
                        fontsize=18,
                        pad=20,
                        fontweight='bold',
                        color='#2C3E50')
    title.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='#EAECEE')])
    
    subtitle = ax.text(0.5, 0.97, f"{season_text}Average {pollutant_short} Concentration with {int(confidence_level*100)}% Confidence Intervals",
                      transform=ax.transAxes,
                      fontsize=12,
                      ha='center',
                      va='top',
                      color='#566573')
    
    ax.set_xlabel('Year', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')
    ax.set_ylabel(f'{pollutant_short} Concentration', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Adjust y-axis limits for better spacing
    ymin = yearly_stats['ci_lower'].min()
    ymax = yearly_stats['ci_upper'].max()
    range_y = ymax - ymin
    ax.set_ylim(ymin - range_y * 0.05, ymax + range_y * 0.1)
    
    # Add source text
    fig.text(0.02, 0.02, 'Source: NYC Air Quality Data', fontsize=8, color='#7F8C8D')
    
    # Add statistical info
    fig.text(0.98, 0.02, f'Statistical Analysis | CI: {int(confidence_level*100)}%', 
             fontsize=9, color='#2C3E50', ha='right', fontweight='bold')
    
    # Tight layout for optimal spacing
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    return yearly_stats


def create_statistics_summary_table(yearly_stats, pollutant_name, filename, summer_only=False):
    """
    Create a summary table of the yearly statistics
    """
    # Create a formatted table
    fig, ax = plt.subplots(figsize=(12, len(yearly_stats) * 0.5 + 2), dpi=300)
    fig.patch.set_facecolor(COLORS['background'])
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for _, row in yearly_stats.iterrows():
        table_data.append([
            int(row['Year']),
            f"{row['mean']:.3f}",
            f"{row['std']:.3f}",
            int(row['count']),
            f"{row['ci_lower']:.3f}",
            f"{row['ci_upper']:.3f}",
            f"±{row['margin_error']:.3f}"
        ])
    
    # Column headers
    columns = ['Year', 'Mean', 'Std Dev', 'Sample Size', 
               '95% CI Lower', '95% CI Upper', 'Margin of Error']
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternating row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F8F9FA')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Title
    pollutant_short = pollutant_name.split('(')[0].strip()
    season_text = "Summer " if summer_only else "Annual "
    title = f"Statistical Summary: NYC {season_text}{pollutant_short} Data"
    
    plt.suptitle(title, fontsize=16, fontweight='bold', color='#2C3E50', y=0.95)
    
    # Save
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()


def main():
    # Load data
    df = pd.read_csv('Air_Quality.csv')
    
    # Example usage with enhanced statistics
    print("Creating enhanced visualizations with uncertainty analysis...")
    
    # 1. Summer ozone with uncertainty
    ozone_stats = plot_yearly_data_with_uncertainty(
        df, 'Ozone (O3)', 'ozone', 
        'nyc_summer_ozone_trend_with_uncertainty.png', 
        summer_only=True
    )
    
    # Create statistics table for ozone
    create_statistics_summary_table(
        ozone_stats, 'Ozone (O3)', 
        'ozone_statistics_table.png', 
        summer_only=True
    )
    
    # 2. NO2 with uncertainty
    no2_stats = plot_yearly_data_with_uncertainty(
        df, 'Nitrogen dioxide (NO2)', 'no2', 
        'nyc_annual_no2_trend_with_uncertainty.png'
    )
    
    # 3. PM2.5 with uncertainty
    pm25_stats = plot_yearly_data_with_uncertainty(
        df, 'Fine particles (PM 2.5)', 'pm25', 
        'nyc_annual_pm25_trend_with_uncertainty.png'
    )
    
    # Print summary statistics
    print("\n=== OZONE STATISTICS (Summer) ===")
    print(ozone_stats[['Year', 'mean', 'std', 'count', 'ci_lower', 'ci_upper']].to_string(index=False))
    
    print("\n=== CALCULATION EXPLANATION ===")
    print("Standard Deviation: Measures variability of data points within each year")
    print("Confidence Interval: Range likely to contain the true population mean")
    print("Margin of Error: Half-width of the confidence interval")
    print("Sample Size (n): Number of measurements available for each year")
    print("\nFor years with n=1, overall dataset std dev is used as uncertainty estimate")
    
    print("\nFiles created:")
    print("- nyc_summer_ozone_trend_with_uncertainty.png")
    print("- ozone_statistics_table.png")
    print("- nyc_annual_no2_trend_with_uncertainty.png") 
    print("- nyc_annual_pm25_trend_with_uncertainty.png")


if __name__ == "__main__":
    main()