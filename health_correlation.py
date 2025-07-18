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

# Color palettes for different boroughs
COLORS = {
    'bronx': ['#641E16', '#922B21', '#C0392B', '#E74C3C', '#F1948A'],
    'brooklyn': ['#1A5276', '#2874A6', '#3498DB', '#5DADE2', '#85C1E9'],
    'manhattan': ['#145A32', '#196F3D', '#229954', '#27AE60', '#58D68D'],
    'queens': ['#6C3483', '#7D3C98', '#8E44AD', '#A569BD', '#BB8FCE'],
    'staten_island': ['#B7950B', '#D4AF37', '#F1C40F', '#F7DC6F', '#FCF3CF'],
    'background': '#F8F9F9'
}

# Borough color mapping
BOROUGH_COLORS = {
    'Bronx': COLORS['bronx'][2],
    'Brooklyn': COLORS['brooklyn'][2], 
    'Manhattan': COLORS['manhattan'][2],
    'Queens': COLORS['queens'][2],
    'Staten Island': COLORS['staten_island'][2]
}

# Neighborhood to Borough mapping
NEIGHBORHOOD_TO_BOROUGH = {
# Bronx neighborhoods
    'Belmont and East Tremont': 'Bronx',
    'Crotona -Tremont': 'Bronx',
    'Fordham - Bronx Pk': 'Bronx',
    'Fordham and University Heights': 'Bronx',
    'High Bridge - Morrisania': 'Bronx',
    'Hunts Point - Mott Haven': 'Bronx',
    'Kingsbridge - Riverdale': 'Bronx',
    'Kingsbridge Heights and Bedford': 'Bronx',
    'Morris Park and Bronxdale': 'Bronx',
    'Morrisania and Crotona': 'Bronx',
    'Mott Haven and Melrose': 'Bronx',
    'Northeast Bronx': 'Bronx',
    'Parkchester and Soundview': 'Bronx',
    'Pelham - Throgs Neck': 'Bronx',
    'Riverdale and Fieldston': 'Bronx',
    'South Bronx': 'Bronx',
    'Throgs Neck and Co-op City': 'Bronx',
    'Williamsbridge and Baychester': 'Bronx',

# Brooklyn neighborhoods
    'Bedford Stuyvesant': 'Brooklyn',
    'Bedford Stuyvesant - Crown Heights': 'Brooklyn',
    'Bensonhurst': 'Brooklyn',
    'Bensonhurst - Bay Ridge': 'Brooklyn',
    'Borough Park': 'Brooklyn',
    'Brownsville': 'Brooklyn',
    'Bushwick': 'Brooklyn',
    'Canarsie - Flatlands': 'Brooklyn',
    'Coney Island': 'Brooklyn',
    'Coney Island - Sheepshead Bay': 'Brooklyn',
    'Crown Heights and Prospect Heights': 'Brooklyn',
    'Downtown - Heights - Slope': 'Brooklyn',
    'East Flatbush': 'Brooklyn',
    'East Flatbush - Flatbush': 'Brooklyn',
    'East New York': 'Brooklyn',
    'East New York and Starrett City': 'Brooklyn',
    'Flatbush and Midwood': 'Brooklyn',
    'Flatlands and Canarsie': 'Brooklyn',
    'Fort Greene and Brooklyn Heights': 'Brooklyn',
    'Greenpoint': 'Brooklyn',
    'Greenpoint and Williamsburg': 'Brooklyn',
    'Park Slope and Carroll Gardens': 'Brooklyn',
    'Sheepshead Bay': 'Brooklyn',
    'South Crown Heights and Lefferts Gardens': 'Brooklyn',
    'Sunset Park': 'Brooklyn',
    'Williamsburg - Bushwick': 'Brooklyn',

# Manhattan neighborhoods
    'Central Harlem': 'Manhattan',
    'Central Harlem - Morningside Heights': 'Manhattan',
    'Chelsea - Clinton': 'Manhattan',
    'Chelsea-Village': 'Manhattan',
    'Clinton and Chelsea': 'Manhattan',
    'East Harlem': 'Manhattan',
    'East Harlem (CD11)': 'Manhattan',
    'Financial District': 'Manhattan',
    'Gramercy Park - Murray Hill': 'Manhattan',
    'Greenwich Village - SoHo': 'Manhattan',
    'Greenwich Village and Soho': 'Manhattan',
    'Lower East Side and Chinatown': 'Manhattan',
    'Lower Manhattan': 'Manhattan',
    'Manhattan': 'Manhattan',
    'Midtown': 'Manhattan',
    'Morningside Heights and Hamilton Heights': 'Manhattan',
    'Stuyvesant Town and Turtle Bay': 'Manhattan',
    'Union Square - Lower East Side': 'Manhattan',
    'Union Square-Lower Manhattan': 'Manhattan',
    'Upper East Side': 'Manhattan',
    'Upper East Side-Gramercy': 'Manhattan',
    'Upper West Side': 'Manhattan',
    'Washington Heights': 'Manhattan',
    'Washington Heights and Inwood': 'Manhattan',

# Queens neighborhoods
    'Bayside - Little Neck': 'Queens',
    'Bayside Little Neck-Fresh Meadows': 'Queens',
    'Bayside and Little Neck': 'Queens',
    'Elmhurst and Corona': 'Queens',
    'Flushing - Clearview': 'Queens',
    'Flushing and Whitestone': 'Queens',
    'Fresh Meadows': 'Queens',
    'Hillcrest and Fresh Meadows': 'Queens',
    'Jackson Heights': 'Queens',
    'Jamaica': 'Queens',
    'Jamaica and Hollis': 'Queens',
    'Kew Gardens and Woodhaven': 'Queens',
    'Queens': 'Queens',
    'Queens Village': 'Queens',
    'Rego Park and Forest Hills': 'Queens',
    'Ridgewood - Forest Hills': 'Queens',
    'Ridgewood and Maspeth': 'Queens',
    'Rockaway and Broad Channel': 'Queens',
    'Rockaways': 'Queens',
    'Southeast Queens': 'Queens',
    'South Ozone Park and Howard Beach': 'Queens',
    'Southwest Queens': 'Queens',
    'West Queens': 'Queens',

# Staten Island neighborhoods
    'Northern SI': 'Staten Island',
    'Port Richmond': 'Staten Island',
    'South Beach - Tottenville': 'Staten Island',
    'South Beach and Willowbrook': 'Staten Island',
    'Southern SI': 'Staten Island',
    'St. George and Stapleton': 'Staten Island',
    'Stapleton - St. George': 'Staten Island',
    'Staten Island': 'Staten Island',
    'Tottenville and Great Kills': 'Staten Island',
    'Willowbrook': 'Staten Island',
}

# Define the respiratory/cardiovascular indicators we want to analyze
RESPIRATORY_INDICATORS = [
    'Respiratory hospitalizations due to PM2.5 (age 20+)',
]


def load_and_filter_data(filename):
    """Load the CSV and filter for respiratory/cardiovascular indicators"""
    try:
        # Load with proper column names
        df = pd.read_csv(filename)
        print(f"Loaded data: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Check if we have the expected columns
        expected_cols = ['Name', 'Geo Place Name', 'Start_Date', 'Data Value']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing expected columns: {missing_cols}")
            print("Available columns:", df.columns.tolist())
        
        # Show unique indicators in the data
        print(f"\nAll available indicators:")
        unique_names = df['Name'].unique()
        for name in sorted(unique_names):
            print(f"  - {name}")
        
        # Filter for respiratory/cardiovascular indicators
        df_filtered = df[df['Name'].isin(RESPIRATORY_INDICATORS)].copy()
        print(f"\nFiltered for respiratory indicators: {len(df_filtered)} rows")
        
        if len(df_filtered) == 0:
            print("No respiratory indicators found. Checking for partial matches...")
            for indicator in RESPIRATORY_INDICATORS:
                partial_matches = df[df['Name'].str.contains(indicator.split()[0], case=False, na=False)]
                if len(partial_matches) > 0:
                    print(f"Partial matches for '{indicator.split()[0]}':")
                    for match in partial_matches['Name'].unique():
                        print(f"  - {match}")
        
        print(f"\nIndicators found in data:")
        for indicator in RESPIRATORY_INDICATORS:
            count = len(df_filtered[df_filtered['Name'] == indicator])
            print(f"  - {indicator}: {count} records")
        
        return df_filtered
        
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please ensure the file is in the current directory.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def add_borough_column(df):
    """Add a Borough column to the dataframe based on neighborhood mapping"""
    if len(df) == 0:
        return df
        
    df_copy = df.copy()
    df_copy['Borough'] = df_copy['Geo Place Name'].map(NEIGHBORHOOD_TO_BOROUGH)
    
    print("\n=== NEIGHBORHOOD TO BOROUGH MAPPING ===")
    unique_neighborhoods = df_copy['Geo Place Name'].unique()
    for neighborhood in sorted(unique_neighborhoods):
        borough = NEIGHBORHOOD_TO_BOROUGH.get(neighborhood, 'UNMAPPED')
        print(f"{neighborhood} → {borough}")
    
    unmapped = df_copy[df_copy['Borough'].isna()]
    if len(unmapped) > 0:
        print(f"\nWARNING: {len(unmapped)} rows with unmapped neighborhoods:")
        unmapped_neighborhoods = unmapped['Geo Place Name'].unique()
        for neighborhood in sorted(unmapped_neighborhoods):
            print(f"  - {neighborhood}")
    
    print(f"\nBorough distribution:")
    borough_counts = df_copy['Borough'].value_counts()
    print(borough_counts)
    
    return df_copy


def prepare_time_data(df):
    """Prepare time-based data from start_date"""
    if len(df) == 0:
        return df
        
    df_copy = df.copy()
    
    # Convert start_date to datetime and extract year
    df_copy['Start_Date'] = pd.to_datetime(df_copy['Start_Date'], errors='coerce')
    df_copy['Year'] = df_copy['Start_Date'].dt.year
    
    # Convert data value to numeric
    df_copy['data_value_numeric'] = pd.to_numeric(df_copy['Data Value'], errors='coerce')
    
    # Remove rows with missing critical data
    df_copy = df_copy.dropna(subset=['Year', 'data_value_numeric', 'Borough'])
    
    print(f"\nTime data preparation:")
    print(f"  - Available years: {sorted(df_copy['Year'].unique())}")
    print(f"  - Records with valid data: {len(df_copy)}")
    
    return df_copy


def calculate_yearly_statistics(df, borough_name, indicator_name=None, confidence_level=0.95):
    """Calculate yearly statistics for air quality indicators by borough"""
    borough_data = df[df['Borough'] == borough_name].copy()
    
    # Optionally filter by specific indicator
    if indicator_name:
        borough_data = borough_data[borough_data['Name'] == indicator_name]
    
    if len(borough_data) == 0:
        print(f"No valid data found for {borough_name}" + (f" - {indicator_name}" if indicator_name else ""))
        return pd.DataFrame()
    
    # Calculate yearly statistics
    yearly_stats = borough_data.groupby('Year')['data_value_numeric'].agg([
        'mean',
        'std',
        'count',
        'sem',
        'min',
        'max'
    ]).reset_index()
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    yearly_stats['degrees_freedom'] = yearly_stats['count'] - 1
    yearly_stats['t_critical'] = yearly_stats['degrees_freedom'].apply(
        lambda df: stats.t.ppf(1 - alpha/2, df) if df > 0 else np.nan
    )
    
    yearly_stats['margin_error'] = yearly_stats['t_critical'] * yearly_stats['sem']
    yearly_stats['ci_lower'] = yearly_stats['mean'] - yearly_stats['margin_error']
    yearly_stats['ci_upper'] = yearly_stats['mean'] + yearly_stats['margin_error']
    
    # For single observations, use overall std as uncertainty estimate
    single_obs_mask = yearly_stats['count'] == 1
    if single_obs_mask.any():
        overall_std = borough_data['data_value_numeric'].std()
        yearly_stats.loc[single_obs_mask, 'std'] = overall_std
        yearly_stats.loc[single_obs_mask, 'ci_lower'] = yearly_stats.loc[single_obs_mask, 'mean'] - overall_std
        yearly_stats.loc[single_obs_mask, 'ci_upper'] = yearly_stats.loc[single_obs_mask, 'mean'] + overall_std
    
    return yearly_stats


def plot_individual_borough_enhanced(df, borough_name, filename, confidence_level=0.95):
    """Create enhanced individual borough chart with all indicators combined"""
    borough_data = df[df['Borough'] == borough_name]
    
    if len(borough_data) == 0:
        print(f"No data found for {borough_name}")
        return None
    
    # Calculate stats for all indicators combined
    yearly_stats = calculate_yearly_statistics(df, borough_name, confidence_level=confidence_level)
    
    if len(yearly_stats) == 0:
        print(f"No valid statistics calculated for {borough_name}")
        return None
    
    borough_key = borough_name.lower().replace(' ', '_')
    color_scheme = COLORS.get(borough_key, COLORS['bronx'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor(COLORS['background'])
    
    # Main chart - combined indicators
    ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=2)
    ax1.set_facecolor(COLORS['background'])
    
    # Plot statistical bands and main line (same as before)
    ax1.fill_between(yearly_stats['Year'], 
                     yearly_stats['mean'] - yearly_stats['std'], 
                     yearly_stats['mean'] + yearly_stats['std'],
                     alpha=0.15, 
                     color=color_scheme[1],
                     label='±1 Standard Deviation')
    
    ax1.fill_between(yearly_stats['Year'], 
                     yearly_stats['ci_lower'], 
                     yearly_stats['ci_upper'],
                     alpha=0.25, 
                     color=color_scheme[2],
                     label=f'{int(confidence_level*100)}% Confidence Interval')
    
    # Main line with gradient effect
    points = np.array([yearly_stats['Year'], yearly_stats['mean']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = LinearSegmentedColormap.from_list(f"{borough_key}_gradient", color_scheme)
    norm = plt.Normalize(yearly_stats['Year'].min(), yearly_stats['Year'].max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4, alpha=0.9)
    lc.set_array(yearly_stats['Year'])
    ax1.add_collection(lc)
    
    # Data points with error bars
    ax1.errorbar(yearly_stats['Year'], yearly_stats['mean'],
                yerr=yearly_stats['margin_error'],
                fmt='o',
                markersize=10,
                color=color_scheme[2],
                markerfacecolor=color_scheme[2],
                markeredgecolor='white',
                markeredgewidth=2,
                ecolor=color_scheme[1],
                elinewidth=2.5,
                capsize=5,
                capthick=2,
                alpha=0.9,
                zorder=5,
                label='Mean ± CI')
    
    # Add value labels
    for _, row in yearly_stats.iterrows():
        label_text = f"{row['mean']:.1f}"
        if row['count'] > 1:
            label_text += f"\n(n={int(row['count'])})"
        
        ax1.annotate(label_text,
                    (row['Year'], row['mean']),
                    xytext=(0, 20),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='#2C3E50',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Trend line
    if len(yearly_stats) > 1:
        z = np.polyfit(yearly_stats['Year'], yearly_stats['mean'], 1)
        p = np.poly1d(z)
        ax1.plot(yearly_stats['Year'], p(yearly_stats['Year']),
                linestyle='--',
                linewidth=3,
                color=color_scheme[0],
                alpha=0.8,
                zorder=2,
                label='Linear Trend')
    
    ax1.set_title(f'{borough_name} Air Quality Health Impact Trends\nCombined PM2.5',
                 fontsize=16, pad=20, fontweight='bold', color='#2C3E50')
    ax1.set_xlabel('Year', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')
    ax1.set_ylabel('Rate/Count per Year', fontsize=12, labelpad=10, color='#2C3E50', fontweight='bold')
    
    ax1.yaxis.grid(True, linestyle='-', alpha=0.2, linewidth=1)
    ax1.xaxis.grid(False)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    
    # Individual indicators subplot
    ax2 = plt.subplot2grid((4, 3), (2, 0), colspan=2, rowspan=2)
    
    # Plot each indicator separately
    indicator_colors = ['#E74C3C', '#3498DB', '#27AE60', '#8E44AD']
    for i, indicator in enumerate(RESPIRATORY_INDICATORS):
        indicator_stats = calculate_yearly_statistics(df, borough_name, indicator, confidence_level)
        if len(indicator_stats) > 0:
            color = indicator_colors[i % len(indicator_colors)]
            ax2.plot(indicator_stats['Year'], indicator_stats['mean'],
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    color=color,
                    label=indicator.split(' due to')[0],  # Shortened label
                    alpha=0.8)
    
    ax2.set_title('Individual Indicators Breakdown', fontsize=12, fontweight='bold', color='#2C3E50')
    ax2.set_xlabel('Year', fontsize=10, color='#2C3E50')
    ax2.set_ylabel('Rate/Count', fontsize=10, color='#2C3E50')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    
    # Statistics panel
    ax3 = plt.subplot2grid((4, 3), (0, 2), rowspan=2)
    ax3.axis('off')
    
    # Calculate trend statistics
    if len(yearly_stats) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(yearly_stats['Year'], yearly_stats['mean'])
        
        stats_text = f"""STATISTICAL SUMMARY

Overall Trend:
• Slope: {slope:.3f} per year
• R²: {r_value**2:.3f}
• p-value: {p_value:.4f}
• {'Significant' if p_value < 0.05 else 'Not significant'}

Rate Statistics:
• Mean: {yearly_stats['mean'].mean():.2f}
• Std Dev: {yearly_stats['std'].mean():.2f}
• Min Year: {yearly_stats['mean'].min():.2f}
• Max Year: {yearly_stats['mean'].max():.2f}

Data Quality:
• Years: {len(yearly_stats)}
• Indicators: {len(borough_data['Name'].unique())}
• Total Records: {yearly_stats['count'].sum()}"""
    else:
        stats_text = f"""STATISTICAL SUMMARY

Limited Data Available

Rate Statistics:
• Mean: {yearly_stats['mean'].mean():.2f}
• Single Year Data

Data Quality:
• Years: {len(yearly_stats)}
• Indicators: {len(borough_data['Name'].unique())}
• Total Records: {yearly_stats['count'].sum()}"""
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=color_scheme[2]))
    
    # Data breakdown table
    ax4 = plt.subplot2grid((4, 3), (2, 2), rowspan=2)
    ax4.axis('off')
    
    # Show indicator breakdown
    indicator_summary = borough_data.groupby('Name')['data_value_numeric'].agg(['count', 'mean']).round(2)
    
    table_data = []
    for indicator, row in indicator_summary.iterrows():
        short_name = indicator.split(' due to')[0][:20] + "..." if len(indicator.split(' due to')[0]) > 20 else indicator.split(' due to')[0]
        table_data.append([short_name, int(row['count']), f"{row['mean']:.1f}"])
    
    if table_data:
        table = ax4.table(cellText=table_data,
                         colLabels=['Indicator', 'Records', 'Avg Value'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Style table
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor(color_scheme[2])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data) + 1):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F9FA')
                else:
                    table[(i, j)].set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    return yearly_stats


def plot_all_boroughs_comparison_enhanced(df, filename, confidence_level=0.95):
    """Create enhanced comparison chart for all boroughs"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12), dpi=300)
    fig.patch.set_facecolor(COLORS['background'])
    
    nyc_boroughs = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
    all_years = set()
    borough_stats = {}
    
    # Calculate stats for each borough
    for borough in nyc_boroughs:
        borough_data = df[df['Borough'] == borough]
        if len(borough_data) > 0:
            borough_yearly_stats = calculate_yearly_statistics(df, borough, confidence_level=confidence_level)
            if len(borough_yearly_stats) > 0:
                borough_stats[borough] = borough_yearly_stats
                all_years.update(borough_yearly_stats['Year'].tolist())
    
    # Main comparison plot (top-left)
    for borough, borough_yearly_stats in borough_stats.items():
        color = BOROUGH_COLORS[borough]
        
        # Plot confidence intervals
        ax1.fill_between(borough_yearly_stats['Year'], 
                        borough_yearly_stats['ci_lower'], 
                        borough_yearly_stats['ci_upper'],
                        alpha=0.15, 
                        color=color)
        
        # Plot main line
        ax1.plot(borough_yearly_stats['Year'], borough_yearly_stats['mean'],
                marker='o',
                linewidth=3,
                markersize=7,
                color=color,
                label=borough,
                alpha=0.9)
    
    ax1.set_title('NYC Boroughs: Air Quality Health Impact Comparison\nWith 95% Confidence Intervals',
                 fontsize=14, pad=15, fontweight='bold', color='#2C3E50')
    ax1.set_xlabel('Year', fontsize=11, color='#2C3E50', fontweight='bold')
    ax1.set_ylabel('Combined Health Impact Rate', fontsize=11, color='#2C3E50', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Standard deviation comparison (top-right)
    for borough, borough_yearly_stats in borough_stats.items():
        color = BOROUGH_COLORS[borough]
        ax2.plot(borough_yearly_stats['Year'], borough_yearly_stats['std'],
                marker='s',
                linewidth=2,
                markersize=6,
                color=color,
                label=f'{borough} Std Dev',
                alpha=0.8)
    
    ax2.set_title('Standard Deviation by Borough',
                 fontsize=14, pad=15, fontweight='bold', color='#2C3E50')
    ax2.set_xlabel('Year', fontsize=11, color='#2C3E50', fontweight='bold')
    ax2.set_ylabel('Standard Deviation', fontsize=11, color='#2C3E50', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Record count comparison (bottom-left)
    for borough, borough_yearly_stats in borough_stats.items():
        color = BOROUGH_COLORS[borough]
        ax3.plot(borough_yearly_stats['Year'], borough_yearly_stats['count'],
                marker='^',
                linewidth=2,
                markersize=6,
                color=color,
                label=f'{borough} Records',
                alpha=0.8)
    
    ax3.set_title('Number of Records by Borough',
                 fontsize=14, pad=15, fontweight='bold', color='#2C3E50')
    ax3.set_xlabel('Year', fontsize=11, color='#2C3E50', fontweight='bold')
    ax3.set_ylabel('Number of Records', fontsize=11, color='#2C3E50', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Trend comparison (bottom-right)
    trend_data = []
    for borough, borough_yearly_stats in borough_stats.items():
        if len(borough_yearly_stats) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(borough_yearly_stats['Year'], borough_yearly_stats['mean'])
            trend_data.append({
                'Borough': borough,
                'Slope': slope,
                'R_squared': r_value**2,
                'P_value': p_value,
                'Significant': p_value < 0.05
            })
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        
        # Bar plot of slopes
        bars = ax4.bar(range(len(trend_df)), trend_df['Slope'], 
                      color=[BOROUGH_COLORS[b] for b in trend_df['Borough']],
                      alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add significance indicators
        for i, (_, row) in enumerate(trend_df.iterrows()):
            if row['Significant']:
                ax4.text(i, row['Slope'] + (abs(row['Slope']) * 0.1 if row['Slope'] != 0 else 0.1), 
                        f"R²={row['R_squared']:.3f}\np={row['P_value']:.3f}*",
                        ha='center', va='bottom' if row['Slope'] >= 0 else 'top',
                        fontsize=8, fontweight='bold')
            else:
                ax4.text(i, row['Slope'] + (abs(row['Slope']) * 0.1 if row['Slope'] != 0 else 0.1),
                        f"R²={row['R_squared']:.3f}\np={row['P_value']:.3f}",
                        ha='center', va='bottom' if row['Slope'] >= 0 else 'top',
                        fontsize=8)
        
        ax4.set_title('Trend Analysis: Rate of Change by Borough\n(* = statistically significant)',
                     fontsize=14, pad=15, fontweight='bold', color='#2C3E50')
        ax4.set_xlabel('Borough', fontsize=11, color='#2C3E50', fontweight='bold')
        ax4.set_ylabel('Slope (Rate Change per Year)', fontsize=11, color='#2C3E50', fontweight='bold')
        ax4.set_xticks(range(len(trend_df)))
        ax4.set_xticklabels(trend_df['Borough'], rotation=45, ha='right')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
    
    # Style all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.set_facecolor(COLORS['background'])
        if all_years and ax in [ax1, ax2, ax3]:
            sorted_years = sorted(list(all_years))
            ax.set_xticks(sorted_years)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    plt.suptitle('NYC Air Quality Health Impact Comprehensive Analysis\nAll Boroughs Statistical Comparison', 
                fontsize=16, fontweight='bold', color='#2C3E50', y=0.95)
    
    fig.text(0.02, 0.02, 'Source: NYC Health Department Air Quality Data', fontsize=9, color='#7F8C8D')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()


def main():
        # Load and filter air quality data
        print("=== LOADING AIR QUALITY DATA ===")
        df = load_and_filter_data('Air_Quality.csv')
        
        if len(df) == 0:
            print("No data loaded. Please check the file name and content.")
            return
        
        # Add borough mapping
        print("\n=== ADDING BOROUGH MAPPING ===")
        df_with_boroughs = add_borough_column(df)
        
        # Prepare time-based data
        print("\n=== PREPARING TIME DATA ===")
        df_processed = prepare_time_data(df_with_boroughs)
        
        if len(df_processed) == 0:
            print("No valid processed data. Check date and numeric conversions.")
            return
        
        # Get list of boroughs with data
        available_boroughs = df_processed['Borough'].unique()
        print(f"Available boroughs: {list(available_boroughs)}")
        
        # Create enhanced individual borough charts
        print("\n=== CREATING ENHANCED INDIVIDUAL BOROUGH CHARTS ===")
        
        for borough in available_boroughs:
            borough_data = df_processed[df_processed['Borough'] == borough]
            if len(borough_data) > 0:
                print(f"\nProcessing {borough}...")
                print(f"  - {len(borough_data)} data points")
                print(f"  - {len(borough_data['Geo Place Name'].unique())} neighborhoods")
                print(f"  - Years: {sorted(borough_data['Year'].unique())}")
                
                # Create enhanced individual chart
                filename = f"air_quality_{borough.lower().replace(' ', '_')}_analysis.png"
                yearly_stats = plot_individual_borough_enhanced(df_processed, borough, filename)
                
                if yearly_stats is not None and len(yearly_stats) > 0:
                    print(f"✓ Created enhanced chart: {filename}")
                    
                    # Print detailed statistics
                    print(f"\n=== {borough.upper()} DETAILED STATISTICS ===")
                    if len(yearly_stats) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(yearly_stats['Year'], yearly_stats['mean'])
                        print(f"Trend Analysis:")
                        print(f"  - Slope: {slope:.4f} per year ({'increasing' if slope > 0 else 'decreasing'})")
                        print(f"  - R-squared: {r_value**2:.4f}")
                        print(f"  - P-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
                        print(f"  - Standard Error: {std_err:.4f}")
                    else:
                        print("Trend Analysis: Insufficient data for trend calculation")
                    
                    print(f"\nYearly Statistics:")
                    for _, row in yearly_stats.iterrows():
                        print(f"  {int(row['Year'])}: {row['mean']:.2f} ± {row['margin_error']:.2f} (n={int(row['count'])})")
                else:
                    print(f"✗ Could not create chart for {borough}")
        
        # Create enhanced comparison chart
        print("\n=== CREATING ENHANCED COMPARISON CHART ===")
        plot_all_boroughs_comparison_enhanced(df_processed, 'air_quality_nyc_all_boroughs_comprehensive.png')
        print("✓ Created enhanced comparison chart: air_quality_nyc_all_boroughs_comprehensive.png")
        
        # Summary statistics for all boroughs
        print("\n=== OVERALL NYC AIR QUALITY SUMMARY ===")
        overall_stats = df_processed.groupby(['Borough', 'Year'])['data_value_numeric'].agg(['mean', 'count']).reset_index()
        
        for borough in available_boroughs:
            borough_summary = overall_stats[overall_stats['Borough'] == borough]
            if len(borough_summary) > 0:
                avg_rate = borough_summary['mean'].mean()
                total_records = borough_summary['count'].sum()
                years_span = f"{int(borough_summary['Year'].min())}-{int(borough_summary['Year'].max())}"
                print(f"{borough}: Avg Rate = {avg_rate:.2f}, Records = {total_records}, Years = {years_span}")
        
        print("\n=== AIR QUALITY ANALYSIS FILES CREATED ===")
        print("Enhanced Individual Borough Analysis:")
        for borough in available_boroughs:
            enhanced_file = f"air_quality_{borough.lower().replace(' ', '_')}_analysis.png"
            print(f"- {borough}: {enhanced_file}")
        print("- Enhanced Comparison: air_quality_nyc_all_boroughs_comprehensive.png")
        
        print("\n=== ENHANCED ANALYSIS FEATURES ===")
        print("Individual Borough Charts Include:")
        print("• Main trend line with gradient coloring and confidence intervals")
        print("• Standard deviation bands showing data variability")
        print("• Individual indicator breakdown subplot")
        print("• Statistical summary panel with trend analysis")
        print("• Data quality metrics and record counts")
        print("• Linear regression analysis with significance testing")
        
        print("\nComparison Chart Includes:")
        print("• All boroughs overlaid with confidence intervals")
        print("• Standard deviation comparison across years")
        print("• Record count tracking by borough and year")  
        print("• Trend analysis with statistical significance indicators")
        print("• R² values and p-values for each borough trend")
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("All charts have been generated successfully!")
        print("Check the current directory for the generated PNG files.")


if __name__ == "__main__":
        main()