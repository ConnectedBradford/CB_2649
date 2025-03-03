"""
lsoa_analysis_helper.py

Contains function to recreate LAC LSOA analysis for CPP and CiNP interventions.


"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import geopandas as gpd
from scipy import stats
from dateutil.relativedelta import relativedelta
color='#702A7D'

def calculate_intervention_rates(
    intervention_data,
    bradford_boundary,
    imd_data,
    population_data,
    intervention_name,
    intervention_col='LSOA'
):
    """
    Calculate intervention rates per LSOA and merge with IMD and population data.

    Parameters:
    -----------
    intervention_data : pd.DataFrame
        DataFrame containing intervention data with LSOA codes
    bradford_boundary : gpd.GeoDataFrame
        GeoDataFrame containing Bradford boundary data with LSOA codes
    imd_data : pd.DataFrame
        DataFrame containing IMD data
    population_data : pd.DataFrame
        DataFrame containing population data
    intervention_name : str
        Name of the intervention (e.g., 'LAC', 'CPP', 'CIN')
    intervention_col : str, default='LSOA'
        Column name containing LSOA codes in intervention_data

    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with intervention rates and associated data
    """
    # Clean intervention data
    clean_data = intervention_data.dropna(subset=[intervention_col])
    english_lsoa = clean_data[
        clean_data[intervention_col].str.startswith('E01', na=False)
    ].copy()

    # Merge with boundary data
    merged_data = bradford_boundary.merge(
        english_lsoa,
        left_on='lsoa21cd',
        right_on=intervention_col,
        how='left'
    )

    # Drop unwanted columns
    columns_to_drop = ['lsoa21nm', 'label', intervention_col]
    merged_data.drop(columns=columns_to_drop, inplace=True)

    # Calculate counts per LSOA
    cases_per_lsoa = merged_data.groupby('lsoa21cd', as_index=False).agg(
        case_count=('lsoa21cd', 'size'),
        geometry=('geometry', 'first')
    ).pipe(gpd.GeoDataFrame)

    # Merge with IMD data
    imd_columns = [
        'lsoa11cd', 'IMD_Decile', 'IMDScore',
        'IDCDec', 'CYPDec', 'IncDec', 'EmpDec',
        'EduDec', 'CriDec', 'BHSDec', 'EnvDec',
        'HDDDec', 'DepChi'
    ]

    cases_with_imd = cases_per_lsoa.merge(
        imd_data[imd_columns],
        left_on='lsoa21cd',
        right_on='lsoa11cd',
        how='left'
    ).drop(columns='lsoa11cd')

    # Merge with population data and calculate rates
    final_data = cases_with_imd.merge(
        population_data[['LSOA 2021 Code', 'TotalPop']],
        left_on='lsoa21cd',
        right_on='LSOA 2021 Code',
        how='left'
    ).drop(columns='LSOA 2021 Code')

    final_data['rate_per_population'] = (
        final_data['case_count'] /
        final_data['TotalPop']
    ) * 100

    return final_data

def plot_intervention_map(
    data,
    intervention_name,
    rate_column='rate_per_population',
    figsize=(8, 8),
    base_output_path="../figs"
):
    """
    Plot choropleth map of intervention rates.

    Parameters:
    -----------
    data : gpd.GeoDataFrame
        GeoDataFrame containing intervention rates and geometry
    intervention_name : str
        Name of the intervention (e.g., 'LAC', 'CPP', 'CIN')
    rate_column : str, default='rate_per_population'
        Column name containing the rates to plot
    figsize : tuple, default=(8, 8)
        Figure size in inches
    base_output_path : str, default="../figs"
        Base path for saving the figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    data.plot(
        ax=ax,
        column=rate_column,
        scheme='NaturalBreaks',
        cmap='YlOrRd',
        legend=True
    )

    ax.set_axis_off()
    ax.set_title(f'Rate of {intervention_name} per LSOA Population')
    plt.tight_layout()

    output_path = f"{base_output_path}/{intervention_name.lower()}_rate_map.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
  
  
def plot_cumulative_distribution(
    data,
    intervention_name,
    count_column='case_count',
    color=color,
    figsize=(10, 6),
    base_output_path="../figs"
):
    """
    Plot cumulative distribution of intervention cases across LSOAs.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing intervention counts per LSOA
    intervention_name : str
        Name of the intervention (e.g., 'LAC', 'CPP', 'CINP')
    count_column : str, default='case_count'
        Column name containing the case counts
    color : str, default='blue'
        Color for the plot line
    figsize : tuple, default=(10, 6)
        Figure size in inches
    base_output_path : str, default="../figs"
        Base path for saving the figure

    Returns:
    --------
    float
        Percentage of LSOAs that account for 50% of cases
    """
    # Calculate total cases
    total_cases = data[count_column].sum()

    # Sort by case count (descending)
    sorted_data = data.sort_values(by=count_column, ascending=False)

    # Calculate cumulative sum and proportion
    sorted_data['cumulative_cases'] = sorted_data[count_column].cumsum()
    sorted_data['cumulative_percent'] = (
        100.0 * sorted_data['cumulative_cases'] / total_cases
    )

    # Calculate percentage of LSOAs accounting for 50% of cases
    lsoas_for_50_percent = len(sorted_data[sorted_data['cumulative_percent'] <= 50])
    percent_lsoas_for_50 = (lsoas_for_50_percent / len(sorted_data)) * 100

    # Create x-axis representing percentage of LSOAs
    num_lsoas = len(sorted_data)
    x_percent = [(i + 1) / num_lsoas * 100 for i in range(num_lsoas)]

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(
        x_percent,
        sorted_data['cumulative_percent'],
        linestyle='--',
        color=color
    )

    plt.title(f'Cumulative Proportion of {intervention_name} Cases by LSOA')
    plt.xlabel('Percentage of LSOAs (%)')
    plt.ylabel(f'Cumulative Proportion of {intervention_name} Cases (%)')
    plt.grid(True, linestyle='--', alpha=.4)

    output_path = f"{base_output_path}/{intervention_name.lower()}_cumulative_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    return f"{percent_lsoas_for_50:.1f}% of LSOAs account for 50% of {intervention_name} cases"
  


def analyze_imd_relationship(
    data,
    intervention_name,
    rate_column='rate_per_population',
    imd_column='IMD_Decile',
    base_output_path="../figs"
):
    """
    Analyze and visualize the relationship between intervention rates and IMD decile.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing intervention rates and IMD data
    intervention_name : str
        Name of the intervention (e.g., 'LAC', 'CPP', 'CIN')
    rate_column : str, default='rate_per_population'
        Column name containing the intervention rates
    imd_column : str, default='IMD_Decile'
        Column name containing the IMD decile values
    base_output_path : str, default="../figs"
        Base path for saving figures

    Returns:
    --------
    dict
        Dictionary containing statistical results
    """
    # Create scatter plot with regression line
    plt.figure(figsize=(10, 6))

    # Create scatter plot
    sns.scatterplot(
        data=data,
        x=imd_column,
        y=rate_column
    )

    # Add regression line
    sns.regplot(
        data=data,
        x=imd_column,
        y=rate_column,
        scatter=False,
        color='red'
    )

    plt.title(f'{intervention_name} Rate vs. IMD Decile by LSOA')
    plt.xlabel('IMD Decile')
    plt.ylabel(f'{intervention_name} Rate (%)')

    # Save plot
    output_path = f"{base_output_path}/{intervention_name.lower()}_rate_vs_imd_decile.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    # Statistical Analysis
    # Drop rows with NaN
    valid = (
        data[imd_column].notna() &
        data[rate_column].notna()
    )
    clean_data = data[valid]

    # Calculate correlations and regression
    pearson_corr, pearson_p = stats.pearsonr(
        clean_data[imd_column],
        clean_data[rate_column]
    )

    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
        clean_data[imd_column],
        clean_data[rate_column]
    )

    # Print results
    print(f"\nStatistical Analysis Results for {intervention_name}:")
    print(f"Pearson Correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3e})")
    print("\nLinear Regression Results:")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"R-squared: {r_value**2:.3f}")
    print(f"P-value: {p_value_reg:.3e}")
    

def prepare_intervention_data(intervention_data, boundary_data, imd_data, population_data,
                             lsoa_column='LSOA', intervention_name='Intervention'):
    """
    Prepare intervention data for analysis by merging with boundary, IMD, and population data.

    Parameters:
    -----------
    intervention_data : pandas.DataFrame
        DataFrame containing intervention data with LSOA codes
    boundary_data : geopandas.GeoDataFrame
        GeoDataFrame containing boundary data with LSOA codes
    imd_data : pandas.DataFrame
        DataFrame containing IMD data
    population_data : pandas.DataFrame
        DataFrame containing population data
    lsoa_column : str, default='LSOA'
        Name of the column in intervention_data containing LSOA codes
    intervention_name : str, default='Intervention'
        Name of the intervention for labeling

    Returns:
    --------
    tuple
        (merged_data, children_population)
    """
    import geopandas as gpd
    import pandas as pd

    # Make a copy to avoid modifying the original
    intervention_data = intervention_data.copy()

    # Filter to keep only English LSOAs (starting with 'E01') and remove empty strings
    intervention_english_lsoa = intervention_data[intervention_data[lsoa_column].str.startswith('E01', na=False)].copy()

    # Print matching information
    print(f"{intervention_name} boundary unique LSOAs:", len(boundary_data['lsoa21cd'].unique()))
    print(f"{intervention_name} English unique LSOAs:", len(intervention_english_lsoa[lsoa_column].unique()))
    print("Common LSOAs:", len(set(boundary_data['lsoa21cd']).intersection(set(intervention_english_lsoa[lsoa_column]))))

    # Use left merge to preserve geometry
    merged_data = boundary_data.merge(
        intervention_english_lsoa,
        left_on='lsoa21cd',
        right_on=lsoa_column,
        how='left',
    )

    # Drop unwanted columns
    columns_to_drop = ['lsoa21nm', 'label', lsoa_column]
    merged_data.drop(columns=[col for col in columns_to_drop if col in merged_data.columns], inplace=True)

    # Group by LSOA and count the number of children
    children_per_lsoa = merged_data.groupby('lsoa21cd', as_index=False).agg(
        children_count=('lsoa21cd', 'size'),
        geometry=('geometry', 'first')
    ).pipe(gpd.GeoDataFrame)

    # Merge with IMD Data
    columns_to_include_from_imd = [
        'lsoa11cd', 'IMD_Decile', 'IMDScore',
        'IDCDec', 'CYPDec', 'IncDec', 'EmpDec',
        'EduDec', 'CriDec', 'BHSDec', 'EnvDec',
        'HDDDec', 'DepChi'
    ]

    children_per_lsoa = children_per_lsoa.merge(
        imd_data[columns_to_include_from_imd],
        left_on='lsoa21cd',
        right_on='lsoa11cd',
        how='left'
    ).drop(columns='lsoa11cd')

    # Merge with population data
    children_population = children_per_lsoa.merge(
        population_data[['LSOA 2021 Code', 'TotalPop']],
        left_on='lsoa21cd',
        right_on='LSOA 2021 Code',
        how='left'
    ).drop(columns='LSOA 2021 Code')

    # Calculate children proportion
    children_population['children_per_total_pop'] = (
        children_population['children_count'] /
        children_population['TotalPop']) * 100

    return merged_data, children_population


def analyze_age_distribution(merged_data, children_population, intervention_name='Intervention',
                           percentile_thresholds=[90, 80], age_column='AgeAtEntry'):
    """
    Analyze and visualize age distribution for high-intervention vs other LSOAs.

    Parameters:
    -----------
    merged_data : pandas.DataFrame
        DataFrame containing merged intervention and demographic data
    children_population : pandas.DataFrame
        DataFrame containing population data with intervention rates
    intervention_name : str, default='Intervention'
        Name of the intervention for labeling
    percentile_thresholds : list, default=[90, 80]
        Percentile thresholds to use for comparison
    age_column : str, default='age_at_entry'
        Name of the column containing age data

    Returns:
    --------
    None
        Displays plots and prints statistics
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Check if age_column exists
    if age_column not in merged_data.columns:
        raise ValueError(f"Column {age_column} not found in merged_data. Please calculate age first using calculate_age_at_entry().")

    # Create a figure with two subplots side by side to compare percentile thresholds
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Define bin edges for ages 0-17
    bins = np.arange(0, 19) - 0.5  # Creates bin edges: [-0.5, 0.5, 1.5, ..., 18.5]
    x = np.arange(18)
    width = 1.0  # Full width to ensure bars touch

    # Process each percentile threshold
    for i, percentile_threshold in enumerate(percentile_thresholds[:2]):  # Limit to first two thresholds
        ax = axes[i]

        # Calculate cutoff value
        cutoff_value = np.percentile(children_population['children_per_total_pop'], percentile_threshold)

        # Select LSOAs above the percentile threshold
        high_intervention_lsoas = children_population[
            children_population['children_per_total_pop'] >= cutoff_value
        ]['lsoa21cd'].tolist()

        # Split the data into high rate and other groups
        high_group = merged_data[merged_data['lsoa21cd'].isin(high_intervention_lsoas)]
        other_group = merged_data[~merged_data['lsoa21cd'].isin(high_intervention_lsoas)]

        # Calculate histograms using counts
        high_counts, _ = np.histogram(high_group[age_column], bins=bins)
        other_counts, _ = np.histogram(other_group[age_column], bins=bins)

        # Convert counts to percentages
        high_pct = high_counts / high_counts.sum() * 100 if high_counts.sum() > 0 else np.zeros_like(high_counts)
        other_pct = other_counts / other_counts.sum() * 100 if other_counts.sum() > 0 else np.zeros_like(other_counts)

        # Plot the histogram
        ax.bar(x, high_pct, width=width, align='center', color='red', alpha=0.7,
              edgecolor='black', linewidth=1.0,
              label=f'>= {percentile_threshold}th Percentile')
        ax.bar(x, -other_pct, width=width, align='center', color='blue', alpha=0.7,
              edgecolor='black', linewidth=1.0,
              label=f'< {percentile_threshold}th Percentile')

        # Format the subplot
        ax.set_ylabel('Percentage of Children (%)')
        ax.set_xlabel('Age')
        ax.set_title(f'Age Distribution - {percentile_threshold}th Percentile Threshold', fontsize=14, pad=20)
        ax.set_xlim(-0.5, 17.5)
        ax.set_xticks(x)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.legend(loc='upper right')

        # Print statistics
        print(f"{percentile_threshold}th percentile cutoff value: {cutoff_value:.4f}")
        print(f"Number of LSOAs above {percentile_threshold}th percentile: {len(high_intervention_lsoas)}")
        print(f"Number of children in high group ({percentile_threshold}th): {len(high_group)}")
        print(f"Number of children in other group ({percentile_threshold}th): {len(other_group)}")
        print("\n")

    # Add an overall title
    fig.suptitle(f'Comparison of Age Distributions for {intervention_name} Using Different Percentile Thresholds',
                fontsize=16, y=1.05)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def analyze_ethnicity_distribution(merged_data, children_population, intervention_name='Intervention',
                                 percentile_thresholds=[90, 80], ethnicity_column='EthnicOrigin'):
    """
    Analyze and visualize ethnic origin distribution for high-intervention vs other LSOAs.

    Parameters:
    -----------
    merged_data : pandas.DataFrame
        DataFrame containing merged intervention and demographic data
    children_population : pandas.DataFrame
        DataFrame containing population data with intervention rates
    intervention_name : str, default='Intervention'
        Name of the intervention for labeling
    percentile_thresholds : list, default=[90, 80]
        Percentile thresholds to use for comparison
    ethnicity_column : str, default='EthnicOrigin'
        Name of the column containing ethnicity data

    Returns:
    --------
    None
        Displays plots and prints statistics
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Check if ethnicity_column exists
    if ethnicity_column not in merged_data.columns:
        raise ValueError(f"Column {ethnicity_column} not found in merged_data")

    # Create a figure with two subplots side by side to compare percentile thresholds
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Process each percentile threshold
    for i, percentile_threshold in enumerate(percentile_thresholds[:2]):  # Limit to first two thresholds
        ax = axes[i]

        # Calculate cutoff value
        cutoff_value = np.percentile(children_population['children_per_total_pop'], percentile_threshold)

        # Select LSOAs above the percentile threshold
        high_intervention_lsoas = children_population[
            children_population['children_per_total_pop'] >= cutoff_value
        ]['lsoa21cd'].tolist()

        # Split the data into high rate and other groups
        high_group = merged_data[merged_data['lsoa21cd'].isin(high_intervention_lsoas)]
        other_group = merged_data[~merged_data['lsoa21cd'].isin(high_intervention_lsoas)]

        # Get the counts for each ethnic origin category
        high_ethnicity_counts = high_group[ethnicity_column].value_counts()
        other_ethnicity_counts = other_group[ethnicity_column].value_counts()

        # Convert to percentages
        high_ethnicity_pct = high_ethnicity_counts / high_ethnicity_counts.sum() * 100 if high_ethnicity_counts.sum() > 0 else high_ethnicity_counts * 0
        other_ethnicity_pct = other_ethnicity_counts / other_ethnicity_counts.sum() * 100 if other_ethnicity_counts.sum() > 0 else other_ethnicity_counts * 0

        # Combine the categories to ensure both groups have the same categories
        all_categories = sorted(set(high_ethnicity_pct.index) | set(other_ethnicity_pct.index))

        # Reindex to include all categories, filling missing values with 0
        high_ethnicity_pct = high_ethnicity_pct.reindex(all_categories, fill_value=0)
        other_ethnicity_pct = other_ethnicity_pct.reindex(all_categories, fill_value=0)

        # Sort categories by total percentage (high + other) in descending order
        total_pct = high_ethnicity_pct + other_ethnicity_pct
        sorted_categories = total_pct.sort_values(ascending=True).index  # Ascending for bottom-to-top

        # Reindex according to sorted order
        high_ethnicity_pct = high_ethnicity_pct.reindex(sorted_categories)
        other_ethnicity_pct = other_ethnicity_pct.reindex(sorted_categories)

        # Create horizontal bars
        y_pos = np.arange(len(sorted_categories))
        width = 0.8  # Bar width

        ax.barh(y_pos, high_ethnicity_pct, height=width, color='red', alpha=0.7,
               edgecolor='black', linewidth=1.0, label=f'>= {percentile_threshold}th Percentile')
        ax.barh(y_pos, -other_ethnicity_pct, height=width, color='blue', alpha=0.7,
               edgecolor='black', linewidth=1.0, label=f'< {percentile_threshold}th Percentile')

        # Format the subplot
        ax.set_xlabel('Percentage of Children (%)')
        ax.set_ylabel('Ethnic Origin')
        ax.set_title(f'Ethnic Origin Distribution - {percentile_threshold}th Percentile Threshold',
                    fontsize=14, pad=20)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_categories)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')

        # Print statistics
        print(f"{percentile_threshold}th percentile cutoff value: {cutoff_value:.4f}")
        print(f"Number of LSOAs above {percentile_threshold}th percentile: {len(high_intervention_lsoas)}")
        print(f"Number of children in high group ({percentile_threshold}th): {len(high_group)}")
        print(f"Number of children in other group ({percentile_threshold}th): {len(other_group)}")
        print("\n")

    # Add an overall title
    fig.suptitle(f'Comparison of Ethnic Origin Distributions for {intervention_name} Using Different Percentile Thresholds',
                fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def analyze_imd_domains(children_population, percentile_threshold=90, intervention_name='Intervention'):
    """
    Analyze and visualize IMD domain deciles for high-intervention vs other LSOAs.

    Parameters:
    -----------
    children_population : pandas.DataFrame
        DataFrame containing population data with intervention rates and IMD data
    percentile_threshold : int, default=90
        Percentile threshold to use for comparison
    intervention_name : str, default='Intervention'
        Name of the intervention for labeling

    Returns:
    --------
    None
        Displays plots
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Calculate the intervention rate threshold for the specified percentile
    cutoff_value = np.percentile(children_population['children_per_total_pop'], percentile_threshold)

    # Identify high-intervention LSOAs
    children_population['high_intervention'] = children_population['children_per_total_pop'] >= cutoff_value

    # Split into high and other groups
    high_lsoa_group = children_population[children_population['high_intervention']]
    other_lsoa_group = children_population[~children_population['high_intervention']]

    # List of IMD domains to analyze (using the deciles)
    imd_domains = ['IMD_Decile', 'IDCDec', 'CYPDec', 'IncDec', 'EmpDec',
                  'EduDec', 'CriDec', 'BHSDec', 'EnvDec', 'HDDDec']

    # Check which domains are available in the data
    available_domains = [domain for domain in imd_domains if domain in children_population.columns]

    if not available_domains:
        raise ValueError("None of the specified IMD domains are available in the data")

    # Create a dictionary to map domain codes to more readable names
    domain_names = {
        'IMD_Decile': 'Overall IMD',
        'IDCDec': 'Income Deprivation Affecting Children',
        'CYPDec': 'Children & Young People',
        'IncDec': 'Income',
        'EmpDec': 'Employment',
        'EduDec': 'Education',
        'CriDec': 'Crime',
        'BHSDec': 'Barriers to Housing & Services',
        'EnvDec': 'Living Environment',
        'HDDDec': 'Health Deprivation & Disability'
    }

    # Calculate number of rows and columns for subplots
    n_domains = len(available_domains)
    n_cols = min(5, n_domains)
    n_rows = (n_domains + n_cols - 1) // n_cols  # Ceiling division

    # Create a figure with multiple subplots - one for each domain
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    if n_domains > 1:
        axes = axes.flatten()  # Flatten to make indexing easier
    else:
        axes = [axes]  # Make it iterable for a single subplot

    fig.suptitle(f'IMD Domain Deciles: High Intervention vs. Other LSOAs ({percentile_threshold}th Percentile) - {intervention_name}',
                fontsize=16, y=0.95)

    # For each domain, create a vertical mirrored histogram
    for i, domain in enumerate(available_domains):
        ax = axes[i]

        # Calculate the distribution of deciles for high and other LSOAs
        high_counts = high_lsoa_group[domain].value_counts().sort_index()
        other_counts = other_lsoa_group[domain].value_counts().sort_index()

        # Convert to percentages
        high_pct = high_counts / high_counts.sum() * 100 if high_counts.sum() > 0 else high_counts * 0
        other_pct = other_counts / other_counts.sum() * 100 if other_counts.sum() > 0 else other_counts * 0

        # Ensure all deciles (1-10) are represented
        all_deciles = range(1, 11)
        high_pct = high_pct.reindex(all_deciles, fill_value=0)
        other_pct = other_pct.reindex(all_deciles, fill_value=0)

        # Create vertical bars with width=1.0 to ensure they touch
        x_pos = np.arange(0.5, 10.5)  # Position bars at 0.5, 1.5, ..., 9.5
        width = 1.0  # Full width to ensure bars touch

        # Plot vertical bars
        ax.bar(x_pos, high_pct, width=width, color='red', alpha=0.7,
              edgecolor='black', linewidth=1.0,
              label=f'>= {percentile_threshold}th Percentile', align='center')
        ax.bar(x_pos, -other_pct, width=width, color='blue', alpha=0.7,
              edgecolor='black', linewidth=1.0,
              label=f'< {percentile_threshold}th Percentile', align='center')

        # Add labels and formatting
        ax.set_ylabel('Percentage of LSOAs (%)')
        ax.set_xlabel('Decile (1=most deprived)')
        ax.set_title(f'{domain_names.get(domain, domain)}', fontsize=12, pad=10)

        # Set x-ticks at the center of each bar
        ax.set_xticks(x_pos)
        ax.set_xticklabels(range(1, 11))

        # Set x-limits to ensure bars are fully visible
        ax.set_xlim(0, 10)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Set y-axis limits to be symmetric
        y_max = max(high_pct.max(), other_pct.max()) * 1.1
        ax.set_ylim(-y_max, y_max)

        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc='upper right')

    # Hide any unused subplots
    for i in range(n_domains, len(axes)):
        axes[i].set_visible(False)

    # Add note about deciles
    fig.text(0.5, 0.02,
            'Note: Decile 1 represents the most deprived 10% of LSOAs, Decile 10 represents the least deprived 10%.',
            ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()