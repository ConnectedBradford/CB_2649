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