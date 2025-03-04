"""
analysis_helpers.py

Contains utility functions for analyzing re-entries in person data,
generating additional calculated columns, and plotting.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

# Project color definitions:
color = '#702A7D'
grey_color = '#A9A9A9'

def analyse_person_ids(dataframe, id_column):
    """
    Analyzes the number of unique IDs and checks for duplicate
    entries in the specified column.

    Parameters:
    - dataframe: pd.DataFrame, the dataframe containing the data
    - id_column: str, the name of the column containing person IDs

    Returns:
    - pd.DataFrame: A dataframe containing duplicate person IDs
      and their details (top 5)
    """
    unique_ids = dataframe[id_column].nunique()
    print(f"Number of Unique IDs: {unique_ids}")
    duplicate_count = dataframe.duplicated(id_column).sum()
    print(f"Number of people that re-entered: {duplicate_count} \n")

    duplicate_entries = dataframe.loc[dataframe.duplicated(id_column, keep=False)]
    print("\nTop 10 Persons with re-entries:")
    print(duplicate_entries[id_column].value_counts().head(10))

    return duplicate_entries.head()


def add_calculated_columns(
    dataframe,
    dob_col='DateOfBirth',
    start_date_col='StartDate',
    end_date_col='EndDate'
):
    """
    Adds calculated columns to dataframe for age and duration analysis.
    Handles UTC timezone in DateOfBirth column.

    Parameters:
    - dataframe: DataFrame containing the data
    - dob_col: Column name for date of birth (default: 'DateOfBirth')
    - start_date_col: Column name for start date (default: 'StartDate')
    - end_date_col: Column name for end date (default: 'EndDate')

    Returns:
    - DataFrame with added columns:
      - AgeAtEntry: Age when entering intervention (years)
      - num_of_days_in_intervention: Duration of intervention (days)
      - entry_agegroup: Categorized age groups
    """
    df = dataframe.copy()
    # Convert DateOfBirth from UTC to timezone-naive
    df[dob_col] = df[dob_col].dt.tz_localize(None)

    # Calculate age at entry (in years)
    df['AgeAtEntry'] = df.apply(
    lambda row: relativedelta(row[start_date_col], row[dob_col]).years,
    axis=1).astype('int')

    # Calculate duration in intervention
    df['num_of_days_in_intervention'] = (df[end_date_col] - df[start_date_col]).dt.days
    df['num_of_days_in_intervention'] = df['num_of_days_in_intervention'].astype('Int64')

    # Create age groups
    bins = [0, 1, 4, 9, 15, 16]
    labels = ['Under 1', '1-4', '5-9', '10-15', '16+']
    df['entry_agegroup'] = pd.cut(
        df['AgeAtEntry'],
        bins=bins,
        labels=labels,
        right=True
    )
    return df


def plot_distributions(dataframe, intervention_name, color=color):
    """
    Creates distribution plots for Gender, Ethnicity and PCArea.

    Parameters:
    - dataframe: pd.DataFrame, the dataframe containing the data
    - color: str, optional, color for the plots (default is #702A7D)
    """
    # Gender Distribution
    plt.figure(figsize=(5,4))
    ax = sns.countplot(data=dataframe, 
                       x='Gender',
                       color=color)
    ax.set_title(f'{intervention_name} Gender Distribution')
    ax.set_xlabel('Gender')

    # Add counts to the bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')

    plt.tight_layout()
    plt.show()
    
    # Ethnicity Distribution
    plt.figure(figsize=(8, 10))
    
    # Calculate percentages
    ethnicity_counts = dataframe['EthnicOrigin'].value_counts(normalize=True) * 100
    ethnicity_order = ethnicity_counts.sort_values(ascending=False).index[:12]
    
    # Filter to top 12 ethnicities
    top_ethnicity_counts = ethnicity_counts[ethnicity_order]
    
    ax = sns.barplot(
    x=top_ethnicity_counts.values,
    y=ethnicity_order,
    color=color)

    ax.set_title(f'{intervention_name} Ethnicity Distribution - Top 12')
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel(None)
    
    # Add percentage labels to the bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(
            width + 0.3,
            p.get_y() + p.get_height()/2,
            f'{width:.1f}%',  
            ha='left',
            va='center',
            fontsize=10)
    plt.tight_layout()
    plt.savefig(f'../figs/{intervention_name}_Ethinicity_Dist.png', dpi=300)
    plt.show()

    # PCArea Distribution
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(
        data=dataframe,
        x="PCArea",
        order=dataframe["PCArea"].value_counts().sort_values(ascending=False).index,
        color=color
    )
    ax.set_title(f'{intervention_name} PCArea Distribution')
    ax.set_xlabel("Count")
    ax.tick_params(axis="x", rotation=45)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()
    plt.show()


def plot_age_distribution(
    dataframe,
    intervention_name,
    age_group_column='entry_agegroup',
    color=color,
    show_percentages=True
):
    """
    Plots age group distribution as percentages with labels on each bar.

    Parameters:
    - dataframe: pd.DataFrame
    - intervention_name: str name of the analysis (e.g., CiC, CPP)
    - age_group_column: str, name of the column containing age group categories
    - color: str color hex code
    - show_percentages: bool, whether to show percentage labels on bars
    - save_fig: bool, whether to save the figure
    - output_path: str, path to save the figure (if save_fig is True)
    """
    # Calculate percentages
    age_counts = dataframe[age_group_column].value_counts(normalize=True).sort_index() * 100

    # Create the plot
    plt.figure(figsize=(6,4))
    ax = sns.barplot(x=age_counts.index, y=age_counts.values, color=color)

    # Add percentage labels on each bar if requested
    if show_percentages:
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 0.5,
                    f'{height:.1f}%',
                    ha="center", fontsize=10)

    # Set titles and labels
    plt.title(f'Age Group Distribution in {intervention_name}', fontsize=14, pad=20)
    plt.xlabel('Entry Age Group', fontsize=12)
    plt.ylabel('Percentage of Cases', fontsize=12)

    # Adjust y-axis to make room for labels
    plt.ylim(0, max(age_counts.values) * 1.15)
    plt.tight_layout()
    plt.show()

def plot_monthly_trends(dataframe, intervention_name, date_column='StartDate', color=color, window=6):
    """
    Plots monthly trends and percentage changes for a given DataFrame.

    Parameters:
    - dataframe: pd.DataFrame
    - intervention_name: str, name of the intervention type (e.g., CPP)
    - date_column: str, name of the column containing dates
    - color: str, optional, color for the plots
    - window: int, optional, window size for moving average
    """
    dataframe["YearMonth"] = dataframe[date_column].dt.to_period("M")
    monthly_trend = dataframe.groupby("YearMonth").size()
    monthly_trend.index = monthly_trend.index.astype(str)

    # Plot 1: Monthly Trend with Moving Average
    plt.figure(figsize=(13, 6))
    plt.plot(monthly_trend.index, monthly_trend.values, marker="o", color=color, label="Monthly Entries")

    moving_average = monthly_trend.rolling(window=window).mean()
    plt.plot(monthly_trend.index, moving_average, color="orange", linewidth=2, label=f"{window}-Month MA")

    plt.title(f"Monthly Trend of entry into {intervention_name}\n"
              f"({dataframe[date_column].min().date()} to {dataframe[date_column].max().date()})")
    plt.xlabel("Month")
    plt.ylabel("Number of Entries")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 2: Percentage Change
    percentage_change = monthly_trend.pct_change() * 100
    plt.figure(figsize=(13, 6))
    plt.plot(percentage_change.index, percentage_change.values, marker="o", color=color, label="MoM % Change")

    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Percentage Change in Monthly Entries into {intervention_name}\n"
              f"({dataframe[date_column].min().date()} to {dataframe[date_column].max().date()})")
    plt.xlabel("Month")
    plt.ylabel("Percentage Change (%)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_monthly_entries_exits(dataframe, start_date_col, end_date_col, intervention_name, color=color, exclude_last_month=True):
    """
    Plots monthly entries, exits, and net change for a given intervention type.

    Parameters:
    - dataframe: pd.DataFrame
    - start_date_col: str
    - end_date_col: str
    - intervention_name: str
    - color: str
    - exclude_last_month: bool, whether to exclude the last month from analysis
    - save_fig: bool, whether to save the figure
    - output_path: str, path to save the figure (if save_fig is True)
    """
    # Count entry cases per month
    monthly_entries = dataframe[start_date_col].dt.to_period("M").value_counts().sort_index()

    # Count exit cases per month
    monthly_exits = dataframe[end_date_col].dt.to_period("M").value_counts().sort_index()

    # Exclude the last month if requested
    if exclude_last_month and len(monthly_entries) > 0:
        last_month = monthly_entries.index[-1]
        monthly_entries = monthly_entries[monthly_entries.index != last_month]

        # Also exclude from exits if it exists there
        if last_month in monthly_exits.index:
            monthly_exits = monthly_exits[monthly_exits.index != last_month]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), height_ratios=[2, 1])

    # Top subplot: Entries vs. Exits
    ax1.plot(monthly_entries.index.astype(str), monthly_entries.values, marker="o", color=color, label="Entries")
    ax1.plot(monthly_exits.index.astype(str), monthly_exits.values, marker="o", color="red", label="Exits")

    # Adjust title based on whether last month is excluded
    ax1.set_title(f"Monthly Trend of {intervention_name} Entries & Exits\n"
                  f"({dataframe[start_date_col].min().date()} to {dataframe[start_date_col].max().date()})")
    ax1.set_ylabel("Number of Cases")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: Net Change
    # Align them on the same index
    all_months = monthly_entries.index.union(monthly_exits.index).sort_values()
    monthly_entries_aligned = monthly_entries.reindex(all_months, fill_value=0)
    monthly_exits_aligned = monthly_exits.reindex(all_months, fill_value=0)
    difference = monthly_entries_aligned - monthly_exits_aligned

    bars = ax2.bar(difference.index.astype(str), difference.values, color="green", alpha=0.6)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    for bar, value in zip(bars, difference.values):
        bar.set_color("red" if value < 0 else "green")

    ax2.set_title(f"Net Change (Entries - Exits)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Net Change")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(f'../figs/{intervention_name}_monthly_entries_exits', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Print summary statistics
   
    print(f"\nSummary of Net Changes for {intervention_name}:")
    print(f"Average monthly net change: {difference.mean():.1f}")
    print(f"Maximum increase: {difference.max():.0f}")
    print(f"Maximum decrease: {difference.min():.0f}")
    print(f"Months with net increase: {(difference > 0).sum()}")
    print(f"Months with net decrease: {(difference < 0).sum()}")


def plot_running_total_in_intervention(dataframe, start_date_col, end_date_col, intervention_name, color=color):
    """
    Plots the running total of individuals in the intervention over time.

    Parameters:
    - dataframe: pd.DataFrame
    - start_date_col: str
    - end_date_col: str
    - intervention_name: str
    - color: str
    """
    monthly_entries = dataframe[start_date_col].dt.to_period("M").value_counts().sort_index()
    monthly_exits = dataframe[end_date_col].dt.to_period("M").value_counts().sort_index()

    # Align and calculate running total
    all_months = monthly_entries.index.union(monthly_exits.index).sort_values()
    monthly_entries_aligned = monthly_entries.reindex(all_months, fill_value=0)
    monthly_exits_aligned = monthly_exits.reindex(all_months, fill_value=0)
    difference = monthly_entries_aligned - monthly_exits_aligned
    cumulative_in_intervention = difference.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_in_intervention.index.astype(str), cumulative_in_intervention.values,
             color=color, linewidth=2)

    # Markers for start and end
    start_date = cumulative_in_intervention.index[0]
    end_date = cumulative_in_intervention.index[-1]
    start_value = cumulative_in_intervention.values[0]
    end_value = cumulative_in_intervention.values[-1]

    plt.scatter([str(start_date), str(end_date)], [start_value, end_value], color=color, s=100, zorder=5)
    plt.annotate(f'{int(start_value)}',
                 xy=(str(start_date), start_value),
                 xytext=(-10, 10),
                 textcoords='offset points',
                 ha='right', va='bottom', color=color)
    plt.annotate(f'{int(end_value)}',
                 xy=(str(end_date), end_value),
                 xytext=(10, 10),
                 textcoords='offset points',
                 ha='left', va='bottom', color=color)

    plt.title(f'Running Total of Individuals In {intervention_name}', pad=20)
    plt.xlabel('Month')
    plt.ylabel('Cumulative Count')
    plt.grid(True, alpha=0.3)

    # Show every 6th month label
    all_xticks = plt.gca().get_xticks()
    plt.gca().set_xticks(all_xticks[::6])
    plt.xticks(rotation=45)

    sns.despine(left=True, right=True, top=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_duration_in_intervention(dataframe, start_date_col, end_date_col, intervention_name, color=color):
    """
    Plots the distribution of intervention durations in bins of 30 days,
    highlighting the highest frequency bin.

    Parameters:
    - dataframe: DataFrame
    - start_date_col: str
    - end_date_col: str
    - intervention_name: str
    - color: str
    """
    # Calculate duration
    dataframe['num_of_days_in_intervention'] = (dataframe[end_date_col] - dataframe[start_date_col]).dt.days
    dataframe['num_of_days_in_intervention'] = dataframe['num_of_days_in_intervention'].astype('Int64')

    grey_color = '#A9A9A9'
    bin_edges = np.arange(0, dataframe['num_of_days_in_intervention'].dropna().max() + 30, 30)

    plt.figure(figsize=(14, 6))
    n, bins, patches = plt.hist(dataframe['num_of_days_in_intervention'].dropna(),
                                bins=bin_edges,
                                color=grey_color,
                                edgecolor='black',
                                alpha=0.7)

    # Identify bin with highest frequency
    max_count_idx = np.argmax(n)
    patches[max_count_idx].set_facecolor(color)
    patches[max_count_idx].set_edgecolor(color)
    patches[max_count_idx].set_alpha(0.9)

    bin_start = bins[max_count_idx]
    bin_end = bins[max_count_idx + 1]
    plt.text(bin_start + 15, n[max_count_idx] + 5,
             f'Between {int(bin_start)}-{int(bin_end)} days',
             ha='center', va='bottom', fontsize=12, color=color)

    plt.title(f'Distribution of {intervention_name} Durations (30 days bin)', fontsize=14)
    plt.xlabel('Number of Days', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    sns.despine(left=True, right=True, top=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_average_duration_by_age(dataframe, intervention_name, color=color):
    """
    Plots the average duration of intervention by age group.

    Parameters:
    - dataframe: DataFrame containing 'entry_agegroup' and 'num_of_days_in_intervention'
    - intervention_name: str
    - color: str
    """
    plt.figure(figsize=(8, 5))
    sns.pointplot(x='entry_agegroup', y='num_of_days_in_intervention',
                  data=dataframe, color=color)
    plt.title(f'Average Duration in {intervention_name} by Entry Age Group', fontsize=14, pad=20)
    plt.xlabel('Entry Age Group', fontsize=12)
    plt.ylabel(f'Average Duration in {intervention_name} (days)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_median_duration_by_age(dataframe, intervention_name, color=color):
    """
    Plots the median duration of intervention by age groups.

    Parameters:
    - dataframe: DataFrame containing the calculated columns (num_of_days_in_intervention, entry_agegroup)
    - intervention_name: Name of the intervention for plot titles
    - color: Color for the plot
    """
    # Calculate medians and confidence intervals for each age group
    medians = dataframe.groupby('entry_agegroup')['num_of_days_in_intervention'].median().reset_index()

    plt.figure(figsize=(10, 6))

    # Create the plot
    plt.plot(medians.index, medians['num_of_days_in_intervention'],
            'o-', color=color, markersize=8)

    # Add error bars using quantiles if desired
    q25 = dataframe.groupby('entry_agegroup')['num_of_days_in_intervention'].quantile(0.25)
    q75 = dataframe.groupby('entry_agegroup')['num_of_days_in_intervention'].quantile(0.75)
    plt.fill_between(medians.index,
                     q25,
                     q75,
                     alpha=0.2,
                     color=color)

    # Customize the plot
    plt.title(f'Median Duration in {intervention_name} by Entry Age Group',
             fontsize=14,
             pad=20)
    plt.xlabel('Entry Age Group', fontsize=12)
    plt.ylabel(f'Median Duration in {intervention_name} (days)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Set x-ticks to age group labels
    plt.xticks(medians.index, medians['entry_agegroup'])

    # Add median values on top of each point
    for idx, median in enumerate(medians['num_of_days_in_intervention']):
        plt.text(idx, median, f'{median:.0f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'../figs/{intervention_name}_median_duration_by_age.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
def plot_median_intervention_duration_over_time(dataframe, intervention_name, end_date_col,
                                                duration_col,
                                                time_freq='Y',
                                                color=color):
    """
    Plots the median intervention duration over time (resampled by the specified frequency).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the relevant columns.
    end_date_col : str, optional
        The column name indicating end date of intervention (default: "EndDate").
    duration_col : str, optional
        The column name that stores the duration of intervention in days (default: "num_of_days_in_intervention").
    time_freq : str, optional
        The frequency for resampling (default: 'Y' for yearly). Common options include
        'M' (monthly), 'Q' (quarterly), etc.
    color : str, optional
        Line color for the plot (default: "#702A7D").
    intervention_name : str, optional
        Name of the intervention, used for plot titles (default: "LAC").
    """

    # Resample by given frequency and compute median
    time_series = (dataframe.set_index(end_date_col)[duration_col]
                              .resample(time_freq)
                              .median())
    plt.figure(figsize=(8,4))
    time_series.plot(color=color)

    plt.title(f'Median Intervention Duration Over Time ({intervention_name})')
    plt.xlabel('Date')
    plt.ylabel('Median Duration (days)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"../figs/{intervention_name}_median_duration_over_time.png", dpi=300, bbox_inches='tight')
    plt.show()
    
