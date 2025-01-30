"""
analysis_helpers.py

Contains utility functions for analyzing re-entries in person data,
generating additional calculated columns, and plotting.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    df['AgeAtEntry'] = ((df[start_date_col] - df[dob_col]).dt.days / 365.25).astype('int')

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


def plot_distributions(dataframe, color=color):
    """
    Creates distribution plots for Ethnicity and PCArea.

    Parameters:
    - dataframe: pd.DataFrame, the dataframe containing the data
    - color: str, optional, color for the plots (default is #702A7D)
    """
    # Ethnicity Distribution
    plt.figure(figsize=(8, 10))
    ax = sns.countplot(
        data=dataframe,
        y="EthnicOrigin",
        order=dataframe["EthnicOrigin"].value_counts().sort_values(ascending=False).index,
        color=color
    )
    ax.set_title("Ethnicity Distribution")
    ax.set_xlabel("Count")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()
    plt.show()

    # PCArea Distribution
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(
        data=dataframe,
        x="PCArea",
        order=dataframe["PCArea"].value_counts().sort_values(ascending=False).index,
        color=color
    )
    ax.set_title("PCArea Distribution")
    ax.set_xlabel("Count")
    ax.tick_params(axis="x", rotation=45)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()
    plt.show()


def plot_age_distribution(
    dataframe,
    intervention_name,
    startdate='StartDate',
    birth_date_column='DateOfBirth',
    color=color,
    grey_color=grey_color
):
    """
    Plots age distribution for a given DataFrame with two subplots:
    1. Continuous age distribution
    2. Age group distribution

    Parameters:
    - dataframe: pd.DataFrame
    - intervention_name: str name of the analysis (e.g., CiC)
    - startdate: str, name of the column containing entry dates
    - birth_date_column: str, name of the column containing birth dates
    - color, grey_color: str color hex codes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Continuous age distribution
    age_at_entry = (dataframe[startdate].dt.year - dataframe[birth_date_column].dt.year).astype(int)
    sns.histplot(age_at_entry, color=color, kde=True, bins=range(0, age_at_entry.max() + 2, 1), ax=ax1)
    ax1.set_title(f"Age Distribution at Entry into {intervention_name}")
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel("Age")

    # 2. Age group distribution
    age_group_counts = dataframe['entry_agegroup'].value_counts()
    sns.countplot(x='entry_agegroup', data=dataframe, ax=ax2, color=grey_color)
    ax2.set_title(f"Age Group Distribution at Entry into {intervention_name}")
    ax2.set_ylabel("Frequency")
    ax2.set_xlabel("Age Group")
    ax2.tick_params(axis='x', rotation=45)
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


def plot_monthly_entries_exits(dataframe, start_date_col, end_date_col, intervention_name, color=color):
    """
    Plots monthly entries, exits, and net change for a given intervention type.

    Parameters:
    - dataframe: pd.DataFrame
    - start_date_col: str
    - end_date_col: str
    - intervention_name: str
    - color: str
    """
    monthly_entries = dataframe[start_date_col].dt.to_period("M").value_counts().sort_index()
    monthly_exits = dataframe[end_date_col].dt.to_period("M").value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), height_ratios=[2, 1])

    # Top subplot: Entries vs. Exits
    ax1.plot(monthly_entries.index.astype(str), monthly_entries.values, marker="o", color=color, label="Entries")
    ax1.plot(monthly_exits.index.astype(str), monthly_exits.values, marker="o", color="red", label="Exits")
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

    ax2.set_title("Net Change (Entries - Exits)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Net Change")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

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
    
def plot_age_groups_for_children_still_in_care(dataframe, intervention_name,
                                               end_date_col,
                                               age_group_col,
                                               color=color):
    """
    Plots the distribution of the specified age group column for children who
    are still in care (i.e., those with a missing end date).

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the relevant columns.
    end_date_col : str, optional
        The column name indicating end date of intervention (default: "EndDate").
    age_group_col : str, optional
        The column name indicating the categorized age group (default: "entry_agegroup").
    color : str, optional
        Bar color for the histogram (default: "#702A7D").
    intervention_name : str, optional
        Name of the intervention, used for plot titles (default: "LAC").
    """
    still_in_care = dataframe.loc[dataframe[end_date_col].isnull()]

    plt.figure(figsize=(7,5))
    sns.histplot(data=still_in_care, x=age_group_col, color=color)
    plt.title(f'Age Group Distribution for Children Still in {intervention_name}', fontsize=10)
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"../figs/{intervention_name}_agedist_still_in_care.png", dpi=300, bbox_inches='tight')
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
    
def plot_top_n_category_distribution(dataframe, intervention_name,
                                     category_col, n=10,
                                     color=color):
    """
    Plots the distribution of the top N categories in the specified column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the relevant columns.
    category_col : str
        The name of the column that specifies categories (e.g., 'CPP_Category', 'Category').
    n : int, optional
        Number of most frequent categories to display (default: 10).
    color : str, optional
        Bar color for the plot (default: "#702A7D").
    intervention_name : str, optional
        Name of the intervention, used for plot titles (default: "LAC").
    """
    # Identify the top N categories
    top_categories = dataframe[category_col].value_counts().nlargest(n).index
    # Filter the DataFrame to include only top N categories
    filtered_df = dataframe[dataframe[category_col].isin(top_categories)]

    plt.figure(figsize=(8,5))
    ax = sns.countplot(data=filtered_df, y=category_col, order=top_categories, color=color)
    ax.set_title(f'Top {n} {category_col} Distribution ({intervention_name})')
    ax.set_ylabel('Category')
    ax.set_xlabel('Count')
    
    # Remove the x-axis labels
    plt.gca().set_xticklabels([])

    # add count labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')

    plt.tight_layout()
    plt.savefig(f"../figs/{intervention_name}_top_{n}_category_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_agegroup_distribution_top_categories(
    dataframe,
    intervention_name,
    category_col='Category',
    agegroup_col='entry_agegroup',
    n=5):
    """
    Plots a bar chart of agegroup percentages, with the top N categories.
    Each age group's categories sum to 100%.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the data.
    category_col : str, optional
        The column name for the category grouping.
    agegroup_col : str, optional
        The column name for the pre-assigned age groups (default 'entry_agegroup').
    n : int, optional
        Number of top categories to display (default 5).
    color_palette : str or list, optional
        A Matplotlib/Seaborn color palette or list of colors for the bars
        (default 'Purples').
    """
    # create pivot table with ALL categories
    full_pivot = dataframe.groupby([agegroup_col, category_col]).size().unstack(fill_value=0)

    #  Calculate percentages based on ALL categories
    full_pivot_pct = full_pivot.div(full_pivot.sum(axis=1), axis=0) * 100

    #  Identify top N categories
    top_categories = dataframe[category_col].value_counts().head(n).index

    # Filter to show only top N categories
    pivot_df_pct = full_pivot_pct[top_categories]

    #  Round percentages to 1 decimal place
    pivot_df_pct = pivot_df_pct.round(1)

    #  Add % symbol to all values
    formatted_df = pivot_df_pct.applymap(lambda x: f"{x}%")

    # If age group is categorical and has a known order, reindex rows to preserve that order
    if hasattr(dataframe[agegroup_col], 'cat') and dataframe[agegroup_col].cat.categories is not None:
        formatted_df = formatted_df.reindex(dataframe[agegroup_col].cat.categories, axis=0)

    # Create color map with evenly spaced colors
    colors = plt.cm.Purples_r(np.linspace(0.3, 0.9, len(top_categories)))  # Added this line

    # Plot it as a bar chart
    ax = pivot_df_pct.plot(
        kind='bar',
        stacked=False,
        figsize=(10, 6),
        color=colors,  # Use our custom colors
        edgecolor='none',
        width=0.8
    )

    # Aesthetic adjustments
    plt.title(f'Category Distribution by Age Group (Top {n} {category_col})', fontsize=14, pad=15)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)

    # Add percentage signs to y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))

    # Place legend to the right
    plt.legend(title=category_col, bbox_to_anchor=(1.01, 1), loc='upper left')

    # Add grid lines for better readability of percentages
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"../figs/{intervention_name}_agegroup_by_category_percent.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()