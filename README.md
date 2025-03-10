# CB_2649 - LAC II 2024/2025

## Table of Contents
1. [Overview](##overview)
2. [Project Structure](#project-structure)
3. [Data Sources](#data-sources)
4. [Reproduction Steps](#reproduction-steps)
      - [Environment Setup](#environment-setup)
      - [Running the Analysis](#running-the-analysis)
    - [Demographic Analysis](#demographic-analysis)
    - [LSOA-Level Analysis](#lsoa-level-analysis)
    - [Comprehensive Assessment](#comprehensive-assessment)
5. [Key Modules](#key-modules)
6. [Results](#results)
7. [License](#license)

## Overview
This project analyzes three interventions:
- Child Protection Plans (CPP)
- Looked After Children (LAC)
- Children in Need Plans (CINP)

The analysis examines both demographic patterns and geographical distribution at Lower Super Output Area (LSOA) level.

## Project Structure
```
.
├── code/               # Reusable modules for analysis
├── data/               # Input datasets
├── docs/               # Documentation
├── figs/               # Generated visualizations
├── notebooks/          # Analysis notebooks     
└── requirements.txt    # Dependencies
```

## Data Sources
- Bradford boundary data (LSOA level) - [LINK](https://borders.ukdataservice.ac.uk/)
- English Index of Multiple Deprivation (IMD) 2019 - [LINK](https://data.cdrc.ac.uk/dataset/index-multiple-deprivation-imd)
- Bradford children population data (0-17, 2021) - [LINK](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates)
- Children's social care data (CPP, LAC, CINP) - Connected Bradford

## Reproduction Steps

### Environment Setup
1. Clone this repository and navigate to the project folder:

```bash
git clone [repository-url]
cd [project-folder-name]
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Analysis

#### Demographic Analysis
The demographic analysis workflow:
1. Run the initial CPP analysis:
   ```bash
   jupyter notebook notebooks/cpp_demographic_analysis.ipynb
   ```
2. The LAC and CINP demographic analyses reuse modules from the CPP analysis:
   ```bash
   jupyter notebook notebooks/lac_demographic_analysis.ipynb
   jupyter notebook notebooks/cinp_demographic_analysis.ipynb
   ```

#### LSOA-Level Analysis
The LSOA analysis workflow:
1. Run the initial LAC analysis:
   ```bash
   `jupyter notebook notebooks/lac_lsoa_analysis.ipynb`
   ```
2. The CPP and CINP LSOA analyses reuse modules from the LAC analysis:
   ```bash
   jupyter notebook notebooks/cpp_lsoa_analysis.ipynb
   jupyter notebook notebooks/cinp_lsoa_analysis.ipynb
   ```

#### Comprehensive Assessment
For a combined analysis of assessment across interventions:
```bash
jupyter notebook notebooks/assessment_analysis.ipynb
```

## Key Modules
- `data_cleaning.py`: Data preparation and standardization
- `analysis_helpers.py`: Analysis functions for demographic analysis
- `lsoa_analysis_helper.py`: Geospatial analysis functions for LSOA-level analysis

## Results
Visualizations are stored in `figs/`.

## License
See the LICENSE file for details.
