# Supply Chain Cost Analytics Portfolio Project

## Overview
Analyzed procurement and shipping costs using DataCo supply chain dataset 
to identify cost drivers and optimization opportunities.

## Tools Used
- Python (pandas, numpy, matplotlib, seaborn)
- Tableau/Power BI

## Key Findings
- Identified 23% of orders with shipping costs >15% of order value
- Standard Class shipping 40% cheaper than Same Day with only 2-day delay
- Top 3 categories account for 68% of total procurement costs
- Seasonal cost spikes in Q4 (holiday season)

## Data Source
Dataset: [DataCo Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

**Note:** Due to file size (91MB), the raw dataset is not included in this repository.
Download from Kaggle and place in project root to run the analysis.

## Files Included
- `supply_chain_analysis.ipynb` - Full data cleaning & analysis
- `supply_chain_sample.csv` - Sample dataset (1000 rows)
- `category_summary_small.csv` - Aggregated category metrics
- `monthly_summary_small.csv` - Monthly KPIs
- `cost_distributions.png` - Visualization outputs

## How to Run
1. Download dataset from Kaggle link above
2. Place `DataCoSupplyChainDataset.csv` in project root
3. Run: `python supply_chain_analysis.py`

## Methodology
1. Data cleaning: handled 5.2% missing values, removed 347 duplicates
2. Feature engineering: created 8 new cost metrics
3. Outlier treatment: capped extreme values at 99th percentile
4. Visualization: 4-page interactive dashboard

## Impact
- Recommended switching 30% of orders to Standard Class → $125K annual savings
- Flagged 15 high-cost suppliers for renegotiation
