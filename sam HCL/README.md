# OTT & Movie Viewing Engagement Analysis

**Student Roll Number:** `2026b01010055`  
**Domain:** OTT and Movies Viewing  

## Business Problem
OTT platform product and retention managers need to understand what drives viewer engagement across subscription tiers, devices, genres, and Indian viewing markets. Analyzing viewing sessions helps identify which usage habits and plan tiers correspond to stronger engagement and which low-engagement sessions need targeted retention actions. This enables more effective content programming, pricing strategy, and user re-engagement.

## Research Questions
1. **What does the outcome look like overall?** What is the distribution, mean, median, and spread of `Engagement_Score` across viewing sessions?
2. **Do `Watch_Duration_Mins` and `Monthly_Spend_INR` affect the outcome?** How strongly do session duration and monthly spending correlate with `Engagement_Score`, compared to `User_Rating`?
3. **Does the outcome differ by `City`, `Subscription_Plan`, `Device_Type`, and `Genre`?** How does engagement vary across user segments and playback devices?
4. **Which records need attention?** Using `Engagement_Score < 50.0` as the **"At Risk"** threshold, which sessions require retention intervention and what characterizes them?

## Project Structure
```text
ott-movie-viewing-engagement-analysis/
├── data/
│   ├── raw/ott_movies_data.csv                 # Original raw dataset (255 rows, 11 columns)
│   └── processed/ott_movies_data_clean.csv     # Cleaned & feature-engineered dataset (250 rows, 14 columns)
├── notebooks/
│   └── ott_analysis.ipynb                      # Complete end-to-end Jupyter Notebook
├── images/
│   ├── 01_engagement_distribution.png
│   ├── 02_numeric_drivers_vs_engagement.png
│   ├── 03_engagement_by_categories.png
│   └── 04_retention_status_breakdown.png
├── requirements.txt
└── README.md