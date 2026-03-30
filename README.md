# Climate & Health Risk Prediction Challenge

A machine learning solution for predicting climate-sensitive deaths in low-resource settings, using an ensemble of gradient boosting models.

## Project Overview

This project builds an ensemble model to predict whether a recorded death is climate-sensitive based on:
- Demographic features (age, gender, zone)
- Climate/weather data (temperature, rainfall, vegetation indices)
- Terrain features (elevation, slope)
- External climate data (Open-Meteo)

**Evaluation Metric**: Final Score = 0.60 × F1-Score + 0.40 × ROC-AUC

---

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost
```

### 2. Download External Data (Optional)

The model uses Open-Meteo historical weather data. To download it:

```bash
python download_openmeteo.py
```

This will create `data/external/openmeteo_climate.csv` with additional climate features.

### 3. Run the Model

```bash
cd "Climate & Health Prediction Challenge"
python model_with_external_data.py
```

### 4. Output

The submission file will be saved to:
```
data/submission.csv
```

---

## Directory Structure

```
climate health 3/
├── model_with_external_data.py    # Main training script
├── download_openmeteo.py           # Script to download external data
├── climate_health_starter_notebook_.ipynb  # Jupyter notebook with EDA
├── data/
│   ├── Train.csv                   # Training data
│   ├── Test.csv                   # Test data
│   ├── climate_features.csv        # Climate features (competition-provided)
│   ├── SampleSubmission.csv        # Sample submission format
│   └── external/
│       └── openmeteo_climate.csv   # External climate data (download separately)
└── README.md
```

---

## Data Description

### Training Data (`Train.csv`)
| Column | Description |
|--------|-------------|
| ID | Unique identifier |
| zone | Urban/Rural/Peri-urban |
| gender | Male/Female |
| deathdate | Date of death |
| age | Age at death |
| latitude | Latitude coordinate |
| longitude | Longitude coordinate |
| location | Text location name |
| is_climate_sensitive | Target variable (1 = climate-sensitive, 0 = not) |

### Climate Features (`climate_features.csv`)
| Column | Description |
|--------|-------------|
| avg_temperature | Average temperature |
| max_temperature | Maximum temperature |
| min_temperature | Minimum temperature |
| precipitation | Precipitation amount |
| elevation | Elevation (meters) |
| hot_days_30d | Number of hot days in last 30 days |
| max_daily_rain_30d | Maximum daily rainfall in last 30 days |
| ndvi_30d | Normalized Difference Vegetation Index (30-day) |
| ndvi_90d | Normalized Difference Vegetation Index (90-day) |
| rain_days_30d | Number of rainy days in last 30 days |
| rain_sum_30d | Total rainfall in last 30 days |
| rain_sum_7d | Total rainfall in last 7 days |
| rain_sum_90d | Total rainfall in last 90 days |
| slope | Terrain slope |
| tavg_30d | 30-day average temperature |
| tavg_7d | 7-day average temperature |
| tavg_90d | 90-day average temperature |
| temp_range_mean_30d | Mean temperature range (30-day) |
| tmax_30d | 30-day maximum temperature |
| tmin_30d | 30-day minimum temperature |

### External Data (`external/openmeteo_climate.csv`)
Download from Open-Meteo API. Variables include:
- `temp_max`, `temp_min`, `temp_mean` - Temperature statistics
- `precipitation` - Precipitation amount
- `rain` - Rain indicator
- `humidity` - Relative humidity

---

## Model Architecture

### Ensemble of 4 Models

| Model | Weight | Description |
|-------|--------|-------------|
| LightGBM | 30% | Gradient boosting with leaf-wise growth |
| CatBoost | 30% | Ordered boosting with symmetric trees |
| XGBoost | 25% | Gradient boosting with level-wise growth |
| RandomForest | 15% | Bagging ensemble |

### Feature Engineering

The model creates ~100 features including:

1. **Temporal Features**: year, month, day, day_of_week, day_of_year, week_of_year, quarter, is_weekend

2. **Seasonal Features**: is_rainy_season, is_dry_season, is_long_rain, is_short_rain

3. **Temperature Features**: 
   - temp_range, temp_vs_tavg30, temp_vs_tavg90
   - tmax_vs_tavg, tmin_vs_tavg
   - temp_stability, temp_volatility
   - tavg_trend_7d_30d, tavg_trend_30d_90d

4. **Rainfall Features**:
   - rain_ratio_7d_30d, rain_ratio_30d_90d
   - rain_intensity, heavy_rain_event
   - drought_indicator, flood_indicator

5. **Vegetation (NDVI)**: ndvi_change, ndvi_ratio, ndvi_product

6. **Demographic Features**:
   - age_squared, age_log, age_group
   - is_infant, is_child, is_elderly

7. **Interaction Features**:
   - temp_age (temperature × age)
   - elderly_heat (elderly × heat stress)
   - rain_age (rainfall × age)

### Hyperparameters

| Model | Key Parameters |
|-------|----------------|
| LightGBM | num_leaves=31, max_depth=6, learning_rate=0.03 |
| CatBoost | depth=6, learning_rate=0.03, l2_leaf_reg=3 |
| XGBoost | max_depth=6, learning_rate=0.03, min_child_weight=20 |
| RandomForest | n_estimators=300, max_depth=10, class_weight=balanced |

---

## Cross-Validation Results

| Model | F1-Score | ROC-AUC | Weighted |
|-------|----------|---------|----------|
| LightGBM | 0.7948 | 0.7985 | 0.7963 |
| XGBoost | 0.8133 | 0.8130 | 0.8132 |
| CatBoost | 0.8127 | 0.8177 | 0.8147 |
| RandomForest | 0.7849 | 0.8121 | 0.7958 |
| **Ensemble** | **0.8106** | **0.8168** | **0.8131** |

---

## How to Reproduce Results

### Option 1: Use Provided Data Only (No External Download)

The model will work without external data. It uses only the competition-provided `climate_features.csv`.

```bash
python model_with_external_data.py
```

### Option 2: With External Data

1. First, download the Open-Meteo data:
```bash
python download_openmeteo.py
```

2. Then run the main script:
```bash
python model_with_external_data.py
```

---

## Submission Format

The output file (`data/submission.csv`) has three columns:

| Column | Description |
|--------|-------------|
| ID | Test sample identifier |
| TargetF1 | Binary prediction (0 or 1) |
| TargetRAUC | Probability score for ROC-AUC evaluation |

Example:
```
ID,TargetF1,TargetRAUC
ID_E760D84B,0,0.4472
ID_6EDEA907,1,0.6442
ID_B9FFC8D8,1,0.5131
```

---

## Troubleshooting

### ImportError: No module named 'catboost'
```bash
pip install catboost
```

### ImportError: No module named 'lightgbm'
```bash
pip install lightgbm
```

### ImportError: No module named 'xgboost'
```bash
pip install xgboost
```

### Missing external data warning
The script will continue without external data if `data/external/openmeteo_climate.csv` is not found. Results may be slightly different.

---

## External Data Source

### Open-Meteo Historical Weather Data
- **Website**: https://open-meteo.com/
- **License**: CC BY 4.0
- **API Documentation**: https://open-meteo.com/en/docs

The external data provides historical weather observations that complement the competition-provided climate features, potentially improving model performance.

---

## Notes

- The model uses 5-fold stratified cross-validation for robust evaluation
- Class weights are automatically balanced to handle class imbalance
- All features are normalized using training set statistics (no data leakage)
- Random seed is set to 42 for reproducibility
