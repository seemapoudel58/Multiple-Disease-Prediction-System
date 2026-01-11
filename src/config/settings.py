import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Database paths
DATABASE_DIR = PROJECT_ROOT / "database"
USER_DB = DATABASE_DIR / "new_user.db"
LEGACY_DB = DATABASE_DIR / "your_database.db"

# EDA/Notebooks paths
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EDA_DIR = NOTEBOOKS_DIR / "eda"
EDA_IMAGES_DIR = RESULTS_DIR / "eda"  
EDA_DIABETES_DIR = EDA_IMAGES_DIR / "diabetes"
EDA_HEART_DISEASE_DIR = EDA_IMAGES_DIR / "heart_disease"
EDA_BREAST_CANCER_DIR = EDA_IMAGES_DIR / "breast_cancer"

STATIC_ASSETS_DIR = RESULTS_DIR / "static"
STATIC_DIABETES_DIR = STATIC_ASSETS_DIR / "diabetes"
STATIC_HEART_DISEASE_DIR = STATIC_ASSETS_DIR / "heart_disease"
STATIC_BREAST_CANCER_DIR = STATIC_ASSETS_DIR / "breast_cancer"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  SAVED_MODELS_DIR, RESULTS_DIR, METRICS_DIR, PLOTS_DIR, 
                  DATABASE_DIR, NOTEBOOKS_DIR, EDA_DIR, EDA_IMAGES_DIR,
                  EDA_DIABETES_DIR, EDA_HEART_DISEASE_DIR, EDA_BREAST_CANCER_DIR,
                  STATIC_ASSETS_DIR, STATIC_DIABETES_DIR, STATIC_HEART_DISEASE_DIR,
                  STATIC_BREAST_CANCER_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
