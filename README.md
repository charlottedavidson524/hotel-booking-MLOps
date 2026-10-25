# hotel-booking-MLOps

This project is to predict whether a hotel booking will be cancelled. This would help hotels manage overbooking and follow up with guests who are considered to be at risk. 

This is a binary classification problem, and is built using the full lifecycle from data to a deployed API. 

## Tech Stack

- **Experiment tracking**: MLFlow
- **Data validation**: Great Expectations
- **Model**: XGBoost 
- **API and UI**: FastAPI and Gradio
- **Container**: Docker
- **CI/CD**: GitHub Actions
- **Cloud**: Azure Container Apps

## Dataset

A hotel booking canecllation dataset found on Kaggle. The dataset is originally from the article Hotel Booking Demand Datasets, written by Nuno Antonio, Ana Almeida and Luis Nunes for Data in Brief, Volume 22, February 2019.

- Kaggle link: https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction

- Original source: https://www.sciencedirect.com/science/article/pii/S2352340918315191

Dataset isn't included in the repo. The download script (see quick start below) fetches it.

## Quick Start

``` cmd
git clone https://github.com/<your-username>/hotel-booking-MLOps.git
cd hotel-booking-MLOps
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.download_data
```

- Add training, serving and Docker commands later

## Results

The baseline results (from the notebook, with 80/20 group-aware stratified split, threshold 0.5):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.632 | 0.760 | 0.690 | 0.859 |
| Random Forest | 0.733 | 0.741 | 0.737 | 0.902 |
| LightGBM | 0.722 | 0.821 | 0.768 | 0.927 |
| **XGBoost** | 0.720 | 0.801 | 0.758 | 0.921 |

XGBoost was chosen for it's prediction speed which was around 5x faster than LightGBM with very similar performance.

Optuna tuning (5-fold grouped CV) didn't provide any improvement over baseline parameters

## Key Findings

- Lead time is the strongest predictor. Guests with special requests, parking or repeat stays rarely cancel

- 28% of rows are duplicates once the ID is ignored (most likely group bookings). These are kept but a group aware split is used to prevent train/test leakage.

- Cancellation rates shiffted sharply between 2017 and 2018, so year is excluded as a feature and drift monitoring is going to be planned after deployment.

## Live demo 

- Add Azure URL later