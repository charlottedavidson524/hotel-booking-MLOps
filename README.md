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
python scripts\download_data.py
```

- Add training, serving and Docker commands later

## Results

- Add metrics later

## Live demo 

- Add Azure URL later