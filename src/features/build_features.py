import pandas as pd

TARGET_COL = "booking_status"

#Rare categories found in EDA (5 and 7 rows) were merged into an 
# 'Other' level
RARE_CATEGORY_MAP = {
    "meal_type": {"Meal Plan 3": "Other"},
    "room_type": {"Room_Type 3": "Room_Type Other"},
}

# Fixed category lists. 
# CRUCIAL: these define the one-hot columns and need to be identical
# in training and serving. The first level of each is the dropped reference
CATEGORIES = {
    "meal_type": ["Meal Plan 1", "Meal Plan 2", "Not Selected", "Other"],
    "room_type": ["Room_Type 1", "Room_Type 2", "Room_Type 4", "Room_Type 5",
                  "Room_Type 6", "Room_Type 7", "Room_Type Other"],
    "market_segment": ["Aviation", "Complementary", "Corporate", "Offline", "Online"],
}

# 3, 4 and 5 special requests all had 0% cancellations
MAX_SPECIAL_REQUESTS = 3

def build_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Apply feature engineering to preprocessed hotel booking data.
        - Combined features: total_nights, total_guests
        - Date features: res_month, res_dayofweek (no year, see EDA on drift)
        - Cap n_special_requests at 3
        - Group rare meal/room types into "Other"
        - One-hot encode categoricals against fixed category lists (drop_first=True)
    Works for both training data and a single booking at serving time.
    """
    df = df.copy()
    print(f"Feature engineering on {df.shape[1]} columns: ")

    # 1: Combined features 
    df["total_nights"] = df["n_weekend_nights"] + df["n_week_nights"]
    df["total_guests"] = df["n_adults"] + df["n_children"]

    # 2: Date features 
    df["res_month"] = df["reservation_date"].dt.month
    df["res_dayofweek"] = df["reservation_date"].dt.day_of_week
    df = df.drop(columns="reservation_date")

    # 3: Cap special requests 
    df["n_special_requests"] = df["n_special_requests"].clip(upper=MAX_SPECIAL_REQUESTS)

    # 4: Group rare categories
    for col, mapping in RARE_CATEGORY_MAP.items():
        df[col] = df[col].replace(mapping)

    # 5: One-hot encode with fixed categories
    for col, categories in CATEGORIES.items():
        unknown = set(df[col].dropna().unique()) - set(categories)
        if unknown:
            raise ValueError(f"Unknown values in '{col}': {unknown}")
        df[col] = pd.Categorical(df[col], categories=categories)

    df = pd.get_dummies(df, columns=list(CATEGORIES), drop_first=True, dtype=int)

    print(f"Feature engineering complete: {df.shape[1]} columns")
    return df
