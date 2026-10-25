import pandas as pd

TARGET_COL = "booking_status"

# Convert raw kaggle column names to snake case (needs to be shared)
# with API later

COLUMN_MAPPING = {
    "Booking_ID": "booking_id",
    "number of adults": "n_adults",
    "number of children": "n_children",
    "number of weekend nights": "n_weekend_nights",
    "number of week nights": "n_week_nights",
    "type of meal": "meal_type",
    "car parking space": "car_parking_space",
    "room type": "room_type",
    "lead time": "lead_time",
    "market segment type": "market_segment",
    "repeated": "repeated_guest",
    "P-C": "prev_cancellations",
    "P-not-C": "prev_not_cancelled",
    "average price": "avg_price",
    "special requests": "n_special_requests",
    "date of reservation": "reservation_date",
    "booking status": TARGET_COL,
}

def preprocess_data(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Basic cleaning for hotel booking data.
        - Trim and rename columns to snake case
        - Drop ID column
        - Map target to 0/1 (Cancelled = 1) if present
        - Parse reservation_date and drop the invalid dates ('2018-2-29')
    Duplicates, zero-night, zero-adult and zero-price rows were determined
    to be legitimate and are left in on prupose (see EDA)
    """
    df = df.copy()

    # Tidy the headers
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAPPING)

    # Drop ID if it's present
    if "booking_id" in df.columns:
        df = df.drop(columns=["booking_id"])

    # Target to 0/1 if it's still text
    if target_col in df.columns and not pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = (df[target_col].str.strip() == "Canceled").astype(int)


    # Parse dates. Invalid ones (37 rows of '2018-2-29') become NaT and are dropped
    if "reservation_date" in df.columns:
        df["reservation_date"] = pd.to_datetime(
            df["reservation_date"], format="%m/%d/%Y", errors="coerce"
        )

        df = df.dropna(subset=["reservation_date"])

    return df.reset_index(drop=True)