import numpy as np
import pandas as pd


class Cleaner:
    """
    Cleaning logic for the weather dataset.
    - Parses date into month/day
    - Converts sentinel values 8888/9999 to NaN
    - Casts numeric columns
    - Creates binary rain label from RR and drops RR to avoid leakage
    """

    def __init__(self):
        self.sentinel_missing = {8888, 9999}

    def _prepare_numeric(self, df: pd.DataFrame, cols):
        """Cast to float and replace sentinel values with NaN."""
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace(list(self.sentinel_missing), np.nan)
        return df

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # If already preprocessed (no TANGGAL, already has Month/Day and Rain), return in canonical order
        already_cleaned_cols = ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "DDD_X", "FF_AVG", "Month", "Day", "DDD_CAR", "Rain"]
        if "TANGGAL" not in df.columns:
            missing = [c for c in already_cleaned_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Input data missing required columns: {missing}")
            return df[already_cleaned_cols]

        # Parse date
        df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], format="%d-%m-%Y", errors="coerce")
        df["Month"] = df["TANGGAL"].dt.month
        df["Day"] = df["TANGGAL"].dt.day

        # Prepare numeric features
        numeric_cols = ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "DDD_X", "FF_AVG", "Month", "Day", "RR"]
        df = self._prepare_numeric(df, numeric_cols)

        # Create binary rain label from RR (1 if rain > 0 mm else 0, missing treated as 0)
        df["Rain"] = (df["RR"].fillna(0) > 0).astype(int)

        # Drop RR to prevent leakage; drop original date string
        df = df.drop(columns=["RR", "TANGGAL"])

        # Ensure label is the last column
        num_features = ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "DDD_X", "FF_AVG", "Month", "Day"]
        cat_features = ["DDD_CAR"]
        feature_cols = num_features + cat_features
        df = df[feature_cols + ["Rain"]]

        return df
