import os
import numpy as np
import pandas as pd

def _generate_weather_rows(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Temperature values (Celsius)
    tn = rng.normal(loc=24, scale=2, size=n).round(1)
    tx = tn + rng.normal(loc=6, scale=1.5, size=n)
    tavg = (tn + tx) / 2

    rh_avg = rng.normal(loc=85, scale=5, size=n).clip(50, 100).round(0)
    rr = rng.exponential(scale=3.0, size=n).round(1)
    ss = rng.normal(loc=5, scale=2, size=n).clip(0, 12).round(1)
    ff_x = rng.normal(loc=5, scale=2, size=n).clip(0, 15).round(1)
    ddd_x = rng.integers(0, 360, size=n)
    ff_avg = (ff_x * rng.uniform(0.3, 0.7, size=n)).round(1)
    ddd_car = rng.choice(["N", "S", "E", "W", "C", "NE", "NW", "SE", "SW"], size=n)

    # Simple rain label: rain if rr > 0.5 mm
    rain = (rr > 0.5).astype(int)

    dates = pd.date_range(start="2025-01-01", periods=n, freq="D").strftime("%d-%m-%Y")

    df = pd.DataFrame(
        {
            "TANGGAL": dates,
            "TN": tn,
            "TX": tx,
            "TAVG": tavg,
            "RH_AVG": rh_avg,
            "RR": rr,
            "SS": ss,
            "FF_X": ff_x,
            "DDD_X": ddd_x,
            "FF_AVG": ff_avg,
            "DDD_CAR": ddd_car,
        }
    )

    return df


def extract_data(train_size: int = 800, test_size: int = 200):
    if not os.path.exists("data"):
        os.mkdir("data")

    df = _generate_weather_rows(train_size + test_size)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"Extracted synthetic weather data: train={train_df.shape}, test={test_df.shape}")

if __name__ == "__main__":
    extract_data()
