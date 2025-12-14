import numpy as np
import pandas as pd
import pytest

from steps.clean import Cleaner


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "TANGGAL": ["01-01-2025", "02-01-2025", "03-01-2025", "04-01-2025"],
            "TN": [23.4, 24.4, 8888, 25.1],
            "TX": [29.2, 33.6, 32.1, 9999],
            "TAVG": [25.9, 28.1, 27.5, 26.8],
            "RH_AVG": [92, 82, 88, np.nan],
            "RR": [0, 5.4, np.nan, 0],
            "SS": [3.4, 0.4, 5.4, 6.4],
            "FF_X": [5, 4, 4, 5],
            "DDD_X": [330, 140, 320, 110],
            "FF_AVG": [1, 2, 1, 2],
            "DDD_CAR": ["C", "N", np.nan, "C"],
        }
    )


def test_clean_data(sample_data):
    cleaner = Cleaner()
    cleaned_data = cleaner.clean_data(sample_data)

    # Label created and last
    assert "Rain" in cleaned_data.columns
    assert cleaned_data.columns[-1] == "Rain"

    # RR dropped to avoid leakage
    assert "RR" not in cleaned_data.columns

    # Date parsed into Month/Day
    assert "Month" in cleaned_data.columns
    assert "Day" in cleaned_data.columns

    # Sentinel values converted to NaN
    assert cleaned_data["TN"].isnull().sum() == 1
    assert cleaned_data["TX"].isnull().sum() == 1
    assert cleaned_data["DDD_CAR"].isnull().sum() == 1

    # Binary label derived from RR > 0
    # sample_data RR: [0, 5.4, NaN, 0] -> Rain: [0,1,0,0]
    assert cleaned_data["Rain"].tolist() == [0, 1, 0, 0]
