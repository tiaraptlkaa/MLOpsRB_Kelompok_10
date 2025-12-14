# Rainfall / Cuaca Prediction 🌧️🌤️
[![GitHub](https://img.shields.io/badge/GitHub-code-blue?style=flat&logo=github&logoColor=white&color=red)](https://github.com/prsdm/mlops-project)

Prediksi hujan (label biner `Rain`) dari fitur cuaca harian: temperatur, kelembapan, hujan (`RR`), penyinaran (`SS`), angin (`FF_X`, `FF_AVG`, `DDD_X`, `DDD_CAR`), serta tanggal (`TANGGAL` → `Month`/`Day`).


## Diagram
Alur dari ingestion → cleaning → training → deployment:
![Image](docs/mlops.jpg)

## Struktur Data
- Mentah (contoh `iklim.csv` / `dataset.py`): `TANGGAL, TN, TX, TAVG, RH_AVG, RR, SS, FF_X, DDD_X, FF_AVG, DDD_CAR`
- Saat training: cleaner menambah `Month, Day`, membuat label `Rain` = `RR > 0`, dan menjatuhkan `RR`.

## Persiapan Lingkungan
Gunakan Python 3.12 (menghindari build `pyarrow` di 3.13):
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Atau:
```bash
make setup
```

## Data
- Pakai data contoh: `iklim.csv` → split manual ke `data/train.csv` & `data/test.csv` via `python dataset.py`.
- Atau siapkan sendiri `data/train.csv` dan `data/test.csv` dengan skema mentah di atas.
- DVC (opsional): remote GDrive `gdrive://1YbgrhKxIgpCDRrrMg72fK1Kwnj7q8cB4`. Untuk service account:
  ```bash
  dvc remote modify --local myremote gdrive_use_service_account true
  dvc remote modify --local myremote gdrive_service_account_json_file_path .dvc/service-account.json
  dvc pull   # tarik data
  dvc push   # dorong cache
  ```

## Train Model
```bash
python main.py          # atau make run
```
Model (default `RandomForestClassifier`) disimpan ke `models/model.pkl`. Logging & register MLflow ada di `train_with_mlflow()`.

## API (FastAPI)
```bash
uvicorn app:app --reload
```
Contoh payload `POST /predict`:
```json
{
  "TANGGAL": "01-01-2025",
  "TN": 23.4,
  "TX": 29.2,
  "TAVG": 25.9,
  "RH_AVG": 92,
  "SS": 3.4,
  "FF_X": 5.0,
  "DDD_X": 330,
  "FF_AVG": 1.0,
  "DDD_CAR": "C"
}
```
Respons: `{"predicted_class": 0|1}` (0=tidak hujan, 1=hujan).

## Docker
```bash
docker build -t weather-fastapi .
docker run -p 80:80 weather-fastapi
```

## Monitoring
`monitor.ipynb` dan laporan drift (`production_drift.html`, `test_drift.html`) perlu disesuaikan ke skema cuaca; Evidently dapat diaktifkan via requirements (baris dikomentari).

## Testing
```bash
python -m pytest   # atau make test
```

## License

Copyright © 2024, [Prasad Mahamulkar](https://github.com/prsdm).

Released under the [Apache-2.0 license](LICENSE).
