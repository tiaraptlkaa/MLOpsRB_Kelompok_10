
# Rainfall / Cuaca Prediction 🌧️🌤️
[![GitHub](https://img.shields.io/badge/GitHub-code-blue?style=flat&logo=github&logoColor=white&color=red)](https://github.com/prsdm/mlops-project)

Prediksi hujan (label biner `Rain`) dari fitur cuaca harian: temperatur, kelembapan, curah hujan, penyinaran, angin, arah angin, serta tanggal.

## Dataset
- Skema mentah: `TANGGAL, TN, TX, TAVG, RH_AVG, RR, SS, FF_X, DDD_X, FF_AVG, DDD_CAR`.
- Konvensi nilai hilang: 8888/9999 dianggap missing (akan diimputasi di cleaner).
- Label: dibuat oleh cleaner sebagai `Rain = (RR > 0)`.
- Fitur turunan: `Month, Day` dari `TANGGAL`; `RR` dijatuhkan setelah label dibuat.

## Pipeline & Tahapan
1. **Ingest**: baca `data/train.csv` & `data/test.csv` (atau data mentah lain) sesuai `config.yml`.
2. **Clean** (`steps/clean.py`): parse tanggal, ganti sentinel 8888/9999 → NaN, imputasi numerik/kat, buat label `Rain` dari `RR>0`, susun fitur + label.
3. **Train** (`steps/train.py`): pipeline imputer + scaler (numeric), imputer + one-hot (`DDD_CAR`), SMOTE, model (default RandomForest). Model disimpan di `models/model.pkl`.
4. **Evaluate** (`steps/predict.py`): accuracy, ROC AUC, classification report.
5. **Track**: `train_with_mlflow()` log params/metrics ke MLflow dan register model.
6. **Serve** (`app.py`): FastAPI `POST /predict` terima payload cuaca mentah.
7. **Monitor**: `monitor.ipynb` + laporan Evidently (sesuaikan ke skema cuaca).

## Mulai Cepat
1) Siapkan environment (Python 3.12 disarankan):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Atau `make setup`.

2) Data
- Gunakan `iklim.csv` atau jalankan `python dataset.py` untuk membuat `data/train.csv` & `data/test.csv` (skema mentah: `TANGGAL, TN, TX, TAVG, RH_AVG, RR, SS, FF_X, DDD_X, FF_AVG, DDD_CAR`).
- DVC remote: `gdrive://1YbgrhKxIgpCDRrrMg72fK1Kwnj7q8cB4` (service account di `.dvc/service-account.json`; set dengan `dvc remote modify --local ...`).

3) Training
```bash
python main.py
```
Model tersimpan di `models/model.pkl`; MLflow logging ada di `train_with_mlflow()`.

4) Serving (FastAPI)
```bash
uvicorn app:app --reload
```
Payload `POST /predict` mengikuti contoh di README.

5) Docker
```bash
docker build -t weather-fastapi .
docker run -p 80:80 weather-fastapi
```

6) Monitoring
Gunakan `monitor.ipynb` / laporan Evidently; sesuaikan ke skema cuaca.

## DVC Quick Notes
- Menarik data: `dvc pull`
- Mengunggah cache baru: `dvc add data && dvc push`
- Service account: `dvc remote modify --local myremote gdrive_use_service_account true` dan `gdrive_service_account_json_file_path .dvc/service-account.json`

## Testing
```bash
python -m pytest
```
Tes tersedia untuk pembersihan data. Tambahkan tes lain bila mengubah pipeline.
