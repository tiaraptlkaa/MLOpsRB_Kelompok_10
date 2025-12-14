import os
import joblib
import pandas as pd
import pytest

from steps.train import Trainer
from steps.predict import Predictor
import dataset as data_module
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier


def _make_sample_clean_data():
    # 12 rows, balanced classes for SMOTE (min 6 minority samples)
    records = []
    for i in range(12):
        records.append(
            {
                "TN": 20.0 + i,
                "TX": 26.0 + i,
                "TAVG": 23.0 + i,
                "RH_AVG": 70 + i,
                "SS": 3.0 + (i % 4),
                "FF_X": 2.0 + (i % 3),
                "DDD_X": 100 + i,
                "FF_AVG": 1.0 + (i % 2),
                "Month": 1 + (i % 12),
                "Day": 1 + i,
                "DDD_CAR": "N" if i % 2 == 0 else "S",
                "Rain": 0 if i < 6 else 1,
            }
        )
    return pd.DataFrame(records)


def test_trainer_pipeline_and_save(monkeypatch, tmp_path):
    config = {
        "model": {
            "name": "DecisionTreeClassifier",
            "params": {"max_depth": None},
            "store_path": str(tmp_path),
        }
    }
    monkeypatch.setattr(Trainer, "load_config", lambda self: config)

    # Override pipeline to avoid SMOTE during tests
    def make_simple_pipeline():
        numeric_features = ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "DDD_X", "FF_AVG", "Month", "Day"]
        categorical_features = ["DDD_CAR"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                ("categorical", SkPipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
            ]
        )
        return SkPipeline(steps=[("preprocessor", preprocessor), ("model", DecisionTreeClassifier())])

    monkeypatch.setattr(Trainer, "create_pipeline", lambda self: make_simple_pipeline())

    trainer = Trainer()
    data = _make_sample_clean_data()
    X, y = trainer.feature_target_separator(data)
    trainer.train_model(X, y)
    trainer.save_model()

    model_file = tmp_path / "model.pkl"
    assert model_file.exists()

    # Ensure pipeline can predict and returns expected length
    preds = trainer.pipeline.predict(X)
    assert len(preds) == len(y)


def test_predictor_evaluate(monkeypatch, tmp_path):
    # Train and persist a model, then evaluate via Predictor
    config = {
        "model": {
            "name": "DecisionTreeClassifier",
            "params": {"max_depth": None},
            "store_path": str(tmp_path),
        }
    }
    monkeypatch.setattr(Trainer, "load_config", lambda self: config)
    def make_simple_pipeline():
        numeric_features = ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "DDD_X", "FF_AVG", "Month", "Day"]
        categorical_features = ["DDD_CAR"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                ("categorical", SkPipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
            ]
        )
        return SkPipeline(steps=[("preprocessor", preprocessor), ("model", DecisionTreeClassifier())])

    monkeypatch.setattr(Trainer, "create_pipeline", lambda self: make_simple_pipeline())

    trainer = Trainer()
    data = _make_sample_clean_data()
    X, y = trainer.feature_target_separator(data)
    trainer.train_model(X, y)
    joblib.dump(trainer.pipeline, os.path.join(config["model"]["store_path"], "model.pkl"))

    monkeypatch.setattr(Predictor, "load_config", lambda self: config)
    predictor = Predictor()
    accuracy, class_report, roc_auc = predictor.evaluate_model(X, y)

    assert accuracy >= 0.9
    assert roc_auc >= 0.9
    assert "precision" in class_report


def test_dataset_extract_data(monkeypatch, tmp_path):
    # Run extract_data in a temp cwd to avoid polluting repo
    monkeypatch.chdir(tmp_path)
    data_module.extract_data(train_size=20, test_size=10)

    train_path = tmp_path / "data" / "train.csv"
    test_path = tmp_path / "data" / "test.csv"
    assert train_path.exists()
    assert test_path.exists()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert train_df.shape[0] == 20
    assert test_df.shape[0] == 10
    assert set(["TANGGAL", "TN", "TX", "Rain"]).issuperset({"TANGGAL", "TN", "TX"})
