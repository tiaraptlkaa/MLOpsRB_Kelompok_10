import os
import joblib
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config["model"]["name"]
        self.model_params = self.config["model"]["params"]
        self.model_path = self.config["model"]["store_path"]
        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open("config.yml", "r") as config_file:
            return yaml.safe_load(config_file)

    def create_pipeline(self):
        numeric_features = ["TN", "TX", "TAVG", "RH_AVG", "SS", "FF_X", "DDD_X", "FF_AVG", "Month", "Day"]
        categorical_features = ["DDD_CAR"]

        numeric_pipeline = SkPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = SkPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )

        smote = SMOTE(sampling_strategy=1.0)

        model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
        }

        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", smote),
                ("model", model),
            ]
        )

        return pipeline

    def feature_target_separator(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        model_file_path = os.path.join(self.model_path, "model.pkl")
        joblib.dump(self.pipeline, model_file_path)
