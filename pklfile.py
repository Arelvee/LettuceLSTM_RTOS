from xgboost import XGBRegressor, XGBClassifier
import pickle
from tensorflow.keras.models import load_model

# Load models
xgb_reg = XGBRegressor()
xgb_reg.load_model("saved_models/xgb_reg.json")

xgb_clf = XGBClassifier()
xgb_clf.load_model("saved_models/xgb_clf.json")

with open("saved_models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("saved_models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

lstm_model = load_model("saved_models/lstm_feature_extractor.keras")