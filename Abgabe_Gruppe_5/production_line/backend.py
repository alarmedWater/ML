import os
import glob
import json
import pickle
import logging
from datetime import datetime

import pandas as pd
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ---------------------- Konfiguration ----------------------
BASELINE_AUC = 0.999
ALERT_DROP_RATIO = 0.9  # 90% der Basis-AUC

LABELED_DATA_DIR = 'labeled_data'
PROCESSED_RECORD = 'processed_files.json'
RAW_DATA_CSV = 'data/mushrooms_full_dataset.csv'
MODEL_DIR = '.'
PIPELINE_NAME = 'pipeline_xgb_full.pkl'

BEST_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 1.0,
    'colsample_bytree': 0.8,
    'n_estimators': 100,
    'use_label_encoder': False,
    'eval_metric': 'auc',
    'random_state': 42
}

# Logging konfigurieren
logging.basicConfig(
    filename='model_monitor.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
# Fehler-Speicher für Frontend
error_messages = []

# --------------- Training Data Schema & Kategorien ---------------
df_train = pd.read_csv(RAW_DATA_CSV)
TRAIN_CATEGORIES = {
    col: set(df_train[col].dropna().unique())
    for col in df_train.columns if col != 'class'
}
EXPECTED_COLUMNS = list(TRAIN_CATEGORIES.keys())

# ---------------------- Helferfunktionen ----------------------
def validate_input(df: pd.DataFrame) -> list:
    """
    Prüft, ob DataFrame-Spalten und Kategorie-Codes dem Training entsprechen.
    Gibt Liste von Fehlern zurück, oder leere Liste.
    """
    errors = []
    cols = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing = expected - cols
    extra = cols - expected
    if missing:
        errors.append(f"Missing columns: {missing}")
    if extra:
        errors.append(f"Unexpected columns: {extra}")
    # Kategorieschema prüfen
    for col in expected & cols:
        invalid = set(df[col].dropna().unique()) - TRAIN_CATEGORIES[col]
        if invalid:
            errors.append(f"Invalid categories in {col}: {invalid}")
    return errors

# ---------------------- Flask API ----------------------
app = Flask(__name__)

# Modelle laden
def load_pipelines():
    global pipelines
    pipelines = {}
    for name in ['lr', 'rf', 'xgb']:
        path = os.path.join(MODEL_DIR, f'pipeline_{name}_full.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                pipelines[name.upper()] = pickle.load(f)
load_pipelines()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Daten einlesen
        if 'file' in request.files:
            df = pd.read_csv(request.files['file'])
        else:
            js = request.get_json(force=True)
            df = pd.DataFrame(js if isinstance(js, list) else [js])
        # Schema-Validierung
        schema_errs = validate_input(df)
        if schema_errs:
            err = f"Schema validation errors: {schema_errs}"
            logging.error(err)
            error_messages.append({'time': datetime.now().isoformat(), 'message': err})
            return jsonify({'error': err}), 400
        # Vorhersage
        results = {}
        for name, pipe in pipelines.items():
            probs = pipe.predict_proba(df)[:, 1]
            labels = ['poisonous', 'edible']
            preds = [labels[int(p >= 0.5)] for p in probs]
            results[name] = preds
        return jsonify(results)
    except Exception as e:
        err = f"Prediction-Fehler: {e}"
        logging.error(err)
        error_messages.append({'time': datetime.now().isoformat(), 'message': err})
        return jsonify({'error': err}), 500

@app.route('/errors', methods=['GET'])
def get_errors():
    """Gibt die gesammelten Fehlermeldungen zurück."""
    return jsonify({'errors': error_messages})

# ---------------------- Monitoring & Data-Drift ----------------------
def load_processed_files() -> set:
    if os.path.exists(PROCESSED_RECORD):
        return set(json.load(open(PROCESSED_RECORD)))
    return set()

def save_processed_files(files: set):
    json.dump(list(files), open(PROCESSED_RECORD, 'w'))

def monitor_performance():
    processed = load_processed_files()
    files = glob.glob(os.path.join(LABELED_DATA_DIR, '*.csv'))
    new_files = [f for f in files if f not in processed]
    for f in new_files:
        try:
            df_new = pd.read_csv(f)
            # Schema & Drift prüfen
            drift_errs = validate_input(df_new.drop(columns=['class']))
            if drift_errs:
                err = f"Data-Drift / Schema-Errors in {os.path.basename(f)}: {drift_errs}"
                logging.warning(err)
                error_messages.append({'time': datetime.now().isoformat(), 'message': err})
            # Performance prüfen
            X_new = df_new.drop(columns=['class'])
            y_true = df_new['class'].map({'e':1, 'p':0})
            y_prob = pipelines['XGB'].predict_proba(X_new)[:, 1]
            current_auc = roc_auc_score(y_true, y_prob)
            logging.info(f'AUC für {os.path.basename(f)}: {current_auc:.3f}')
            if current_auc < ALERT_DROP_RATIO * BASELINE_AUC:
                msg = f"Modell-AUC gesunken: {current_auc:.3f} < {ALERT_DROP_RATIO*BASELINE_AUC:.3f}"
                logging.error(msg)
                error_messages.append({'time': datetime.now().isoformat(), 'message': msg})
        except Exception as e:
            err = f"Monitoring-Fehler bei {f}: {e}"
            logging.error(err)
            error_messages.append({'time': datetime.now().isoformat(), 'message': err})
        processed.add(f)
    save_processed_files(processed)

# ---------------------- Retraining ----------------------
def prepare_data(df: pd.DataFrame):
    df_clean = df[df['stalk-root'] != '?'].copy()
    y = df_clean['class'].map({'e':1, 'p':0})
    X = df_clean.drop(columns=['class'])
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def retrain_model():
    try:
        df_all = pd.read_csv(RAW_DATA_CSV)
        X_tr, X_te, y_tr, y_te = prepare_data(df_all)
        ohe = ColumnTransformer([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X_tr.columns)
        ], remainder='drop')
        X_tr_ohe = ohe.fit_transform(X_tr)
        X_te_ohe = ohe.transform(X_te)
        model = XGBClassifier(**BEST_PARAMS)
        model.fit(X_tr_ohe, y_tr)
        y_prob = model.predict_proba(X_te_ohe)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        logging.info(f'Retraining AUC: {auc:.3f}')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_name = f'pipeline_xgb_full_{timestamp}.pkl'
        pipeline = Pipeline([('onehot', ohe), ('clf', model)])
        with open(new_name, 'wb') as f:
            pickle.dump(pipeline, f)
        os.replace(new_name, os.path.join(MODEL_DIR, PIPELINE_NAME))
        load_pipelines()
        logging.info('Pipeline neu trainiert und aktualisiert')
    except Exception as e:
        err = f"Retraining-Fehler: {e}"
        logging.error(err)
        error_messages.append({'time': datetime.now().isoformat(), 'message': err})

# ---------------------- Scheduler ----------------------
if __name__ == '__main__':
    scheduler = BackgroundScheduler(timezone='Europe/Berlin')
    scheduler.add_job(monitor_performance, 'cron', hour=0, minute=0)
    scheduler.add_job(retrain_model, 'cron', day=1, month='*/3', hour=3, minute=0)
    scheduler.start()
    app.run(host='0.0.0.0', port=5000)

# ---------------------- Unit-Tests (pytest) ----------------------
# Speichere diesen Bereich als tests/test_backend.py
import pytest
from flask.testing import FlaskClient

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

# Sample valid record aus Training
valid_sample = df_train.drop(columns=['class']).iloc[0].to_dict()

def test_predict_valid_json(client: FlaskClient):
    resp = client.post('/predict', json=valid_sample)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'XGB' in data

def test_predict_missing_field(client: FlaskClient):
    resp = client.post('/predict', json={'foo': 'bar'})
    assert resp.status_code == 400
    assert 'error' in resp.get_json()

def test_validate_input():
    # Fehlende Spalte
    df = pd.DataFrame([{'cap-shape': 'b'}])
    errs = validate_input(df)
    assert errs and 'Missing columns' in errs[0]
