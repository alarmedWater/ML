# =============================================================================
# Mushroom Edibility Classification (ohne 'odor')
# =============================================================================

import pandas as pd
import numpy as np

# Matplotlib-Backend festlegen, um Qt-Plugin-Fehler zu vermeiden
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from scipy.stats import randint, uniform  # für RandomizedSearch


def main():
    # -----------------------------------------------------------------------------
    # 1. Daten laden & Saubere Kopie erstellen
    # -----------------------------------------------------------------------------
    df = pd.read_csv('data/mushrooms.csv').copy()

    # -----------------------------------------------------------------------------
    # 2. Preprocessing
    #    - fehlende Werte entfernen
    #    - Zielvariable kodieren
    #    - 'odor' ausschließen
    # -----------------------------------------------------------------------------
    # 2.1 Entferne Zeilen mit '?' bei stalk-root
    df = df[df['stalk-root'] != '?'].reset_index(drop=True)

    # 2.2 Kodierung der Zielvariable: 'e' → 0, 'p' → 1
    df['target'] = df['class'].map({'e': 0, 'p': 1})

    # 2.3 'odor' entfernen, um Modell ohne dieses starke Feature zu trainieren
    df.drop(columns=['odor'], inplace=True)

    # 2.4 Features und Ziel trennen
    X = df.drop(columns=['class', 'target'])
    y = df['target']

    # -----------------------------------------------------------------------------
    # 3. Train/Test-Split (stratifiziert für gleiche Klassenverteilung)
    # -----------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------------------------------------------------------------
    # 4. Preprocessor: OneHotEncoding aller kategorialen Spalten
    # -----------------------------------------------------------------------------
    categorical_cols = X_train.columns.tolist()
    preprocessor = ColumnTransformer([
        (
            'onehot',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            categorical_cols
        )
    ])

    # -----------------------------------------------------------------------------
    # 5. Cross-Validation-Setup
    # -----------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -----------------------------------------------------------------------------
    # 6a. Random Forest Pipeline + RandomizedSearch
    # -----------------------------------------------------------------------------
    rf_pipe = Pipeline([
        ('prep', preprocessor),
        ('feature_sel', SelectFromModel(
            estimator=RandomForestClassifier(random_state=42),
            threshold='median'
        )),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Suchraum für RandomizedSearchCV
    rf_param_dist = {
        'clf__n_estimators': randint(50, 500),
        'clf__max_depth': [None] + list(randint(5, 50).rvs(5)),
        'clf__min_samples_split': randint(2, 20),
        # 'auto' ist veraltet – erlaubte Werte sind None, 'sqrt', 'log2' oder Float/Int
        'clf__max_features': [None, 'sqrt', 'log2']
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions=rf_param_dist,
        n_iter=30,                # 30 zufällige Kombinationen
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        error_score='raise'       # Fehler sofort anzeigen
    )
    rf_search.fit(X_train, y_train)

    print("=== Random Forest ohne 'odor' ===")
    print("Best RF-Params:", rf_search.best_params_)
    y_pred_rf = rf_search.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    print("ROC-AUC RF:", roc_auc_score(y_test, rf_search.predict_proba(X_test)[:, 1]))

    # -----------------------------------------------------------------------------
    # 6b. Logistic Regression Pipeline + RandomizedSearch
    # -----------------------------------------------------------------------------
    lr_pipe = Pipeline([
        ('prep', preprocessor),
        ('scale', StandardScaler(with_mean=False)),  # Skalieren nach OneHot
        ('clf', LogisticRegression(max_iter=10_000, random_state=42))
    ])

    # Suchraum für Logistic Regression
    lr_param_dist = {
        'clf__C': uniform(0.001, 10),    # C ∈ [0.001, 10.001)
        'clf__penalty': ['l2', 'l1'],
        'clf__solver': ['saga']          # saga unterstützt L1 & L2
    }

    lr_search = RandomizedSearchCV(
        estimator=lr_pipe,
        param_distributions=lr_param_dist,
        n_iter=20,              # 20 zufällige Kombinationen
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        error_score='raise'
    )
    lr_search.fit(X_train, y_train)

    print("\n=== Logistic Regression ohne 'odor' ===")
    print("Best LR-Params:", lr_search.best_params_)
    y_pred_lr = lr_search.predict(X_test)
    print(classification_report(y_test, y_pred_lr))
    print("ROC-AUC LR:", roc_auc_score(y_test, lr_search.predict_proba(X_test)[:, 1]))

    # -----------------------------------------------------------------------------
    # 7. Evaluation & Visualisierung
    # -----------------------------------------------------------------------------
    # 7.1 Confusion Matrix für Random Forest
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_rf,
        display_labels=['edible', 'poisonous'],
        cmap=plt.cm.Blues
    )
    plt.title("Confusion Matrix: Random Forest (ohne 'odor')")
    plt.savefig("rf_confusion_matrix.png")
    plt.close()

    # 7.2 ROC-Kurvenvergleich RF vs. LR
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(
        rf_search.best_estimator_, X_test, y_test, name="Random Forest", ax=ax
    )
    RocCurveDisplay.from_estimator(
        lr_search.best_estimator_, X_test, y_test, name="Logistic Regression", ax=ax
    )
    # Diagonale eines Zufallsmodells
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_title("ROC-Kurvenvergleich (ohne 'odor')")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.savefig("roc_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
