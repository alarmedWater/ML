import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# ─── Backend: Load pipelines for both feature subsets ─────────────────────────
@st.cache_data
def load_pipelines():
    """Load pre-trained sklearn pipelines for Full and Forest subsets."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'Models')

    full_pipes = {}
    forest_pipes = {}
    for name in ['lr', 'rf', 'xgb']:
        full_path = os.path.join(models_dir, f'pipeline_{name}_full_tuned.pkl')
        forest_path = os.path.join(models_dir,
                                   f'pipeline_{name}_forest_tuned.pkl')

        with open(full_path, 'rb') as f_full:
            full_pipes[name.upper()] = pickle.load(f_full)
        with open(forest_path, 'rb') as f_forest:
            forest_pipes[name.upper()] = pickle.load(f_forest)

    return full_pipes, forest_pipes


full_pipelines, forest_pipelines = load_pipelines()

# ─── GUI: Page configuration and title ────────────────────────────────────────
st.set_page_config(page_title="Mushroom Predictor", layout="wide")
st.title("🍄 Mushroom Edibility Predictor")

# ─── GUI: Subset selection (in German) ────────────────────────────────────────
subset = st.sidebar.radio(
    "Wähle Feature-Set:",
    options=["Full (alle Features)", "Forest (10 Wald-Features)"]
)
pipelines = (full_pipelines
             if subset.startswith("Full")
             else forest_pipelines)

# ─── Feature mapping for user inputs ─────────────────────────────────────────
feature_maps = {
    'cap-shape': {
        'b': 'bell', 'c': 'conical', 'x': 'convex',
        'f': 'flat', 'k': 'knobbed', 's': 'sunken'
    },
    'cap-surface': {
        'f': 'fibrous', 'g': 'grooves',
        'y': 'scaly', 's': 'smooth'
    },
    'cap-color': {
        'n': 'brown', 'b': 'buff', 'c': 'cinnamon',
        'g': 'gray', 'r': 'green', 'p': 'pink',
        'u': 'purple', 'e': 'red', 'w': 'white',
        'y': 'yellow'
    },
    'bruises': {'t': 'bruises', 'f': 'no'},
    'odor': {
        'a': 'almond', 'l': 'anise', 'c': 'creosote',
        'y': 'fishy', 'f': 'foul', 'm': 'musty',
        'n': 'none', 'p': 'pungent', 's': 'spicy'
    },
    'gill-attachment': {
        'a': 'attached', 'd': 'descending',
        'f': 'free', 'n': 'notched'
    },
    'gill-spacing': {
        'c': 'close', 'w': 'crowded', 'd': 'distant'
    },
    'gill-size': {'b': 'broad', 'n': 'narrow'},
    'gill-color': {
        'k': 'black', 'n': 'brown', 'b': 'buff',
        'h': 'chocolate', 'g': 'gray', 'r': 'green',
        'o': 'orange', 'p': 'pink', 'u': 'purple',
        'e': 'red', 'w': 'white', 'y': 'yellow'
    },
    'stalk-shape': {'e': 'enlarging', 't': 'tapering'},
    'stalk-root': {
        'b': 'bulbous', 'c': 'club', 'u': 'cup',
        'e': 'equal', 'z': 'rhizomorphs',
        'r': 'rooted', '?': 'missing'
    },
    'stalk-surface-above-ring': {
        'f': 'fibrous', 'y': 'scaly',
        'k': 'silky', 's': 'smooth'
    },
    'stalk-surface-below-ring': {
        'f': 'fibrous', 'y': 'scaly',
        'k': 'silky', 's': 'smooth'
    },
    'stalk-color-above-ring': {
        'n': 'brown', 'b': 'buff', 'c': 'cinnamon',
        'g': 'gray', 'o': 'orange', 'p': 'pink',
        'e': 'red', 'w': 'white', 'y': 'yellow'
    },
    'stalk-color-below-ring': {
        'n': 'brown', 'b': 'buff', 'c': 'cinnamon',
        'g': 'gray', 'o': 'orange', 'p': 'pink',
        'e': 'red', 'w': 'white', 'y': 'yellow'
    },
    'veil-color': {
        'n': 'brown', 'o': 'orange',
        'w': 'white', 'y': 'yellow'
    },
    'ring-number': {'n': 'none', 'o': 'one', 't': 'two'},
    'ring-type': {
        'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring',
        'l': 'large', 'n': 'none', 'p': 'pendant',
        's': 'sheathing', 'z': 'zone'
    },
    'spore-print-color': {
        'k': 'black', 'n': 'brown', 'b': 'buff',
        'h': 'chocolate', 'r': 'green', 'o': 'orange',
        'u': 'purple', 'w': 'white', 'y': 'yellow'
    },
    'population': {
        'a': 'abundant', 'c': 'clustered',
        'n': 'numerous', 's': 'scattered',
        'v': 'several', 'y': 'solitary'
    },
    'habitat': {
        'g': 'grasses', 'l': 'leaves', 'm': 'meadows',
        'p': 'paths', 'u': 'urban', 'w': 'waste',
        'd': 'woods'
    }
}

# ─── Exclusion list for Forest subset ────────────────────────────────────────
exclude_forest = [
    "gill-size", "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring",
    "veil-color", "spore-print-color", "population"
]

# ─── Determine which features to display ─────────────────────────────────────
if subset.startswith("Full"):
    visible_feats = list(feature_maps.keys())
else:
    visible_feats = [
        feat for feat in feature_maps
        if feat not in exclude_forest
    ]

# ─── Render dynamic selectboxes for each feature ─────────────────────────────
cols = st.columns(3)
user_sel = {}
for idx, feat in enumerate(visible_feats):
    col = cols[idx % 3]
    options = list(feature_maps[feat].values())
    label = feat.replace('-', ' ').capitalize()
    user_sel[feat] = col.selectbox(
        label, options, key=f"{subset}_{feat}"
    )

# ─── Map user selections back to original codes ───────────────────────────────
coded = {
    feat: next(
        code for code, desc in feature_maps[feat].items()
        if desc == user_sel[feat]
    )
    for feat in user_sel
}

# ─── Prediction and display function ─────────────────────────────────────────
def predict_and_display(pipes, df_input):
    """Run each pipeline and show edibility probabilities."""
    cols_out = st.columns(len(pipes))
    for (name, pipe), col in zip(pipes.items(), cols_out):
        with col:
            prob_poison = pipe.predict_proba(df_input)[0, 1]
            prob_edible = 1 - prob_poison
            if prob_edible >= 0.95:
                label = "✅ Essbar"
            elif prob_poison >= 0.95:
                label = "💀 Giftig"
            else:
                label = "❓ Unsicher"

            col.subheader(name)
            col.write(label)
            col.write(f"P(Essbar) = {prob_edible:.2f}")
            col.write(f"P(Giftig) = {prob_poison:.2f}")

# ─── Trigger prediction on button click ──────────────────────────────────────
if st.button("Vorhersage ausführen"):
    df_input = pd.DataFrame([coded])
    st.markdown("---")
    st.header(f"🔍 Ergebnisse ({subset})")
    predict_and_display(pipelines, df_input)
    st.markdown("---")
    st.caption("*Hinweis: Kein Ersatz für professionelle Pilzbestimmung.*")
