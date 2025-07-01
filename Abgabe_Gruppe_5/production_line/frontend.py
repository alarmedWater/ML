import streamlit as st
import pandas as pd
import requests
import json

# ---------------------- Konfiguration ----------------------
API_URL = 'http://localhost:5000'

st.set_page_config(page_title="Mushroom Predictor", layout="wide")
st.title("🍄 Mushroom Edibility Predictor")

# ---------------------- Feature Mapping ----------------------
feature_maps = {
    'cap-shape': {'b': 'bell (Glocke)', 'c': 'conical (Kegel)', 'x': 'convex (konvex)', 'f': 'flat (flach)', 'k': 'knobbed (knollig)', 's': 'sunken (eingedrückt)'},
    'cap-surface': {'f': 'fibrous (fasrig)', 'g': 'grooves (Rillen)', 'y': 'scaly (schuppig)', 's': 'smooth (glatt)'},
    'cap-color': {'n': 'brown (braun)', 'b': 'buff (gelblich)', 'c': 'cinnamon (zimt)', 'g': 'gray (grau)', 'r': 'green (grün)', 'p': 'pink (rosa)', 'u': 'purple (lila)', 'e': 'red (rot)', 'w': 'white (weiß)', 'y': 'yellow (gelb)'},
    'bruises': {'t': 'bruises (verfärbt)', 'f': 'no (keine Verfärbung)'},
    'odor': {'a': 'almond (mandel)', 'l': 'anise (anis)', 'c': 'creosote (teer)', 'y': 'fishy (fischig)', 'f': 'foul (faulig)', 'm': 'musty (muffig)', 'n': 'none (kein)', 'p': 'pungent (stechend)', 's': 'spicy (würzig)'},
    'gill-attachment': {'a': 'attached (anliegend)', 'd': 'descending (herablaufend)', 'f': 'free (frei)', 'n': 'notched (eingekerbt)'},
    'gill-spacing': {'c': 'close (eng)', 'w': 'crowded (überfüllt)', 'd': 'distant (weit)'},
    'gill-size': {'b': 'broad (breit)', 'n': 'narrow (schmal)'},
    'gill-color': {'k': 'black (schwarz)', 'n': 'brown (braun)', 'b': 'buff (gelblich)', 'h': 'chocolate (schokolade)', 'g': 'gray (grau)', 'r': 'green (grün)', 'o': 'orange (orange)', 'p': 'pink (rosa)', 'u': 'purple (lila)', 'e': 'red (rot)', 'w': 'white (weiß)', 'y': 'yellow (gelb)'},
    'stalk-shape': {'e': 'enlarging (erweiternd)', 't': 'tapering (verjüngend)'},
    'stalk-root': {'b': 'bulbous (knollig)', 'c': 'club (keulenförmig)', 'u': 'cup (becherförmig)', 'e': 'equal (gleich)', 'z': 'rhizomorphs', 'r': 'rooted (verwurzelt)', '?': 'missing (fehlend)'},
    'stalk-surface-above-ring': {'f': 'fibrous (fasrig)', 'y': 'scaly (schuppig)', 'k': 'silky (seidig)', 's': 'smooth (glatt)'},
    'stalk-surface-below-ring': {'f': 'fibrous (fasrig)', 'y': 'scaly (schuppig)', 'k': 'silky (seidig)', 's': 'smooth (glatt)'},
    'stalk-color-above-ring': {'n': 'brown (braun)', 'b': 'buff (gelblich)', 'c': 'cinnamon (zimt)', 'g': 'gray (grau)', 'o': 'orange (orange)', 'p': 'pink (rosa)', 'e': 'red (rot)', 'w': 'white (weiß)', 'y': 'yellow (gelb)'},
    'stalk-color-below-ring': {'n': 'brown (braun)', 'b': 'buff (gelblich)', 'c': 'cinnamon (zimt)', 'g': 'gray (grau)', 'o': 'orange (orange)', 'p': 'pink (rosa)', 'e': 'red (rot)', 'w': 'white (weiß)', 'y': 'yellow (gelb)'},
    'veil-type': {'p': 'partial (teilweise)', 'u': 'universal (ganz)'},
    'veil-color': {'n': 'brown (braun)', 'o': 'orange (orange)', 'w': 'white (weiß)', 'y': 'yellow (gelb)'},
    'ring-number': {'n': 'none (kein Ring)', 'o': 'one (ein)', 't': 'two (zwei)'},
    'ring-type': {'c': 'cobwebby (netzig)', 'e': 'evanescent (flüchtig)', 'f': 'flaring (ausladend)', 'l': 'large (groß)', 'n': 'none (kein)', 'p': 'pendant (hängend)', 's': 'sheathing (hüllend)', 'z': 'zone (gegliedert)'},
    'spore-print-color': {'k': 'black (schwarz)', 'n': 'brown (braun)', 'b': 'buff (gelblich)', 'h': 'chocolate (schokolade)', 'r': 'green (grün)', 'o': 'orange (orange)', 'u': 'purple (lila)', 'w': 'white (weiß)', 'y': 'yellow (gelb)'},
    'population': {'a': 'abundant (reichlich)', 'c': 'clustered (gebündelt)', 'n': 'numerous (zahlreich)', 's': 'scattered (vereinzelt)', 'v': 'several (einige)', 'y': 'solitary (einzeln)'},
    'habitat': {'g': 'grasses (gräser)', 'l': 'leaves (blätter)', 'm': 'meadows (wiesen)', 'p': 'paths (wege)', 'u': 'urban (städtisch)', 'w': 'waste (abfall)', 'd': 'woods (wald)'}
}

# ---------------------- UI Eingaben ----------------------
cols = st.columns(3)
user_input = {}
visible_feats = list(feature_maps.keys())
for idx, feat in enumerate(visible_feats):
    col = cols[idx % 3]
    opts = list(feature_maps[feat].values())
    label = feat.replace('-', ' ').capitalize()
    user_input[feat] = col.selectbox(label, opts, key=feat)
# Map back to codes
coded = {feat: next(code for code, desc in feature_maps[feat].items() if desc == user_input[feat])
         for feat in visible_feats}

# ---------------------- API Calls ----------------------

def get_prediction(record: dict) -> dict:
    """Sendet eine Vorhersage-Anfrage an das Backend und gibt das Ergebnis zurück."""
    try:
        resp = requests.post(f"{API_URL}/predict", json=record)
        return resp.json()
    except Exception as e:
        return {'error': str(e)}


def get_errors() -> list:
    """Holt die aktuellen Fehler-Messages vom Backend."""
    try:
        resp = requests.get(f"{API_URL}/errors")
        data = resp.json()
        return data.get('errors', [])
    except:
        return []

# ---------------------- Ergebnisse anzeigen ----------------------
if st.button("Vorhersage ausführen"):
    st.markdown("---")
    df_record = coded
    result = get_prediction(df_record)
    if 'error' in result:
        st.error(f"Fehler: {result['error']}")
    else:
        st.header("🔍 Vorhersage")
        for model, preds in result.items():
            label = preds[0] if isinstance(preds, list) else preds
            color = "✅ Essbar" if label == "edible" else "💀 Giftig"
            st.subheader(f"{model}")
            st.write(color)
    # Fehler-Log
    errs = get_errors()
    if errs:
        st.markdown("---")
        st.subheader("🛑 Backend-Fehler")
        for e in errs:
            st.write(f"[{e['time']}] {e['message']}")

st.caption("*Hinweis: Kein Ersatz für professionelle Pilzbestimmung.*")
