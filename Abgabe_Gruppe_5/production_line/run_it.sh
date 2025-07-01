#!/usr/bin/env bash
# Startet Backend (Flask) und Frontend (Streamlit) gleichzeitig

# 1) Backend im Hintergrund starten
echo "Starte Backend..."
python3 backend.py &
BACKEND_PID=$!

# 2) Kurze Pause, damit Flask hochfährt
sleep 3

echo "Starte Streamlit Frontend..."
# 3) Frontend starten (Streamlit blockiert hier das Terminal)
streamlit run frontend.py

# Wenn Streamlit beendet wird, Backend ebenfalls stoppen
echo "Beende Backend (PID=$BACKEND_PID)..."
kill $BACKEND_PID
