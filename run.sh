#!/usr/bin/env bash
# Start ORBIT: seed the demo data if needed, then launch the console.
set -e
cd "$(dirname "$0")"
python -m app.seed
exec streamlit run frontend/streamlit_app.py "$@"
