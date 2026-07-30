#!/bin/bash
# Run script for Mac / Linux
# Double-click or run: bash run.sh

echo ""
echo "  Sales Analysis Agent"
echo "  ─────────────────────────────"
echo "  Installing dependencies..."
pip install -r requirements.txt -q

echo "  Starting server..."
echo "  Open http://localhost:5000 in your browser"
echo ""
cd ui && python app.py
