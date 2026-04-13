#!/bin/bash
# Bookify — Local Development Startup
# Starts Flask backend + Vite dev server concurrently
# Usage: ./start_dev.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  Bookify — Starting Development Server"
echo "========================================="

# Check .env exists
if [ ! -f .env ]; then
    echo "WARNING: .env file not found. Copy .env.example to .env and fill in your keys."
fi

# Check frontend .env exists
if [ ! -f frontend/.env ]; then
    echo "WARNING: frontend/.env not found. Copy frontend/.env.example to frontend/.env."
fi

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "  Python venv activated"
fi

# Install frontend deps if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "  Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Start Vite dev server in background
echo "  Starting Vite dev server on http://localhost:5173 ..."
cd frontend && npm run dev &
VITE_PID=$!
cd ..

# Start Flask backend
echo "  Starting Flask backend on http://localhost:5001 ..."
echo ""
echo "  Open http://localhost:5173 in your browser"
echo "  Press Ctrl+C to stop both servers"
echo "========================================="
python app.py

# Cleanup: kill Vite when Flask exits
kill $VITE_PID 2>/dev/null
