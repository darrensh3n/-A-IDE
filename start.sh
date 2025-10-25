#!/bin/bash

# Drowning Detection System Startup Script

echo "=========================================="
echo "Drowning Detection System Startup"
echo "=========================================="

# Function to start backend
start_backend() {
    echo "Starting FastAPI backend..."
    cd backend
    
    # Check if Python 3.13 is available
    if ! command -v python3.13 &> /dev/null; then
        echo "Error: Python 3.13 is not installed or not in PATH"
        echo "Please install Python 3.13 to use this backend"
        exit 1
    fi
    
    # Check if venv exists, create if not
    if [ ! -d ".venv" ]; then
        echo "Virtual environment not found. Creating .venv with Python 3.13..."
        python3.13 -m venv .venv
        echo "Virtual environment created."
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Check if requirements are installed
    if ! python -c "import fastapi" 2>/dev/null; then
        echo "Installing Python dependencies..."
        pip install -q --upgrade pip
        pip install -q -r requirements.txt
        pip install -q -r scripts/requirements.txt
        echo "Dependencies installed."
    fi
    
    # Start the backend server
    python api_server.py &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"
    cd ..
}

# Function to start frontend
start_frontend() {
    echo "Starting Next.js frontend..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    echo "Frontend started with PID: $FRONTEND_PID"
    cd ..
}

# Start services
start_backend
sleep 3
start_frontend

echo ""
echo "Services started successfully!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap 'echo "Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
