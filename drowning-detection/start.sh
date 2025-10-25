#!/bin/bash

# Drowning Detection System Startup Script

echo "=========================================="
echo "Drowning Detection System Startup"
echo "=========================================="

# Function to start backend
start_backend() {
    echo "Starting FastAPI backend..."
    cd backend
    python3 api_server.py &
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
