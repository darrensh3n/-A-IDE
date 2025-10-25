# Drowning Detection Backend

FastAPI backend service for AI-powered drowning detection using YOLOv8.

**Requirements:** Python 3.13

## Setup

### Automatic Setup (Recommended)

The simplest way to set up and run the backend is using the startup script from the root directory:

```bash
# From the root directory
./start.sh
```

This will automatically:
- Create a virtual environment using Python 3.13 if it doesn't exist
- Install all dependencies
- Start the backend server

### Manual Setup

If you prefer manual setup:

1. **Create and activate virtual environment:**
```bash
cd backend
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r scripts/requirements.txt
```

Or use the provided setup script:
```bash
cd backend
./setup.sh
```

3. **Setup the model:**
```bash
python scripts/setup_model.py
```

4. **Start the server:**
```bash
python api_server.py
```

### Virtual Environment

The backend uses a Python virtual environment located at `backend/.venv/`. This is automatically created by the startup script. To manually work with the venv:

- **Activate**: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
- **Deactivate**: `deactivate`

## API Endpoints

- `GET /` - Health check
- `GET /api/health` - Detailed health check
- `POST /api/detect-drowning` - Detect drowning in uploaded image/video
- `POST /api/detect-frame` - Optimized endpoint for real-time camera feed

## API Documentation

Once running, visit: http://localhost:8000/docs

## Model

The system uses YOLOv8 for object detection. For production drowning detection, train a custom model on drowning/swimming datasets.
