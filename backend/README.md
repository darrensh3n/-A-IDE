# Drowning Detection Backend

FastAPI backend service for AI-powered drowning detection using YOLOv8.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Setup the model:
```bash
python scripts/setup_model.py
```

3. Start the server:
```bash
python api_server.py
```

## API Endpoints

- `GET /` - Health check
- `GET /api/health` - Detailed health check
- `POST /api/detect-drowning` - Detect drowning in uploaded image/video
- `POST /api/detect-frame` - Optimized endpoint for real-time camera feed

## API Documentation

Once running, visit: http://localhost:8000/docs

## Model

The system uses YOLOv8 for object detection. For production drowning detection, train a custom model on drowning/swimming datasets.
