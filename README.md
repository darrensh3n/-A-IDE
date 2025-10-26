# Drowning Detection System

An AI-powered lifeguard assistant that uses real-time video analysis to detect potential drowning incidents. Built with YOLOv8, FastAPI, and Next.js, it provides instant alerts, visual detection feedback, and a live monitoring dashboard — helping make swimming environments safer through intelligent automation.

## Project Structure

```
drowning-detection/
├── backend/           # FastAPI backend service
│   ├── api_server.py  # Main FastAPI application
│   ├── scripts/       # Setup and utility scripts
│   ├── models/        # AI model files
│   └── requirements.txt
├── frontend/          # Next.js frontend application
│   ├── app/           # Next.js app directory
│   ├── components/    # React components
│   ├── lib/           # Utility libraries
│   └── package.json
└── README.md          # This file
```

## Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python scripts/setup_model.py
python api_server.py
```

### Frontend Setup
```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

## Services

- **Backend**: http://localhost:8000 (FastAPI + YOLOv8)
- **Frontend**: http://localhost:3000 (Next.js dashboard)
- **API Docs**: http://localhost:8000/docs

## Features

- Real-time video monitoring
- Image/video upload for analysis
- Camera integration
- Detection visualization with bounding boxes
- Alert history tracking
- System statistics dashboard

## Technology Stack

- **Backend**: FastAPI, YOLOv8, OpenCV, PyTorch
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **AI Model**: Ultralytics YOLOv8 for object detection
