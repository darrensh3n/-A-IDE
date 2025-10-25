# Aide

A full-stack application with a Next.js frontend and FastAPI backend.

## Project Structure

```
aide/
├── frontend/          # Next.js frontend application
├── backend/           # FastAPI backend application
└── README.md
```

## Prerequisites

- **Node.js** (v18 or higher) and **npm** for the frontend
- **Python** (v3.8 or higher) and **pip** for the backend

## Getting Started

### Backend Setup

1. Navigate to the backend directory:

```bash
cd backend
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the development server:

```bash
python main.py
```

The backend server will start at [http://localhost:8000](http://localhost:8000)

You can test the API by visiting [http://localhost:8000/ping](http://localhost:8000/ping)

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Run the development server:

```bash
npm run dev
```

The frontend will start at [http://localhost:3000](http://localhost:3000)

Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

### Running Both Services

For local development, you'll need to run both the backend and frontend servers simultaneously in separate terminal windows:

**Terminal 1 (Backend):**

```bash
cd backend
source venv/bin/activate  # if using virtual environment
python main.py
```

**Terminal 2 (Frontend):**

```bash
cd frontend
npm run dev
```

## API Documentation

Once the backend is running, you can access the interactive API documentation at:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Development

### Frontend

- Built with [Next.js 16](https://nextjs.org/)
- React 19.2.0
- TypeScript
- Tailwind CSS v4
- Edit pages in `frontend/src/app/`

### Backend

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Python with Uvicorn server
- CORS enabled for frontend communication
- Edit API routes in `backend/app/main.py`

## Available Scripts

### Frontend

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server

### Backend

- `python main.py` - Start development server with auto-reload
