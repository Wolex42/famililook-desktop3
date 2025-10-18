# FamiliLook Backend API

FastAPI backend for face detection, recognition, and family resemblance analysis.

## Features

- Face detection using InsightFace
- Face embedding generation
- Parent-child resemblance analysis
- Analytics tracking
- CORS enabled for frontend integration

## Setup

### 1. Create Virtual Environment
```cmd
cd C:\Users\wole\Documents\famililook-desktop3
python -m venv .venv311
```

### 2. Activate Virtual Environment
```cmd
.venv311\Scripts\activate
```

### 3. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 4. Run Server
```cmd
uvicorn app.main:app --reload --port 8008
```

Or using Python directly:
```cmd
.venv311\Scripts\python.exe -m uvicorn app.main:app --reload --port 8008
```

## API Endpoints

### Health & Status
- `GET /` - Root endpoint
- `GET /healthz` - Health check
- `GET /status` - Engine status
- `GET /version` - API version info

### Face Detection & Analysis
- `POST /detect` - Detect faces in image
- `POST /embed` - Generate face embeddings

### Analytics
- `GET /analytics/summary` - Daily analytics summary
- `GET /analytics/dashboard` - Analytics dashboard
- `POST /analytics/track` - Track frontend events

### Documentation
- Visit `http://localhost:8008/docs` for interactive API docs

## Directory Structure
```
famililook-desktop3/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   └── analytics.py     # Analytics logger
├── analytics_data/      # Analytics logs (created automatically)
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Analytics Data

Analytics are stored in `analytics_data/` as JSONL files:
- `upload_YYYY-MM-DD.jsonl`
- `detection_YYYY-MM-DD.jsonl`
- `analysis_YYYY-MM-DD.jsonl`
- `feature_use_YYYY-MM-DD.jsonl`
- `error_YYYY-MM-DD.jsonl`

View daily summary: `http://localhost:8008/analytics/summary`

## Development

### Testing
```cmd
# Test health endpoint
curl http://localhost:8008/healthz

# Test detection with image
curl -X POST -F "file=@test_image.jpg" http://localhost:8008/detect
```

### Virtual Environment Commands
```cmd
# Activate
.venv311\Scripts\activate

# Deactivate
deactivate

# Install new package
pip install package-name

# Update requirements
pip freeze > requirements.txt
```

## Troubleshooting

### Models Not Loading
- InsightFace downloads models on first use (~300MB)
- Models stored in `~/.insightface/models/`
- First run may take 5-10 minutes

### Port Already in Use
```cmd
# Kill process on port 8008
netstat -ano | findstr :8008
taskkill /PID <PID> /F
```

### Import Errors
```cmd
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## License

Private project - All rights reserved