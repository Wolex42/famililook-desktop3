# Security Policy

## Sensitive Data

This repository does NOT contain:
- ✅ API keys or secrets
- ✅ User data or analytics
- ✅ Face recognition models (downloaded at runtime)
- ✅ Virtual environment files
- ✅ Environment variables

## What Gets Deployed

Only these files are deployed to production:
- Source code (`app/*.py`)
- Dependencies list (`requirements.txt`)
- Configuration (`Procfile`)
- Documentation (`README.md`)

Models are downloaded automatically by InsightFace on first run.

## Reporting Security Issues

Contact: [your-email@example.com]