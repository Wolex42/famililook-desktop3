from fastapi import Request, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from datetime import datetime, timedelta
import json
from typing import Optional
from pathlib import Path

# Load security config
config_path = Path(__file__).parent.parent / "security-config.json"
with open(config_path, 'r') as f:
    security_config = json.load(f)['testing']

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Track analysis counts per IP
analysis_tracker = {}

def check_test_session_active() -> bool:
    """Check if testing session is still valid"""
    if not security_config['enabled']:
        return True  # If disabled, allow all
    
    if security_config.get('testEndTime'):
        end_time = datetime.fromisoformat(security_config['testEndTime'])
        if datetime.now() > end_time:
            return False
    
    return True

async def verify_test_session(request: Request):
    """Dependency to verify test session is active"""
    if not check_test_session_active():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Testing session has ended. Thank you for testing!"
        )

async def check_analysis_limit(request: Request, max_analyses: int = 5):
    """Dependency to check analysis limit per IP"""
    ip = request.client.host
    
    if ip not in analysis_tracker:
        analysis_tracker[ip] = 0
    
    if analysis_tracker[ip] >= max_analyses:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Analysis limit reached ({max_analyses} max for testing). Please upgrade for unlimited analysis."
        )
    
    analysis_tracker[ip] += 1

async def validate_image_file(file) -> None:
    """Validate uploaded image file"""
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Check file type
    allowed_types = security_config['allowedFileTypes']
    content_type = file.content_type
    
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{content_type}'. Allowed: {', '.join(allowed_types)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    max_size = security_config['maxFileSize']
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large ({file_size / 1024 / 1024:.2f}MB). Maximum: {max_size / 1024 / 1024}MB"
        )

def initialize_test_session(hours: int = 2):
    """Initialize test session with expiration time"""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=hours)
    
    security_config['testStartTime'] = start_time.isoformat()
    security_config['testEndTime'] = end_time.isoformat()
    
    with open(config_path, 'w') as f:
        json.dump({'testing': security_config}, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🔒 SECURE TEST SESSION INITIALIZED")
    print(f"{'='*60}")
    print(f"   Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {hours} hour(s)")
    print(f"{'='*60}\n")
    
    return end_time

def cleanup_temp_files():
    """Clean up any temporary files from previous sessions"""
    temp_dir = Path("temp")
    if temp_dir.exists():
        count = 0
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                    count += 1
                except Exception as e:
                    print(f"⚠️  Error deleting {file_path.name}: {e}")
        if count > 0:
            print(f"🗑️  Cleaned up {count} temporary file(s)")