import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import csv

# Analytics storage
ANALYTICS_DIR = Path("analytics_data")
ANALYTICS_DIR.mkdir(exist_ok=True)

class AnalyticsLogger:
    def __init__(self):
        self.session_start = time.time()
        
    def _get_log_file(self, event_type: str) -> Path:
        """Get log file for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        return ANALYTICS_DIR / f"{event_type}_{today}.jsonl"
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to JSONL file"""
        log_file = self._get_log_file(event_type)
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "session_time": time.time() - self.session_start,
            **data
        }
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')
    
    # === Upload Events ===
    def log_upload(self, ip: str, filename: str, file_size: int, content_type: str):
        """Log file upload"""
        self._log_event("upload", {
            "ip": self._anonymize_ip(ip),
            "filename_ext": Path(filename).suffix,
            "file_size_kb": round(file_size / 1024, 2),
            "content_type": content_type
        })
    
    # === Detection Events ===
    def log_detection(self, ip: str, faces_detected: int, processing_time: float, success: bool):
        """Log face detection"""
        self._log_event("detection", {
            "ip": self._anonymize_ip(ip),
            "faces_detected": faces_detected,
            "processing_time_ms": round(processing_time * 1000, 2),
            "success": success
        })
    
    # === Analysis Events ===
    def log_analysis(self, ip: str, child_count: int, processing_time: float, 
                     results: Optional[Dict] = None):
        """Log resemblance analysis"""
        self._log_event("analysis", {
            "ip": self._anonymize_ip(ip),
            "child_count": child_count,
            "processing_time_ms": round(processing_time * 1000, 2),
            "has_results": results is not None,
            "avg_confidence": self._calculate_avg_confidence(results) if results else None
        })
    
    # === Feature Usage ===
    def log_feature_use(self, ip: str, feature: str, details: Optional[Dict] = None):
        """Log feature usage (cards, keepsakes, etc.)"""
        self._log_event("feature_use", {
            "ip": self._anonymize_ip(ip),
            "feature": feature,
            "details": details or {}
        })
    
    # === Errors ===
    def log_error(self, ip: str, error_type: str, error_message: str, 
                  endpoint: Optional[str] = None):
        """Log errors"""
        self._log_event("error", {
            "ip": self._anonymize_ip(ip),
            "error_type": error_type,
            "error_message": error_message[:200],  # Truncate long messages
            "endpoint": endpoint
        })
    
    # === Session Events ===
    def log_session_start(self, ip: str, plan: str):
        """Log when user starts a session"""
        self._log_event("session_start", {
            "ip": self._anonymize_ip(ip),
            "plan": plan
        })
    
    def log_session_end(self, ip: str, duration: float, actions_count: int):
        """Log when session ends"""
        self._log_event("session_end", {
            "ip": self._anonymize_ip(ip),
            "duration_seconds": round(duration, 2),
            "actions_count": actions_count
        })
    
    # === Helper Methods ===
    @staticmethod
    def _anonymize_ip(ip: str) -> str:
        """Anonymize IP for privacy (hash it)"""
        import hashlib
        return hashlib.sha256(ip.encode()).hexdigest()[:12]
    
    @staticmethod
    def _calculate_avg_confidence(results: Optional[Dict]) -> Optional[float]:
        """Calculate average confidence from results"""
        if not results or 'results' not in results:
            return None
        
        confidences = [r.get('confidence', 0) for r in results['results']]
        return round(sum(confidences) / len(confidences), 2) if confidences else None
    
    # === Summary Reports ===
    def generate_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Generate summary for a specific date"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        summary = {
            "date": date,
            "uploads": 0,
            "detections": 0,
            "analyses": 0,
            "errors": 0,
            "unique_users": set(),
            "total_faces_detected": 0,
            "avg_processing_time": []
        }
        
        # Read all event types for the day
        for event_type in ["upload", "detection", "analysis", "error"]:
            log_file = ANALYTICS_DIR / f"{event_type}_{date}.jsonl"
            if not log_file.exists():
                continue
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        summary["unique_users"].add(event.get("ip", "unknown"))
                        
                        if event_type == "upload":
                            summary["uploads"] += 1
                        elif event_type == "detection":
                            summary["detections"] += 1
                            summary["total_faces_detected"] += event.get("faces_detected", 0)
                            if "processing_time_ms" in event:
                                summary["avg_processing_time"].append(event["processing_time_ms"])
                        elif event_type == "analysis":
                            summary["analyses"] += 1
                        elif event_type == "error":
                            summary["errors"] += 1
                    except json.JSONDecodeError:
                        continue
        
        summary["unique_users"] = len(summary["unique_users"])
        if summary["avg_processing_time"]:
            summary["avg_processing_time"] = round(
                sum(summary["avg_processing_time"]) / len(summary["avg_processing_time"]), 2
            )
        else:
            summary["avg_processing_time"] = 0
        
        return summary

# Global analytics instance
analytics = AnalyticsLogger()