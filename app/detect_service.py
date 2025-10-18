"""
Face detection service using InsightFace
"""
import numpy as np
from insightface.app import FaceAnalysis
import cv2
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DetectService:
    """Face detection and embedding service"""
    
    def __init__(self):
        self.fa = None
        self.ready = False
        
    def initialize(self):
        """Load InsightFace models"""
        try:
            logger.info("Loading InsightFace models...")
            self.fa = FaceAnalysis(
                name='buffalo_l',
                allowed_modules=['detection', 'recognition']
            )
            self.fa.prepare(ctx_id=-1, det_size=(640, 640))
            self.ready = True
            logger.info("✅ Face detection models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            self.ready = False
            raise
    
    def detect_faces(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Detect faces in image
        
        Args:
            image_bytes: Image as bytes
            
        Returns:
            List of detected faces with embeddings
        """
        if not self.ready:
            raise RuntimeError("DetectService not initialized")
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return []
        
        # Detect faces
        faces = self.fa.get(img)
        
        # Format results
        results = []
        for i, face in enumerate(faces):
            results.append({
                'face_id': i,
                'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None,
                'confidence': float(face.det_score) if hasattr(face, 'det_score') else 1.0,
                'embedding': face.embedding.tolist() if hasattr(face, 'embedding') else None,
                'age': int(face.age) if hasattr(face, 'age') else None,
                'gender': int(face.gender) if hasattr(face, 'gender') else None,
            })
        
        return results
    
    def get_embedding(self, image_bytes: bytes, bbox: Optional[List[float]] = None) -> Optional[np.ndarray]:
        """
        Get face embedding from image
        
        Args:
            image_bytes: Image as bytes
            bbox: Optional bounding box [x1, y1, x2, y2]
            
        Returns:
            Face embedding as numpy array or None
        """
        faces = self.detect_faces(image_bytes)
        
        if not faces:
            return None
        
        # Return first face embedding
        embedding = faces[0].get('embedding')
        if embedding:
            return np.array(embedding)
        
        return None

# Global instance
detect_service = DetectService()