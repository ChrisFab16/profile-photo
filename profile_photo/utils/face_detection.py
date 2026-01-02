"""
OpenCV-based face detection module to replace AWS Rekognition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np

from .face_models import DetectFacesResp, DetectLabelsResp, BoundingBox, FaceDetail
from ..log import LOG


@dataclass
class FaceDetector:
    """
    Face detector using OpenCV's DNN face detection model.
    Uses a pre-trained model for accurate face detection.
    """
    
    def __init__(self, model_path: str | Path | None = None, confidence_threshold: float = 0.5):
        """
        Initialize the face detector.
        
        :param model_path: Path to the OpenCV DNN face detection model files.
                          If None, will use default model files.
        :param confidence_threshold: Minimum confidence for face detection (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        
        # Default to Haar Cascade (built into OpenCV, no external files needed)
        if model_path is None:
            self.use_haar = True
            self.use_dnn = False
            self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
            LOG.info('Using Haar Cascade face detection (default)')
            return
        
        # Use DNN-based face detection if model path is provided
        self.use_dnn = True
        self.use_haar = False
        
        if isinstance(model_path, str):
            model_path = Path(model_path)
        
        # For DNN, we need prototxt and model files
        # If only directory is provided, look for standard filenames
        if model_path.is_dir():
            prototxt = model_path / 'deploy.prototxt'
            model_file = model_path / 'res10_300x300_ssd_iter_140000.caffemodel'
        else:
            # Assume model_path is the .caffemodel file
            prototxt = model_path.parent / 'deploy.prototxt'
            model_file = model_path
        
        if prototxt.exists() and model_file.exists():
            self.net = cv.dnn.readNetFromCaffe(str(prototxt), str(model_file))
            LOG.info('Loaded DNN face detection model')
        else:
            # Fallback to Haar Cascade
            LOG.warning('DNN model files not found, falling back to Haar Cascade')
            self.use_haar = True
            self.use_dnn = False
            self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, im_bytes: bytes | None = None, image: np.ndarray | None = None,
                     debug: bool = False) -> DetectFacesResp:
        """
        Detect faces in an image.
        
        :param im_bytes: Image data as bytes
        :param image: OpenCV image (numpy array) - alternative to im_bytes
        :param debug: Enable debug logging
        :return: DetectFacesResp containing detected faces
        """
        if image is None:
            if im_bytes is None:
                raise ValueError("Either im_bytes or image must be provided")
            # Decode image from bytes
            img_array = np.frombuffer(im_bytes, dtype=np.uint8)
            image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        height, width = image.shape[:2]
        
        if self.use_dnn:
            # Use DNN-based face detection
            face_details = self._detect_faces_dnn(image, width, height, debug)
        else:
            # Use Haar Cascade face detection
            face_details = self._detect_faces_haar(image, width, height, debug)
        
        if debug:
            LOG.info('Detected %d face(s)', len(face_details))
            if face_details:
                LOG.info('Face detection response: %s', 
                        json.dumps([{
                            'bounding_box': {
                                'left': f.bounding_box.left,
                                'top': f.bounding_box.top,
                                'width': f.bounding_box.width,
                                'height': f.bounding_box.height
                            },
                            'confidence': f.confidence
                        } for f in face_details], indent=2))
        
        return DetectFacesResp(face_details=face_details)
    
    def _detect_faces_dnn(self, image: np.ndarray, width: int, height: int, 
                          debug: bool) -> List[FaceDetail]:
        """Detect faces using DNN model."""
        # Create blob from image
        blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
                                    (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        face_details = []
        
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            
            if confidence > self.confidence_threshold:
                # Get bounding box coordinates (normalized 0-1)
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Convert to normalized coordinates (left, top, width, height)
                left = x1 / width
                top = y1 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                bounding_box = BoundingBox(
                    left=left,
                    top=top,
                    width=box_width,
                    height=box_height
                )
                
                # Create FaceDetail (simplified - no landmarks, emotions, etc.)
                face_detail = FaceDetail(
                    bounding_box=bounding_box,
                    confidence=confidence
                )
                
                face_details.append(face_detail)
        
        # Sort by confidence (highest first)
        face_details.sort(key=lambda f: f.confidence, reverse=True)
        
        return face_details
    
    def _detect_faces_haar(self, image: np.ndarray, width: int, height: int,
                           debug: bool) -> List[FaceDetail]:
        """Detect faces using Haar Cascade (fallback method)."""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_details = []
        
        for (x, y, w, h) in faces:
            # Convert to normalized coordinates
            left = x / width
            top = y / height
            box_width = w / width
            box_height = h / height
            
            bounding_box = BoundingBox(
                left=left,
                top=top,
                width=box_width,
                height=box_height
            )
            
            # Haar Cascade doesn't provide confidence, use default
            face_detail = FaceDetail(
                bounding_box=bounding_box,
                confidence=0.8  # Default confidence for Haar Cascade
            )
            
            face_details.append(face_detail)
        
        return face_details
    
    def detect_labels(self, im_bytes: bytes | None = None, image: np.ndarray | None = None,
                     face: FaceDetail | None = None, debug: bool = False) -> DetectLabelsResp:
        """
        Detect person/body in the image. 
        This is a simplified version that estimates person bounding box from face.
        
        :param im_bytes: Image data as bytes
        :param image: OpenCV image (numpy array)
        :param face: Detected face to help estimate person bounding box
        :param debug: Enable debug logging
        :return: DetectLabelsResp containing person detection
        """
        # Since we don't have a person/body detector, we'll estimate
        # the person bounding box from the face bounding box
        # This is a reasonable approximation for headshot/profile photos
        
        from .face_models import Label, Instance
        
        if face is None:
            # No face provided, return empty labels
            return DetectLabelsResp(labels=[])
        
        # Estimate person box: typically 2-3x the face height and centered
        # This is a heuristic based on typical human proportions
        face_box = face.bounding_box
        
        # Person box is typically wider and taller than face
        # Estimate: person is about 2.5x the face height, and 1.5x the face width
        person_height = min(face_box.height * 2.5, 1.0)  # Cap at image height
        person_width = min(face_box.width * 1.8, 1.0)  # Cap at image width
        
        # Center the person box on the face, but extend downward
        person_left = max(0, face_box.left - (person_width - face_box.width) / 2)
        person_top = max(0, face_box.top - face_box.height * 0.2)  # Extend slightly upward
        
        # Adjust if out of bounds
        if person_left + person_width > 1.0:
            person_left = 1.0 - person_width
        if person_top + person_height > 1.0:
            person_height = 1.0 - person_top
        
        person_box = BoundingBox(
            left=person_left,
            top=person_top,
            width=person_width,
            height=person_height
        )
        
        person_label = Label(
            name='Person',
            confidence=0.9,  # High confidence since we detected a face
            instances=[Instance(
                bounding_box=person_box,
                confidence=0.9
            )],
            parents=[]
        )
        
        if debug:
            LOG.info('Estimated person bounding box from face')
        
        return DetectLabelsResp(labels=[person_label])

