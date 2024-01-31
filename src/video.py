import os
import json
import argparse
import numpy as np
from datetime import datetime
from time import time, sleep
from threading import Thread
from statistics import mode
from pprint import pprint

class VideoInference:
    
    def __init__(self, frames, face_detection_model, emotion_detection_model, img_size=512):
        self.face_detection_model = face_detection_model
        self.emotion_detection_model = emotion_detection_model
        self.img_size = img_size
        self.frames = frames
        self.frame = frames[0]
        self.emotion = 'None'
        self.faces = [[0, 0, 0, 0]]
        
        self.data = dict()
            
        # Thread
        self.stop_thread = False
        self.t = Thread(target=self.emotion_detection)
        self.t.daemon = True
        self.t.start()
        
    def class_convert(self, emotion):
        happy_emotions = ["happy", "surprise"]
        sad_emotions = ["sad", "fear", "anger", "disgust"]
        
        if emotion in happy_emotions:
            return "happy"
        elif emotion in sad_emotions:
            return "sad"
        else: return "neutral"
            
    def zoom_out_bounding_box(self, face):
        x1, y1, x2, y2 = face
                
        percent_zoom_out = 0.2
        x1_zoom = int(x1 - x1*percent_zoom_out)
        y1_zoom = int(y1 - y1*percent_zoom_out)
        x2_zoom = int(x2 + x2*percent_zoom_out)
        y2_zoom = int(y2 + y2*percent_zoom_out)
        return (x1_zoom, y1_zoom, x2_zoom, y2_zoom)
            
    def emotion_detection(self):
        
        timestamp = time()
        
        while True:
            
            result = self.face_detection_model.track(self.frame, persist=True, verbose=False, imgsz=self.img_size)
            self.faces = result[0].boxes.xyxy.numpy()
            
            for face in self.faces:
                
                x1_zoom, y1_zoom, x2_zoom, y2_zoom = self.zoom_out_bounding_box(face)
                try:
                    roi = self.frame[y1_zoom:y2_zoom, x1_zoom:x2_zoom]
                except TypeError:
                    continue
                pred = self.emotion_detection_model.detect_emotion_for_single_frame(roi)
                
                try:
                    self.emotion = self.class_convert(pred[0]['emo_label'])
                except IndexError:
                    continue
            
            if timestamp + 1 < time():
                for i, face in enumerate(self.faces):
                    if f'face_{i}' not in self.data:
                        self.data[f'face_{i}'] = {
                            'timestamp': list(),
                            'coordinates': list(),
                            'emotion': list(),
                        }
                    self.data[f'face_{i}']['timestamp'].append(datetime.now().strftime("%m/%d/%Y %H:%M:%S")),
                    self.data[f'face_{i}']['coordinates'].append(face.tolist()),
                    self.data[f'face_{i}']['emotion'].append(self.emotion)
                timestamp = time()
                
            if self.stop_thread:
                break

    def inference(self):
        
        for self.frame in self.frames:
            sleep(0.05)
        
        for face, face_data in self.data.items():
            face_data['dominant_emotion'] = mode(face_data['emotion'])
            face_data['last_coordinates'] = face_data['coordinates'][-1]
        
        self.stop_thread = True
        self.t.join()
        return self.data
    
