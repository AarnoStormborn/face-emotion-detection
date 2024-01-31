from time import time
import datetime
import numpy as np
from ultralytics import YOLO
from rmn import RMN
from flask import Flask, request
from concurrent.futures import ProcessPoolExecutor
from video import VideoInference

app = Flask(__name__)

data = dict()
face_detection_model = YOLO('src/models/yolov8n-face.pt')
emotion_detection_model = RMN()

@app.route("/ping", methods=["GET"])
def sample_endpoint():
    
    """
    Sample Endpoint for testing
    Method: GET
    """
    
    "Testing Endpoint to ensure status"
    return {"Message":"Hello API"}

@app.route('/invocations', methods=["POST"])
def predict():
    
    
    """
    Prediciton function. 
    Receives a batch of frames and runs inference in a separate process.
    method: POST
    """
    
    
    imgSize = 640
    try:
        frames = request.data
        framesArray = np.frombuffer(frames, dtype=np.uint8)
        
        # Resizing frames array to have (num_frames, imgSize, imgSize, num_channels)
        framesShape = (len(framesArray) // (3 * imgSize * imgSize), imgSize, imgSize, 3)
        framesArrayReshaped = framesArray.reshape(framesShape)
        
        # Creating a chunk for the entire batch, not splitting the array.
        # Needed to execute parallel processing
        
        numChunks = 1
        framesChunks = np.array_split(framesArrayReshaped, numChunks)
        start = time()
        
        # Starting another process in the background to handle inference. Allows multithreading.
        with ProcessPoolExecutor(max_workers=numChunks) as executor:
            results = list(executor.map(process_frame_chunks, framesChunks))
        
        dataResults = {}
        for result in results:
            dataResults.update(result)
        
        return dataResults

    except Exception as e:
        return {"Error": e}

def process_frame_chunks(framesChunks):
    
    """
    Called in a separate process. Instantiates Pipeline and runs Inference
    args:
        frameChunks : Batch of Frames
    returns:
        data : inference results
    """
    
    imgSize = 640
    vid = VideoInference(frames=framesChunks,
                         face_detection_model=face_detection_model,
                         emotion_detection_model=emotion_detection_model,
                         img_size=imgSize)
    data = vid.inference()
    
    return data

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)