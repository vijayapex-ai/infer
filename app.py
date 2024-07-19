from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

import cv2
import supervision as sv
from flask import Flask, Response, render_template, jsonify, request
from threading import Thread, Lock

label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoundingBoxAnnotator()
tracker = sv.ByteTrack()

app = Flask(__name__)

global_frame = None
class_counts = {}
count_lock = Lock()

# Shared variables for user inputs
model_id = "face_recognition-bbdu5/2"
confidence = 0.5
iou_threshold = 0.5
pipeline = None
pipeline_thread = None

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    global global_frame, class_counts, tracker
    detections = sv.Detections.from_inference(predictions)
    labels = [p["class"] for p in predictions["predictions"]]
    
    # Ensure the number of labels matches the number of detections
    if len(labels) != len(detections):
        print(f"Mismatch: {len(labels)} labels and {len(detections)} detections")
        return

    tracked_detections = tracker.update_with_detections(detections)
    
    # Make sure each detection has a label
    if len(tracked_detections) == len(labels):
        image = label_annotator.annotate(
            scene=video_frame.image.copy(), detections=tracked_detections, labels=labels
        )
    else:
        image = video_frame.image.copy()

    image = box_annotator.annotate(image, detections=tracked_detections)
    ret, jpeg = cv2.imencode('.jpg', image)
    global_frame = jpeg.tobytes()
    
    with count_lock:
        # Reset class counts
        class_counts = {}
        for label in labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

def generate():
    global global_frame
    while True:
        if global_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/class_counts')
def class_counts_route():
    global class_counts
    with count_lock:
        counts = class_counts.copy()
    return jsonify(counts=counts)

@app.route('/update_parameters', methods=['POST'])
def update_parameters():
    global confidence, iou_threshold, model_id
    data = request.json
    confidence = data.get('confidence', confidence)
    iou_threshold = data.get('iou_threshold', iou_threshold)
    model_id = data.get('model_id', model_id)
    # Restart the pipeline with new parameters
    restart_pipeline()
    return jsonify(success=True)

def restart_pipeline():
    global pipeline_thread, pipeline
    if pipeline is not None:
        pipeline.terminate()  # Ensure this is a safe operation, or replace with appropriate cleanup if needed
        pipeline_thread.join()
    
    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=2,
        confidence=confidence,
        iou_threshold=iou_threshold,
        active_learning_enabled=True,
        on_prediction=my_custom_sink,
        video_source_properties={
            "frame_width": 1280.0,
            "frame_height": 720.0,
            "fps": 30.0,
        }
    )
    pipeline_thread = Thread(target=pipeline.start)
    pipeline_thread.start()

@app.route('/')
def index():
    return render_template('index.html', confidence=confidence, iou_threshold=iou_threshold, model_id=model_id)

if __name__ == '__main__':
    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=2,
        confidence=confidence,
        iou_threshold=iou_threshold,
        active_learning_enabled=True,
        on_prediction=my_custom_sink,
        video_source_properties={
            "frame_width": 1280.0,
            "frame_height": 720.0,
            "fps": 30.0,
        }
    )

    pipeline_thread = Thread(target=pipeline.start)
    pipeline_thread.start()

    app.run(port=5000)
