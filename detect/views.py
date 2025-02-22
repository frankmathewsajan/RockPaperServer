import base64
import tempfile
import os
import cv2
import numpy as np
import json
import logging
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the YOLO model (do this at module level for efficiency)
try:
    model = YOLO("best_r_p_game_5.pt")
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None


@csrf_exempt
def process_image(request):
    logger.info("Received image processing request", request)
    if request.method != "POST":
        logger.warning("Invalid request method: %s", request.method)
        return JsonResponse({"error": "Only POST requests are allowed"}, status=400)

    # Parse JSON payload
    try:
        data = json.loads(request.body)
        logger.info("JSON payload parsed successfully")
    except json.JSONDecodeError as e:
        logger.error("JSON decode error: %s", str(e))
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "image" not in data:
        logger.error("No image provided in JSON payload")
        return JsonResponse({"error": "No image provided in JSON"}, status=400)

    base64_image = data["image"]
    # Remove data URI prefix if present (e.g., "data:image/png;base64,")
    if base64_image.startswith("data:"):
        base64_image = base64_image.split(",")[1]

    try:
        image_bytes = base64.b64decode(base64_image)
        logger.info("Base64 image decoded successfully")
    except Exception as e:
        logger.error("Base64 decode error: %s", str(e))
        return JsonResponse({"error": "Invalid base64 encoding"}, status=400)

    try:
        image = Image.open(BytesIO(image_bytes))
        logger.info("Image opened successfully with PIL")
    except Exception as e:
        logger.error("Error opening image: %s", str(e))
        return JsonResponse({"error": "Error processing image"}, status=400)

    # Convert image to OpenCV format
    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logger.info("Image converted to OpenCV format")
    except Exception as e:
        logger.error("Error converting image to OpenCV format: %s", str(e))
        return JsonResponse({"error": "Error converting image"}, status=400)

    # Run YOLO detection
    try:
        results = model(image_cv)
        logger.info("YOLO detection complete. Number of results: %d", len(results))
    except Exception as e:
        logger.error("Error running YOLO detection: %s", str(e))
        return JsonResponse({"error": "Error during detection"}, status=500)

    # Collect detections info and draw detections on image
    detections_info = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            name = model.names[cls] if hasattr(model, 'names') and cls in model.names else str(cls)
            label = f"{name} {conf:.2f}"
            logger.info("Detection: %s at (%d, %d, %d, %d)", label, x1, y1, x2, y2)

            # Append textual detection information
            detections_info.append({
                "label": name,
                "confidence": conf,
                "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

            # Draw rectangle and label on image
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert image back to PIL format
    try:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(image_rgb)
        logger.info("Processed image created successfully")
    except Exception as e:
        logger.error("Error converting processed image to PIL format: %s", str(e))
        return JsonResponse({"error": "Error processing image"}, status=500)

    # Save processed image to a buffer and encode as base64
    buffer = BytesIO()
    try:
        processed_image.save(buffer, format="JPEG")
        logger.info("Processed image saved to buffer")
    except Exception as e:
        logger.error("Error saving processed image to buffer: %s", str(e))
        return JsonResponse({"error": "Error saving image"}, status=500)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    image_data = "data:image/jpeg;base64," + img_base64

    # If no detections were found, add a message
    if not detections_info:
        message = "No detections found"
    else:
        message = f"{len(detections_info)} detections found"

    # Return a JSON response containing both the processed image and textual detection info
    response_payload = {
        "image": image_data,
        "detections": detections_info,
        "message": message
    }
    logger.info("Returning processed image and detection info", response_payload)
    print(response_payload)
    return JsonResponse(response_payload)


def process_video(request):
    """
    Endpoint to receive video, process it with YOLO model, and return the processed video
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are accepted'}, status=405)

    try:
        # Parse request body
        data = json.loads(request.body)
        base64_video = data.get('video')
        video_format = data.get('format', 'mp4')

        if not base64_video:
            return JsonResponse({'error': 'No video data provided'}, status=400)

        # Create temporary files for input and output videos
        with tempfile.NamedTemporaryFile(suffix=f'.{video_format}', delete=False) as input_video_file:
            input_video_path = input_video_file.name
            # Decode and save base64 video
            video_data = base64.b64decode(base64_video)
            input_video_file.write(video_data)

        output_video_path = input_video_path + f'_processed.{video_format}'

        # Process the video with YOLO
        process_with_yolo(input_video_path, output_video_path)

        # Read the processed video and encode to base64
        with open(output_video_path, 'rb') as f:
            processed_video_data = f.read()
        processed_video_base64 = base64.b64encode(processed_video_data).decode('utf-8')

        # Clean up temporary files
        try:
            os.unlink(input_video_path)
            os.unlink(output_video_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {str(e)}")

        return JsonResponse({
            'processed_video': processed_video_base64,
            'message': 'Video processed successfully'
        })

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def process_with_yolo(input_path, output_path):
    """
    Process video file with YOLO detection and save the results
    """
    if model is None:
        raise Exception("YOLO model not loaded")

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file {input_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Processing video: {width}x{height} at {fps} FPS, {total_frames} frames")

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for h264
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 10 == 0:  # Log progress every 10 frames
            logger.info(f"Processing frame {frame_count}/{total_frames}")

        # Run YOLO detection
        results = model(frame, stream=True)

        # Draw detections on frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if hasattr(model, 'names') and cls in model.names:
                    label = f"{model.names[cls]} {conf:.2f}"
                else:
                    label = f"Class {cls} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    logger.info(f"Video processing complete: {frame_count} frames processed")


def index(request):
    return render(request, 'index.html')
