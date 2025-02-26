import cv2
from ultralytics import YOLO

# Disease information dictionary
DISEASE_INFO = {
    "Black Fungus Pod": {"medicine": "Mancozeb or Carbendazim"},
    "Early and Late leaf spot": {"medicine": "Chlorothalonil or Propiconazole"},
    "Fungus leaf": {"medicine": "Copper Oxychloride"},
    "Rust-Leaf": {"medicine": "Hexaconazole or Sulphur"},
    "black fungus-groundnut": {"medicine": "Carbendazim or Thiophanate-methyl"},
    "brown Fungus Pod": {"medicine": "Mancozeb or Chlorothalonil"},
    "bakteri_daun_bergaris": {"medicine": "Streptomycin or Copper Hydroxide"},
    "bercak_coklat": {"medicine": "Propiconazole or Mancozeb"},
    "bercak_coklat_sempit": {"medicine": "Carbendazim or Propiconazole"},
    "hawar_daun_bakteri": {"medicine": "Copper Oxychloride or Streptomycin"},
    "tungro": {"medicine": "Buprofezin or Imidacloprid"}
}

def process_detection(results, frame):
    """Process YOLO detection results and annotate the frame with detection and recommendation."""
    current_disease = None
    current_medicine = None
    annotated_frame = frame.copy()
    height, width = annotated_frame.shape[:2]
    
    try:
        # Iterate through all detections from the first result (assuming a single image/frame)
        for result in results[0].boxes:
            # Extract class id and confidence
            class_id = int(result.cls.item())
            confidence = float(result.conf.item())
            
            # Get disease name from the model's names dictionary
            disease = results[0].names[class_id]
            
            # Process only if disease is recognized and confidence is high enough (>70%)
            if disease in DISEASE_INFO and confidence > 0.7:
                current_disease = disease
                current_medicine = DISEASE_INFO[disease]["medicine"]
                
                # Draw detection bounding box and label on the frame
                box = result.xyxy[0].cpu().numpy()  # Get box coordinates
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{disease} {confidence:.2f}"
                cv2.putText(annotated_frame, label, 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Overlay the detected disease on the top-left of the frame
        if current_disease:
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (width // 2, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.putText(annotated_frame, f"Disease: {current_disease}", 
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Overlay the recommended medicine on the bottom-right of the frame
        if current_medicine:
            text = f"Medicine: {current_medicine}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (width - text_size[0] - 30, height - 40),
                          (width - 10, height - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.putText(annotated_frame, text,
                        (width - text_size[0] - 20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
        return annotated_frame
    
    except Exception as e:
        print(f"Error processing detection: {e}")
        return frame

def main():
    # Model selection
    model_choice = input("Which model do you want to use? (1 for paddy, 2 for groundnuts): ")
    model_path = "PaddyDet.pt" if model_choice == "1" else "GroundNutDet.pt"
    
    try:
        # Load the YOLO model
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
        
        # Open the default camera (webcam)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open video device")
        
        print("Starting camera stream. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Perform inference on the frame
            results = model(frame, device='cpu')
            
            # Process detections and annotate frame
            annotated_frame = process_detection(results, frame)
            
            # Display the frame with annotations
            cv2.imshow('Disease Detection and Recommendation', annotated_frame)
            
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    main()

