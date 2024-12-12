import cv2
import numpy as np

# Load YOLO model
def load_yolo_model():
    """
    Load YOLOv3 model with pre-trained weights and config files.
    """
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    
    return net, output_layers

# Function to process the image and detect objects
def detect_objects(img, net, output_layers):
    """
    Detect objects in the image and return the detected objects' info.
    """
    # Get the height, width and channels of the image
    height, width, channels = img.shape
    
    # Convert image to blob format (input for the model)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Initialize lists for detected bounding boxes, confidences, and class IDs
    class_ids = []
    confidences = []
    boxes = []
    
    # Process the output of the YOLO model
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Only proceed if the confidence is high enough (thresholding)
            if confidence > 0.5:
                # Get the coordinates of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle top-left corner
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression to avoid multiple boxes for the same object
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    return boxes, confidences, class_ids, indexes

# Function to draw the bounding boxes around detected objects
def draw_labels(img, boxes, confidences, class_ids, indexes, classes):
    """
    Draw bounding boxes and labels on the image for detected objects.
    """
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0)  # Green color for bounding box
            
            # Draw rectangle and text on image
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img

# Main function for Object Detection
def object_detection(img_path):
    """
    Detect objects in the provided image and display the result.
    """
    # Load the YOLO model and classes
    net, output_layers = load_yolo_model()
    
    # Load class labels
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Read the input image
    img = cv2.imread(img_path)
    
    # Detect objects in the image
    boxes, confidences, class_ids, indexes = detect_objects(img, net, output_layers)
    
    # Draw bounding boxes and labels on the image
    img = draw_labels(img, boxes, confidences, class_ids, indexes, classes)
    
    # Display the image with detected objects
    cv2.imshow("Object Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Real-time object detection using webcam
def real_time_object_detection():
    """
    Perform real-time object detection using webcam stream.
    """
    cap = cv2.VideoCapture(0)  # Use webcam as input
    
    net, output_layers = load_yolo_model()
    
    # Load class labels
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        # Detect objects in the frame
        boxes, confidences, class_ids, indexes = detect_objects(frame, net, output_layers)
        
        # Draw bounding boxes and labels on the frame
        frame = draw_labels(frame, boxes, confidences, class_ids, indexes, classes)
        
        # Display the frame with detected objects
        cv2.imshow("Real-time Object Detection", frame)
        
        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose whether to perform real-time detection or static image detection
    mode = input("Choose detection mode:\n1. Static Image Detection\n2. Real-time Webcam Detection\nEnter choice (1 or 2): ")
    
    if mode == '1':
        image_path = input("Enter the image file path: ")
        object_detection(image_path)
    elif mode == '2':
        real_time_object_detection()
    else:
        print("Invalid input! Please choose 1 or 2.")
