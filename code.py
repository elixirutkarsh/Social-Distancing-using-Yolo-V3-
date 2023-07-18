import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load image
image = cv2.imread("image.jpg")

# Resize image
image = cv2.resize(image, None, fx=0.4, fy=0.4)

# Convert image to blob
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Pass blob through the network
net.setInput(blob)
outs = net.forward(output_layers)

# Get bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []
height, width, channels = image.shape
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 0:  # Class ID 0 represents "person"
            # Scale the bounding box coordinates back to the original image size
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Save bounding box coordinates, confidences, and class IDs
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression to remove overlapping bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and calculate distances between people
dist_thresh = 100  # Distance threshold in pixels
distances = []
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate centroid of the bounding box
        cx = x + w // 2
        cy = y + h // 2

        # Calculate distance between centroids of people
        for j in range(i + 1, len(boxes)):
            if j in indexes:
                x2, y2, w2, h2 = boxes[j]
                cx2 = x2 + w2 // 2
                cy2 = y2 + h2 // 2

                # Calculate Euclidean distance between centroids
                distance = np.sqrt((cx - cx2) ** 2 + (cy - cy2) ** 2)
                if distance < dist_thresh:
                    # Draw line between people if distance is less than threshold
                    cv2.line(image, (cx, cy), (cx2, cy2), (0, 0, 255), 2)
                    distances.append(distance)

# Display image with bounding boxes and social distancing violations
cv2.imshow("Social Distancing", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
