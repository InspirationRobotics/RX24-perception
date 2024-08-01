from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List
import cv2
import numpy as np

# Load a YOLOv8n PyTorch model
#model = YOLO("yolov8n.pt")

# Export the model
#model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
trt_model = YOLO("yolov8n.engine", task="detect")

# Run inference
results : List[Results]= trt_model("bus.jpg")

results[0].save("output.jpg")
# Display results on the frame and save it
# Load the input image

# img = cv2.imread("bus.jpg")

# # Loop over the detections and draw bounding boxes and labels
# for result in results:
#     for bbox in result.boxes.xyxy:
#         x1, y1, x2, y2 = map(int, bbox)
#         label = result.names[int(result.cls)]
#         conf = result.conf
#         color = (0, 255, 0)  # Green color for bounding box
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# cv2.imshow("Img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Note: Run this: export LD_PRELOAD=/lib/aarch64-linux-gnu/libstdc++.so.6:$LD_PRELOAD