# Download and Use RF-DETR Models from Roboflow

This guide shows you how to download model weights from a Roboflow-trained RF-DETR model and use them locally for inference.

## Overview

If you trained an RF-DETR model using Roboflow's cloud training, you can download the trained weights and run inference locally on your own hardware. This is particularly useful for:

- Running inference on large datasets or videos
- Deploying on edge devices without internet connectivity
- Fine-tuning or further customization of your model
- Batch processing of data

## Quick Start

```python
from rfdetr import RFDETRBase

# Download your trained model from Roboflow
model = RFDETRBase.download_from_roboflow(
    workspace="your-workspace",
    project_id="your-project-id",
    version="1",
    api_key="your_roboflow_api_key"
)

# Use the model for inference
detections = model.predict("path/to/image.jpg", threshold=0.5)
```

## Prerequisites

1. **A trained RF-DETR model on Roboflow**: You need to have successfully trained an RF-DETR model using Roboflow's cloud training.
2. **Roboflow API key**: Get your API key from your [Roboflow account settings](https://app.roboflow.com/settings/api).
3. **Workspace and Project IDs**: Know your workspace name and project ID where the model was trained.

## Finding Your Model Information

### Workspace Name
Your workspace name is visible in the URL when you're in Roboflow: `https://app.roboflow.com/{workspace-name}/...`

### Project ID
The project ID is also in the URL: `https://app.roboflow.com/{workspace}/{project-id}/...`

### Version Number
Each trained model has a version number, typically starting from "1".

## Basic Usage

### Download and Load Model

```python
from rfdetr import RFDETRBase
import os

# Set your API key (alternatively, set ROBOFLOW_API_KEY environment variable)
os.environ["ROBOFLOW_API_KEY"] = "your_api_key_here"

# Download model weights
model = RFDETRBase.download_from_roboflow(
    workspace="medical-imaging",
    project_id="colonoscopy-polyp-detection",
    version="2"
)

print(f"Model loaded successfully!")
print(f"Model supports {len(model.class_names)} classes: {list(model.class_names.values())}")
```

### Run Inference on Single Image

```python
from PIL import Image
import supervision as sv

# Load and predict on an image
image = Image.open("colonoscopy_frame.jpg")
detections = model.predict(image, threshold=0.5)

# Visualize results
annotated_image = sv.BoxAnnotator().annotate(image.copy(), detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)

# Save or display
annotated_image.save("results.jpg")
```

## Advanced Use Cases

### Video Processing

Process a saved colonoscopy video frame by frame:

```python
import cv2
import supervision as sv
from PIL import Image

# Load your model
model = RFDETRBase.download_from_roboflow(
    workspace="medical-imaging",
    project_id="colonoscopy-detection", 
    version="1"
)

# Open video file
cap = cv2.VideoCapture("colonoscopy_procedure.mp4")

# Prepare video writer for saving results
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('annotated_video.avi', fourcc, 30.0, (640, 480))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB for the model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # Run inference
    detections = model.predict(image, threshold=0.3)
    
    # Convert back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Annotate detections
    annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
    
    # Add frame info
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save frame
    out.write(annotated_frame)
    
    # Optional: display real-time (remove for faster processing)
    cv2.imshow('Colonoscopy Analysis', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video analysis complete! Processed {frame_count} frames.")
```

### Batch Processing

Process multiple images in a directory:

```python
import os
import json
from pathlib import Path
import supervision as sv

# Load model
model = RFDETRBase.download_from_roboflow(
    workspace="your-workspace",
    project_id="your-project",
    version="1"
)

# Process all images in a directory
image_dir = Path("colonoscopy_images")
results_dir = Path("detection_results")
results_dir.mkdir(exist_ok=True)

results_summary = []

for image_path in image_dir.glob("*.jpg"):
    # Run inference
    detections = model.predict(str(image_path), threshold=0.5)
    
    # Save annotated image
    image = Image.open(image_path)
    annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
    annotated.save(results_dir / f"annotated_{image_path.name}")
    
    # Store results
    results_summary.append({
        "image": image_path.name,
        "detections": len(detections),
        "confidence_scores": detections.confidence.tolist() if len(detections) > 0 else []
    })
    
    print(f"Processed {image_path.name}: {len(detections)} detections")

# Save summary
with open(results_dir / "detection_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"Batch processing complete! Results saved to {results_dir}")
```

### Model Optimization for Inference

For better performance, especially when processing many images or videos:

```python
# Load and optimize model for inference
model = RFDETRBase.download_from_roboflow(
    workspace="your-workspace",
    project_id="your-project",
    version="1"
)

# Optimize for faster inference (optional but recommended)
model.optimize_for_inference()

# Now run inference - should be faster
detections = model.predict("image.jpg", threshold=0.5)
```

## Customizing Download Location

```python
# Specify where to save the weights
model = RFDETRBase.download_from_roboflow(
    workspace="your-workspace",
    project_id="colonoscopy-detection",
    version="1",
    local_weights_path="./my_models/colonoscopy_model.pt"
)

# Or load from previously downloaded weights
model = RFDETRBase(pretrain_weights="./my_models/colonoscopy_model.pt")
```

## Error Handling

```python
try:
    model = RFDETRBase.download_from_roboflow(
        workspace="your-workspace",
        project_id="your-project",
        version="1"
    )
except FileNotFoundError as e:
    print(f"Download failed: {e}")
    print("Please check your workspace, project ID, version, and API key")
except RuntimeError as e:
    print(f"Model loading failed: {e}")
    print("The downloaded weights may be incompatible")
```

## Troubleshooting

### Common Issues

1. **"API key not found"**: Set your API key using `api_key` parameter or `ROBOFLOW_API_KEY` environment variable.

2. **"Model not found"**: Verify your workspace name, project ID, and version number are correct.

3. **"Download failed"**: Check your internet connection and that you have access to the specified project.

4. **"Model loading failed"**: Ensure the model was trained with RF-DETR (not YOLOv8 or another architecture).

### Manual Download Alternative

If the automatic download fails, you can manually download weights:

1. Go to your Roboflow project dashboard
2. Navigate to the "Models" tab
3. Find your RF-DETR model and download the weights file
4. Load the model manually:

```python
model = RFDETRBase(pretrain_weights="path/to/manually_downloaded_weights.pt")
```

## Next Steps

- [Learn more about RF-DETR training](train.md)
- [Explore deployment options](deploy.md)
- [Check out benchmarks and performance](benchmarks.md)