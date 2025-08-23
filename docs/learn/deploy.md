# Deploy and Download RF-DETR Models with Roboflow

You can both deploy a fine-tuned RF-DETR model to Roboflow and download trained models from Roboflow for local use.

## Deploy to Roboflow

Deploying to Roboflow allows you to create multi-step computer vision applications that run both in the cloud and your own hardware.

To deploy your model to Roboflow, run:

```python
from rfdetr import RFDETRNano

x = RFDETRNano(pretrain_weights="<path/to/pretrain/weights/dir>")
x.deploy_to_roboflow(
  workspace="<your-workspace>",
  project_id="<your-project-id>",
  version=1,
  api_key="<YOUR_API_KEY>"
)
```

Above, set your Roboflow Workspace ID, the ID of the project to which you want to upload your model, and your Roboflow API key.

- [Learn how to find your Workspace and Project ID.](https://docs.roboflow.com/developer/authentication/workspace-and-project-ids)
- [Learn how to find your API key.](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key)

## Download from Roboflow

If you trained an RF-DETR model on Roboflow and want to download the weights to use locally on your own hardware, you can use the `download_from_roboflow` method:

```python
from rfdetr import RFDETRBase

# Download your trained model from Roboflow
model = RFDETRBase.download_from_roboflow(
    workspace="your-workspace",
    project_id="your-project-id", 
    version="1",
    api_key="your_api_key"  # or set ROBOFLOW_API_KEY environment variable
)

# Now use the model for local inference
detections = model.predict("path/to/your/image.jpg", threshold=0.5)
```

### Use Case: Colonoscopy Video Analysis

If you trained an RF-DETR model on Roboflow for colonoscopy detection and want to run inference on saved colonoscopy videos on your PC:

```python
import cv2
import supervision as sv
from rfdetr import RFDETRBase
from PIL import Image

# Download your trained colonoscopy model
model = RFDETRBase.download_from_roboflow(
    workspace="your-workspace",
    project_id="colonoscopy-detection",
    version="1"
)

# Process a colonoscopy video
cap = cv2.VideoCapture("path/to/colonoscopy_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # Run inference
    detections = model.predict(image, threshold=0.5)
    
    # Annotate frame
    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    
    # Display or save results
    cv2.imshow('Colonoscopy Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Parameters

The `download_from_roboflow` method accepts the following parameters:

- `workspace` (str): Your Roboflow workspace name
- `project_id` (str): The ID of the project containing your trained model
- `version` (str): The version number of your trained model
- `api_key` (str, optional): Your Roboflow API key (can also be set via `ROBOFLOW_API_KEY` environment variable)
- `local_weights_path` (str, optional): Local path to save the weights (defaults to `./roboflow_weights_{project_id}_v{version}.pt`)

## Using Downloaded Models with Roboflow Inference

You can then run your model with Roboflow Inference:

```python
import os
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests

url = "https://media.roboflow.com/dog.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

model = get_model("rfdetr-base")  # replace with your Roboflow model ID

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
```

Above, replace `rfdetr-base` with the your Roboflow model ID. You can find this ID from the "Models" list in your Roboflow dashboard:

![](https://media.roboflow.com/rfdetr/models-list.png)

When you first run this model, your model weights will be cached for local use with Inference.

You will then see the results from your fine-tuned model.