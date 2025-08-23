# RF-DETR Examples

This directory contains example scripts demonstrating how to use RF-DETR with Roboflow.

## Colonoscopy Video Analysis

The `colonoscopy_analysis.py` script shows how to:

1. Download a trained RF-DETR model from Roboflow
2. Analyze colonoscopy videos frame by frame
3. Generate annotated videos with detections
4. Save detailed analysis results

### Usage

```bash
python examples/colonoscopy_analysis.py \
    --workspace your-workspace \
    --project-id colonoscopy-detection \
    --version 1 \
    --api-key your_api_key \
    --video path/to/colonoscopy_video.mp4 \
    --output-dir ./results \
    --confidence 0.5
```

### Prerequisites

- An RF-DETR model trained on Roboflow for colonoscopy detection
- Roboflow API key
- OpenCV installed (`pip install opencv-python`)
- Input video file

### Example Output

The script will generate:
- `video_name_annotated.avi` - Annotated video with bounding boxes
- `video_name_analysis.json` - Detailed frame-by-frame analysis results

### Setting Up Your Colonoscopy Model

1. **Create a Roboflow project** for colonoscopy detection
2. **Upload and annotate** your colonoscopy images/videos
3. **Train an RF-DETR model** using Roboflow's cloud training
4. **Get your model details**:
   - Workspace name (from URL: `app.roboflow.com/{workspace}/...`)
   - Project ID (from URL: `app.roboflow.com/{workspace}/{project-id}/...`)
   - Version number (usually starts at "1")
   - API key (from account settings)

### Troubleshooting

**"Model not found"**:
- Verify workspace, project ID, and version are correct
- Ensure your API key has access to the project

**"Download failed"**:
- Check internet connection
- Verify the model was trained with RF-DETR (not YOLOv8 or other architectures)

**"Video not opening"**:
- Ensure the video file exists and is in a supported format
- Try converting to MP4 or AVI format

For more detailed information, see the [Download from Roboflow guide](../docs/learn/download-from-roboflow.md).