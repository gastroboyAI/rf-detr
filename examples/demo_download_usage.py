#!/usr/bin/env python3
"""
Demo script showing how to use download_from_roboflow method.
This script demonstrates the API without actually making network calls.
"""

import os
import sys

def demo_download_from_roboflow():
    """Demonstrate the download_from_roboflow method usage"""
    print("=" * 60)
    print("RF-DETR Download from Roboflow - Demo")
    print("=" * 60)
    
    print("\n1. Basic Usage:")
    print("```python")
    print("from rfdetr import RFDETRBase")
    print("")
    print("# Download your trained model from Roboflow")
    print("model = RFDETRBase.download_from_roboflow(")
    print("    workspace='your-workspace',")
    print("    project_id='colonoscopy-detection',")
    print("    version='1',")
    print("    api_key='your_api_key'")
    print(")")
    print("")
    print("# Use for inference")
    print("detections = model.predict('colonoscopy_frame.jpg')")
    print("```")
    
    print("\n2. Video Analysis Example:")
    print("```python")
    print("import cv2")
    print("from PIL import Image")
    print("")
    print("# Process colonoscopy video")
    print("cap = cv2.VideoCapture('colonoscopy_video.mp4')")
    print("while True:")
    print("    ret, frame = cap.read()")
    print("    if not ret: break")
    print("    ")
    print("    # Convert and analyze frame")
    print("    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)")
    print("    image = Image.fromarray(frame_rgb)")
    print("    detections = model.predict(image, threshold=0.5)")
    print("    ")
    print("    # Process detections...")
    print("```")
    
    print("\n3. Environment Setup:")
    print("```bash")
    print("# Set your API key")
    print("export ROBOFLOW_API_KEY='your_api_key_here'")
    print("")
    print("# Or pass it directly to the method")
    print("model = RFDETRBase.download_from_roboflow(..., api_key='your_key')")
    print("```")
    
    print("\n4. Command Line Tool:")
    print("```bash")
    print("# Use the provided example script")
    print("python examples/colonoscopy_analysis.py \\")
    print("    --workspace your-workspace \\")
    print("    --project-id colonoscopy-detection \\")
    print("    --version 1 \\")
    print("    --video colonoscopy_video.mp4 \\")
    print("    --output-dir ./results")
    print("```")
    
    print("\n" + "=" * 60)
    print("Key Benefits:")
    print("✓ Download weights from Roboflow-trained models")
    print("✓ Run inference locally on your PC/hardware")  
    print("✓ Process large videos without cloud API limits")
    print("✓ Work offline after initial download")
    print("✓ Integrate with existing computer vision pipelines")
    
    print("\nUse Cases:")
    print("• Medical imaging analysis (colonoscopy, radiology)")
    print("• Security camera footage analysis") 
    print("• Manufacturing quality control")
    print("• Agricultural monitoring")
    print("• Any custom object detection on local videos/images")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Train your RF-DETR model on Roboflow")
    print("2. Get your workspace, project_id, and API key")
    print("3. Use download_from_roboflow() to get the weights")
    print("4. Run inference on your local data!")
    print("=" * 60)

if __name__ == "__main__":
    demo_download_from_roboflow()