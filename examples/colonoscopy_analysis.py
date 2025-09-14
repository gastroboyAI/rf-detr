#!/usr/bin/env python3
"""
Example: Download RF-DETR Model from Roboflow for Colonoscopy Video Analysis

This script demonstrates how to:
1. Download a trained RF-DETR model from Roboflow
2. Use it to analyze colonoscopy videos on your local PC
3. Save annotated results

Prerequisites:
- RF-DETR model trained on Roboflow
- Roboflow API key
- Input video file(s)
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase

def download_model(workspace, project_id, version, api_key=None):
    """Download RF-DETR model from Roboflow"""
    print(f"Downloading RF-DETR model from Roboflow...")
    print(f"Workspace: {workspace}")
    print(f"Project: {project_id}")
    print(f"Version: {version}")
    
    try:
        model = RFDETRBase.download_from_roboflow(
            workspace=workspace,
            project_id=project_id,
            version=version,
            api_key=api_key
        )
        
        print(f"✓ Model downloaded successfully!")
        print(f"✓ Model classes: {list(model.class_names.values())}")
        return model
        
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is correct")
        print("2. Verify workspace and project_id are correct")
        print("3. Ensure the model was trained with RF-DETR")
        print("4. Check you have access to the project")
        return None

def analyze_video(model, video_path, output_path, confidence_threshold=0.5):
    """Analyze a colonoscopy video with the RF-DETR model"""
    print(f"\nAnalyzing video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Analysis results
    results = []
    frame_count = 0
    detection_count = 0
    
    print("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Run inference
        detections = model.predict(image, threshold=confidence_threshold)
        
        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Annotate detections
        if len(detections) > 0:
            frame_bgr = sv.BoxAnnotator().annotate(frame_bgr, detections)
            frame_bgr = sv.LabelAnnotator().annotate(frame_bgr, detections)
            detection_count += len(detections)
        
        # Add frame info
        cv2.putText(frame_bgr, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"Detections: {len(detections)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save frame
        out.write(frame_bgr)
        
        # Store frame results
        results.append({
            "frame": frame_count,
            "timestamp": frame_count / fps,
            "detections": len(detections),
            "confidence_scores": detections.confidence.tolist() if len(detections) > 0 else []
        })
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"✓ Video analysis complete!")
    print(f"✓ Total frames processed: {frame_count}")
    print(f"✓ Total detections: {detection_count}")
    print(f"✓ Average detections per frame: {detection_count/frame_count:.2f}")
    print(f"✓ Annotated video saved: {output_path}")
    
    return results

def save_analysis_results(results, output_dir, video_name):
    """Save detailed analysis results"""
    results_file = output_dir / f"{video_name}_analysis.json"
    
    # Calculate summary statistics
    total_frames = len(results)
    frames_with_detections = sum(1 for r in results if r["detections"] > 0)
    total_detections = sum(r["detections"] for r in results)
    
    summary = {
        "video_name": video_name,
        "total_frames": total_frames,
        "frames_with_detections": frames_with_detections,
        "detection_rate": frames_with_detections / total_frames,
        "total_detections": total_detections,
        "average_detections_per_frame": total_detections / total_frames,
        "frame_results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Analysis results saved: {results_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Analyze colonoscopy videos with RF-DETR model from Roboflow")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace name")
    parser.add_argument("--project-id", required=True, help="Roboflow project ID")
    parser.add_argument("--version", default="1", help="Model version (default: 1)")
    parser.add_argument("--api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--video", required=True, help="Path to colonoscopy video file")
    parser.add_argument("--output-dir", default="./analysis_results", help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Download model
    model = download_model(args.workspace, args.project_id, args.version, args.api_key)
    if not model:
        return 1
    
    # Analyze video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"✗ Video file not found: {video_path}")
        return 1
    
    video_name = video_path.stem
    output_video = output_dir / f"{video_name}_annotated.avi"
    
    results = analyze_video(model, video_path, output_video, args.confidence)
    if not results:
        return 1
    
    # Save results
    summary = save_analysis_results(results, output_dir, video_name)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Summary:")
    print(f"   - Detection rate: {summary['detection_rate']:.1%}")
    print(f"   - Frames with detections: {summary['frames_with_detections']}/{summary['total_frames']}")
    print(f"   - Total detections: {summary['total_detections']}")
    print(f"📁 Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())