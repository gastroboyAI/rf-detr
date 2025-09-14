# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import json
import os
import tempfile
from collections import defaultdict
from logging import getLogger
from typing import Union, List
from copy import deepcopy

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
from PIL import Image

try:
    torch.set_float32_matmul_precision('high')
except:
    pass

from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    TrainConfig,
    ModelConfig
)
from rfdetr.main import Model, download_pretrain_weights
from rfdetr.util.metrics import MetricsPlotSink, MetricsTensorBoardSink, MetricsWandBSink
from rfdetr.util.coco_classes import COCO_CLASSES

logger = getLogger(__name__)
class RFDETR:
    """
    The base RF-DETR class implements the core methods for training RF-DETR models,
    running inference on the models, optimising models, and uploading trained
    models for deployment.
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = None

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    def maybe_download_pretrain_weights(self):
        """
        Download pre-trained weights if they are not already downloaded.
        """
        download_pretrain_weights(self.model_config.pretrain_weights)

    def get_model_config(self, **kwargs):
        """
        Retrieve the configuration parameters used by the model.
        """
        return ModelConfig(**kwargs)

    def train(self, **kwargs):
        """
        Train an RF-DETR model.
        """
        config = self.get_train_config(**kwargs)
        self.train_from_config(config, **kwargs)
    
    def optimize_for_inference(self, compile=True, batch_size=1, dtype=torch.float32):
        self.remove_optimized_model()

        self.model.inference_model = deepcopy(self.model.model)
        self.model.inference_model.eval()
        self.model.inference_model.export()

        self._optimized_resolution = self.model.resolution
        self._is_optimized_for_inference = True

        self.model.inference_model = self.model.inference_model.to(dtype=dtype)
        self._optimized_dtype = dtype

        if compile:
            self.model.inference_model = torch.jit.trace(
                self.model.inference_model,
                torch.randn(
                    batch_size, 3, self.model.resolution, self.model.resolution, 
                    device=self.model.device,
                    dtype=dtype
                )
            )
            self._optimized_has_been_compiled = True
            self._optimized_batch_size = batch_size
    
    def remove_optimized_model(self):
        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_half = False
    
    def export(self, **kwargs):
        """
        Export your model to an ONNX file.

        See [the ONNX export documentation](https://rfdetr.roboflow.com/learn/train/#onnx-export) for more information.
        """
        self.model.export(**kwargs)

    def train_from_config(self, config: TrainConfig, **kwargs):
        with open(
            os.path.join(config.dataset_dir, "train", "_annotations.coco.json"), "r"
        ) as f:
            anns = json.load(f)
            num_classes = len(anns["categories"])
            class_names = [c["name"] for c in anns["categories"] if c["supercategory"] != "none"]
            self.model.class_names = class_names

        if self.model_config.num_classes != num_classes:
            logger.warning(
                f"num_classes mismatch: model has {self.model_config.num_classes} classes, but your dataset has {num_classes} classes\n"
                f"reinitializing your detection head with {num_classes} classes."
            )
            self.model.reinitialize_detection_head(num_classes)
        
        train_config = config.dict()
        model_config = self.model_config.dict()
        model_config.pop("num_classes")
        if "class_names" in model_config:
            model_config.pop("class_names")
        
        if "class_names" in train_config and train_config["class_names"] is None:
            train_config["class_names"] = class_names

        for k, v in train_config.items():
            if k in model_config:
                model_config.pop(k)
            if k in kwargs:
                kwargs.pop(k)
        
        all_kwargs = {**model_config, **train_config, **kwargs, "num_classes": num_classes}

        metrics_plot_sink = MetricsPlotSink(output_dir=config.output_dir)
        self.callbacks["on_fit_epoch_end"].append(metrics_plot_sink.update)
        self.callbacks["on_train_end"].append(metrics_plot_sink.save)

        if config.tensorboard:
            metrics_tensor_board_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
            self.callbacks["on_fit_epoch_end"].append(metrics_tensor_board_sink.update)
            self.callbacks["on_train_end"].append(metrics_tensor_board_sink.close)

        if config.wandb:
            metrics_wandb_sink = MetricsWandBSink(
                output_dir=config.output_dir,
                project=config.project,
                run=config.run,
                config=config.model_dump()
            )
            self.callbacks["on_fit_epoch_end"].append(metrics_wandb_sink.update)
            self.callbacks["on_train_end"].append(metrics_wandb_sink.close)

        if config.early_stopping:
            from rfdetr.util.early_stopping import EarlyStoppingCallback
            early_stopping_callback = EarlyStoppingCallback(
                model=self.model,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                use_ema=config.early_stopping_use_ema
            )
            self.callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)

        self.model.train(
            **all_kwargs,
            callbacks=self.callbacks,
        )

    def get_train_config(self, **kwargs):
        """
        Retrieve the configuration parameters that will be used for training.
        """
        return TrainConfig(**kwargs)

    def get_model(self, config: ModelConfig):
        """
        Retrieve a model instance based on the provided configuration.
        """
        return Model(**config.dict())
    
    # Get class_names from the model
    @property
    def class_names(self):
        """
        Retrieve the class names supported by the loaded model.

        Returns:
            dict: A dictionary mapping class IDs to class names. The keys are integers starting from
        """
        if hasattr(self.model, 'class_names') and self.model.class_names:
            return {i+1: name for i, name in enumerate(self.model.class_names)}
            
        return COCO_CLASSES

    def predict(
        self,
        images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
        threshold: float = 0.5,
        **kwargs,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """Performs object detection on the input images and returns bounding box
        predictions.

        This method accepts a single image or a list of images in various formats
        (file path, PIL Image, NumPy array, or torch.Tensor). The images should be in
        RGB channel order. If a torch.Tensor is provided, it must already be normalized
        to values in the [0, 1] range and have the shape (C, H, W).

        Args:
            images (Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]]):
                A single image or a list of images to process. Images can be provided
                as file paths, PIL Images, NumPy arrays, or torch.Tensors.
            threshold (float, optional):
                The minimum confidence score needed to consider a detected bounding box valid.
            **kwargs:
                Additional keyword arguments.

        Returns:
            Union[sv.Detections, List[sv.Detections]]: A single or multiple Detections
                objects, each containing bounding box coordinates, confidence scores,
                and class IDs.
        """
        if not self._is_optimized_for_inference and not self._has_warned_about_not_being_optimized_for_inference:
            logger.warning(
                "Model is not optimized for inference. "
                "Latency may be higher than expected. "
                "You can optimize the model for inference by calling model.optimize_for_inference()."
            )
            self._has_warned_about_not_being_optimized_for_inference = True

            self.model.model.eval()

        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []

        for img in images:

            if isinstance(img, str):
                img = Image.open(img)

            if not isinstance(img, torch.Tensor):
                img = F.to_tensor(img)
            
            if (img > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is "
                    "normalized (scaled to [0, 1])."
                )
            if img.shape[0] != 3:
                raise ValueError(
                    f"Invalid image shape. Expected 3 channels (RGB), but got "
                    f"{img.shape[0]} channels."
                )
            img_tensor = img
            
            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))

            img_tensor = img_tensor.to(self.model.device)
            img_tensor = F.normalize(img_tensor, self.means, self.stds)
            img_tensor = F.resize(img_tensor, (self.model.resolution, self.model.resolution))

            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)

        if self._is_optimized_for_inference:
            if self._optimized_resolution != batch_tensor.shape[2]:
                # this could happen if someone manually changes self.model.resolution after optimizing the model
                raise ValueError(f"Resolution mismatch. "
                                 f"Model was optimized for resolution {self._optimized_resolution}, "
                                 f"but got {batch_tensor.shape[2]}. "
                                 "You can explicitly remove the optimized model by calling model.remove_optimized_model().")
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != batch_tensor.shape[0]:
                    raise ValueError(f"Batch size mismatch. "
                                     f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                                     f"but got {batch_tensor.shape[0]}. "
                                     "You can explicitly remove the optimized model by calling model.remove_optimized_model(). "
                                     "Alternatively, you can recompile the optimized model for a different batch size "
                                     "by calling model.optimize_for_inference(batch_size=<new_batch_size>).")

        with torch.inference_mode():
            if self._is_optimized_for_inference:
                predictions = self.model.inference_model(batch_tensor.to(dtype=self._optimized_dtype))
            else:
                predictions = self.model.model(batch_tensor)
            if isinstance(predictions, tuple):
                predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0]
                }
            target_sizes = torch.tensor(orig_sizes, device=self.model.device)
            results = self.model.postprocessors["bbox"](predictions, target_sizes=target_sizes)

        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            detections = sv.Detections(
                xyxy=boxes.float().cpu().numpy(),
                confidence=scores.float().cpu().numpy(),
                class_id=labels.cpu().numpy(),
            )
            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]
    
    def deploy_to_roboflow(self, workspace: str, project_id: str, version: str, api_key: str = None, size: str = None):
        """
        Deploy the trained RF-DETR model to Roboflow.

        Deploying with Roboflow will create a Serverless API to which you can make requests.

        You can also download weights into a Roboflow Inference deployment for use in Roboflow Workflows and on-device deployment.

        Args:
            workspace (str): The name of the Roboflow workspace to deploy to.
            project_ids (List[str]): A list of project IDs to which the model will be deployed
            api_key (str, optional): Your Roboflow API key. If not provided,
                it will be read from the environment variable `ROBOFLOW_API_KEY`.
            size (str, optional): The size of the model to deploy. If not provided,
                it will default to the size of the model being trained (e.g., "rfdetr-base", "rfdetr-large", etc.).
            model_name (str, optional): The name you want to give the uploaded model.
            If not provided, it will default to "<size>-uploaded".
        Raises:
            ValueError: If the `api_key` is not provided and not found in the environment
                variable `ROBOFLOW_API_KEY`, or if the `size` is not set for custom architectures.
        """
        from roboflow import Roboflow
        import shutil
        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError("Set api_key=<KEY> in deploy_to_roboflow or export ROBOFLOW_API_KEY=<KEY>")


        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace)

        if self.size is None and size is None:
            raise ValueError("Must set size for custom architectures")

        size = self.size or size
        tmp_out_dir = ".roboflow_temp_upload"
        os.makedirs(tmp_out_dir, exist_ok=True)
        outpath = os.path.join(tmp_out_dir, "weights.pt")
        torch.save(
            {
                "model": self.model.model.state_dict(),
                "args": self.model.args
            }, outpath
        )
        project = workspace.project(project_id)
        version = project.version(version)
        version.deploy(
            model_type=size,
            model_path=tmp_out_dir,
            filename="weights.pt"
        )
        shutil.rmtree(tmp_out_dir)

    @classmethod
    def download_from_roboflow(cls, workspace: str, project_id: str, version: str, api_key: str = None, local_weights_path: str = None):
        """
        Download trained model weights from Roboflow and create a local RF-DETR model instance.
        
        This method allows you to download weights from a model you trained on Roboflow
        and use them locally for inference on your own hardware.
        
        Args:
            workspace (str): The name of the Roboflow workspace where your model is stored.
            project_id (str): The ID of the project containing your trained model.
            version (str): The version number of your trained model.
            api_key (str, optional): Your Roboflow API key. If not provided,
                it will be read from the environment variable `ROBOFLOW_API_KEY`.
            local_weights_path (str, optional): Local path to save the downloaded weights.
                If not provided, weights will be saved to ./roboflow_weights_{project_id}_v{version}.pt
                
        Returns:
            RFDETR: An RF-DETR model instance with the downloaded weights loaded.
            
        Raises:
            ValueError: If the `api_key` is not provided and not found in the environment
                variable `ROBOFLOW_API_KEY`.
            FileNotFoundError: If the weights cannot be downloaded from Roboflow.
            
        Example:
            ```python
            from rfdetr import RFDETRBase
            
            # Download your colonoscopy model trained on Roboflow
            model = RFDETRBase.download_from_roboflow(
                workspace="your-workspace",
                project_id="colonoscopy-detection",
                version="1",
                api_key="your_api_key"  # or set ROBOFLOW_API_KEY env var
            )
            
            # Now use the model for inference on your local videos
            detections = model.predict("path/to/colonoscopy_video_frame.jpg")
            ```
        """
        from roboflow import Roboflow
        import tempfile
        import requests
        
        # Get API key
        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError("Set api_key=<KEY> in download_from_roboflow or export ROBOFLOW_API_KEY=<KEY>")
        
        # Set default weights path if not provided
        if local_weights_path is None:
            local_weights_path = f"./roboflow_weights_{project_id}_v{version}.pt"
        
        print(f"Downloading weights from Roboflow project {project_id} version {version}...")
        
        try:
            # Initialize Roboflow client
            rf = Roboflow(api_key=api_key)
            workspace_obj = rf.workspace(workspace)
            project = workspace_obj.project(project_id)
            version_obj = project.version(version)
            
            # Try to get the model weights via the Roboflow API
            # Method 1: Try to download the model directly
            try:
                # Some versions of Roboflow SDK support model.download()
                model_info = version_obj.model()
                if hasattr(model_info, 'download_weights'):
                    weights_file = model_info.download_weights(local_weights_path)
                elif hasattr(model_info, 'download'):
                    weights_file = model_info.download(local_weights_path)
                else:
                    raise AttributeError("No download method found")
                    
            except (AttributeError, Exception) as e:
                print(f"Direct download method not available ({e}), trying API approach...")
                
                # Method 2: Use REST API to get model weights URL
                base_url = "https://api.roboflow.com"
                headers = {"Authorization": f"Bearer {api_key}"}
                
                # Get model information
                model_url = f"{base_url}/{workspace}/{project_id}/{version}/model"
                response = requests.get(model_url, headers=headers)
                
                if response.status_code != 200:
                    raise Exception(f"Failed to get model info: {response.status_code} - {response.text}")
                
                model_data = response.json()
                
                # Look for weights download URL in the response
                weights_url = None
                if 'weights' in model_data:
                    weights_url = model_data['weights'].get('url') or model_data['weights'].get('download_url')
                elif 'model' in model_data:
                    weights_url = model_data['model'].get('weights_url') or model_data['model'].get('download_url')
                
                if not weights_url:
                    # Method 3: Try to construct the weights URL based on common patterns
                    weights_url = f"{base_url}/{workspace}/{project_id}/{version}/weights"
                
                # Download the weights file
                print(f"Downloading weights from: {weights_url}")
                weights_response = requests.get(weights_url, headers=headers, stream=True)
                
                if weights_response.status_code != 200:
                    raise Exception(f"Failed to download weights: {weights_response.status_code} - {weights_response.text}")
                
                # Save weights to local file
                with open(local_weights_path, 'wb') as f:
                    for chunk in weights_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Weights downloaded and saved to: {local_weights_path}")
        
        except Exception as e:
            # Provide helpful error message with instructions
            error_msg = (
                f"Failed to download weights from Roboflow: {str(e)}\n\n"
                f"Please verify:\n"
                f"1. Your workspace '{workspace}' exists and you have access\n"
                f"2. Project ID '{project_id}' is correct\n"
                f"3. Version '{version}' exists and has a trained RF-DETR model\n"
                f"4. Your API key is valid and has access to this project\n"
                f"5. The model was trained with RF-DETR (not another model type)\n\n"
                f"Alternative: You can manually download weights from the Roboflow dashboard:\n"
                f"1. Go to your project in Roboflow\n"
                f"2. Navigate to the 'Models' tab\n"
                f"3. Find your RF-DETR model and download the weights file\n"
                f"4. Use RFDETRBase(pretrain_weights='path/to/downloaded/weights.pt')\n"
            )
            raise FileNotFoundError(error_msg)
        
        # Create model instance with downloaded weights
        try:
            model_instance = cls(pretrain_weights=local_weights_path)
            print(f"Successfully loaded RF-DETR model with weights from Roboflow!")
            return model_instance
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model with downloaded weights: {str(e)}. "
                f"The downloaded weights file may be corrupted or incompatible."
            )


class RFDETRBase(RFDETR):
    """
    Train an RF-DETR Base model (29M parameters).
    """
    size = "rfdetr-base"
    def get_model_config(self, **kwargs):
        return RFDETRBaseConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRLarge(RFDETR):
    """
    Train an RF-DETR Large model.
    """
    size = "rfdetr-large"
    def get_model_config(self, **kwargs):
        return RFDETRLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRNano(RFDETR):
    """
    Train an RF-DETR Nano model.
    """
    size = "rfdetr-nano"
    def get_model_config(self, **kwargs):
        return RFDETRNanoConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRSmall(RFDETR):
    """
    Train an RF-DETR Small model.
    """
    size = "rfdetr-small"
    def get_model_config(self, **kwargs):
        return RFDETRSmallConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRMedium(RFDETR):
    """
    Train an RF-DETR Medium model.
    """
    size = "rfdetr-medium"
    def get_model_config(self, **kwargs):
        return RFDETRMediumConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)