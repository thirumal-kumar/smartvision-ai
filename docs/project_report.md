# SmartVision AI – Project Report

## 1. Introduction
SmartVision AI is a unified computer vision system integrating image classification 
and real-time object detection using deep learning architectures.

## 2. Literature Review
- CNNs for image classification  
- Residual networks and efficiency scaling  
- Single-shot detection networks (YOLO family)  
- Applications in surveillance and analytics  

## 3. Methodology
### 3.1 System Architecture
See `architecture.png`.

### 3.2 Classification Pipeline
- Pretrained CNNs (VGG16, ResNet50, MobileNetV2, EfficientNetB0)
- ImageNet preprocessing
- Top-5 prediction

### 3.3 Object Detection
- YOLOv8n pretrained model
- Adjustable confidence & IoU
- Bounding box visualization

## 4. Results
- Sample outputs included  
- Performance characteristics summarized  

## 5. Discussion
- EfficientNetB0 offers best accuracy-efficiency balance  
- YOLOv8n enables real-time detection even on CPU  
- Modular design allows future expansion  

## 6. Conclusion
SmartVision AI demonstrates the integration of advanced deep learning models into 
a coherent recognition system suitable for research and deployment.

## 7. Future Work
- Domain-specific dataset training  
- Grad-CAM explainability  
- ONNX deployment  
- Video feed analytics  

