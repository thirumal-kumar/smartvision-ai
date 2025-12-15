# SmartVision AI – Intelligent Multi-Class Object Recognition System

SmartVision AI is a modular, extensible computer vision system capable of performing  
**high-accuracy image classification** and **real-time object detection** using modern  
deep learning models such as VGG16, ResNet50, MobileNetV2, EfficientNetB0, and YOLOv8.

This repository implements a full working prototype with a clean Streamlit interface.

---

## 🚀 Features

### 🔹 Image Classification (PyTorch Pretrained Models)
- VGG16  
- ResNet50  
- MobileNetV2  
- EfficientNetB0  
- ImageNet Top‑5 prediction  
- Unified inference module  

### 🔹 Object Detection (YOLOv8n)
- Real-time CPU-compatible inference  
- Bounding box visualization  
- Supports 25-class COCO subset  
- Adjustable confidence & IoU thresholds  

### 🔹 Modular Architecture
- Separation between classification, detection, and utilities  
- Easy to extend or replace with custom models  
- Clear project structure  

### 🔹 Streamlit Web Interface
- Multi‑page navigation  
- Upload or use sample images  
- Human‑readable outputs  
- Works fully on CPU  

---

## 📂 Repository Structure

```
smartvision/
│── app.py
│── README.md
│── requirements.txt
│── imagenet_classes.txt
│
├── classification/
│   └── models_loader.py
│
├── detection/
│   └── yolo_detect.py
│
├── utils/
│   └── viz.py
│
└── assets/
    └── sample_images/
```

---

## 🧠 Model Comparison Summary

| Model | Strengths | Limitations | Typical Use |
|-------|-----------|-------------|--------------|
| **VGG16** | Strong baseline, stable features | Heavy | Academic baselines |
| **ResNet50** | Good generalization | Moderately heavy | General-purpose CV |
| **MobileNetV2** | Lightweight, fast | Slightly lower accuracy | Mobile/edge deployment |
| **EfficientNetB0** | Best accuracy‑size ratio | Moderate speed | Production apps |
| **YOLOv8n** | Very fast detection | Lower accuracy than larger YOLOs | Real‑time tasks |

---

## 🏗 System Architecture

SmartVision AI follows a simplified three‑layer architecture:

1. **Input & Preprocessing Layer**  
2. **Inference Layer (Classification / Detection)**  
3. **Visualization & Output Layer**  

For detailed diagram, see `/docs/architecture.png`.

---

## 🔧 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10 recommended.

---

## 📘 Documentation

A structured project report can be found under:

```
/docs/project_report.md
```

---

## 🎯 Future Enhancements

- Custom dataset training  
- Grad‑CAM explainability  
- Real‑time video detection  
- ONNX/TensorRT deployment  
- REST API integration  

---

## 📝 License
This project is intended for academic and research demonstration.

