import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Lazy loaded models
MODELS = {}

# Load ImageNet class labels (1000 classes)
# Official label mapping from: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
IMAGENET_LABELS = None
def load_imagenet_labels():
    global IMAGENET_LABELS
    if IMAGENET_LABELS is None:
        with open("imagenet_classes.txt", "r") as f:
            IMAGENET_LABELS = [line.strip() for line in f.readlines()]
    return IMAGENET_LABELS

# Preprocessing transform for ImageNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(name):
    if name not in MODELS:
        if name == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif name == "mobilenetv2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif name == "efficientnetb0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unknown model name")

        model.eval()
        MODELS[name] = model
    return MODELS[name]

def predict_topk(img_pil, model_name, k=5):
    model = load_model(model_name)
    labels = load_imagenet_labels()

    img_t = preprocess(img_pil).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_t)
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_idxs = probs.topk(k)

    results = []
    for prob, idx in zip(topk_probs[0], topk_idxs[0]):
        cls_name = labels[idx]
        results.append((cls_name, float(prob)))

    return results
