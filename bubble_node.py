import os
import logging

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import folder_paths


logger = logging.getLogger("ComfyUI-EasyOCR")

# Models live under ComfyUI's models dir in a dedicated "YOLO" folder, so users
# can drop in any custom .pt (seg or detect) and pick it from the dropdown.
YOLO_MODEL_DIR = os.path.join(folder_paths.models_dir, "YOLO")
DEFAULT_MODEL = "comic-speech-bubble-detector.pt"

# Known models that are auto-downloaded on first use. Add entries here to ship
# more preconfigured bubble/text detectors.
MODEL_URLS = {
    "comic-speech-bubble-detector.pt": (
        "https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m"
        "/resolve/main/comic-speech-bubble-detector.pt"
    ),
}

BUBBLE_CATEGORY = "ComfyUI-EasyOCR"


def get_yolo_model_list():
    models = []
    if os.path.isdir(YOLO_MODEL_DIR):
        for f in os.listdir(YOLO_MODEL_DIR):
            if f.lower().endswith((".pt", ".pth", ".onnx")):
                models.append(f)
    if DEFAULT_MODEL not in models:
        models.insert(0, DEFAULT_MODEL)
    if not models:
        models.append(DEFAULT_MODEL)
    return models


def ensure_model(model_name):
    os.makedirs(YOLO_MODEL_DIR, exist_ok=True)
    path = os.path.join(YOLO_MODEL_DIR, model_name)
    if not os.path.exists(path) and model_name in MODEL_URLS:
        logger.info("Downloading YOLO model '%s'...", model_name)
        torch.hub.download_url_to_file(MODEL_URLS[model_name], path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YOLO model '{model_name}' not found at {path}. Place a .pt file "
            f"in {YOLO_MODEL_DIR} or choose a model that can be auto-downloaded."
        )
    return path


def detect_bubbles(image_np, model_path, conf, device):
    """Run a YOLO seg/detect model and return a list of detections.

    Each detection is a dict: {"points": [[x,y],...], "shape_type": str,
    "label": str, "conf": float}. Polygons are used when the model exposes
    segmentation masks; bounding boxes are used as a fallback for detection-only
    weights. Mask rasterization is done at full image resolution, so output
    masks always match the input H/W regardless of the model's internal size.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model(image_np, conf=conf, verbose=False, device=device)
    res = results[0]

    H, W = image_np.shape[:2]
    detections = []

    masks_obj = getattr(res, "masks", None)
    if masks_obj is not None and getattr(masks_obj, "xy", None):
        names = getattr(res, "names", {}) or {}
        boxes_obj = getattr(res, "boxes", None)
        confs = (
            boxes_obj.conf.cpu().numpy() if boxes_obj is not None else None
        )
        cls_ids = (
            boxes_obj.cls.cpu().numpy().astype(int)
            if boxes_obj is not None and getattr(boxes_obj, "cls", None) is not None
            else None
        )
        for i, poly in enumerate(masks_obj.xy):
            if len(poly) < 3:
                continue
            cid = int(cls_ids[i]) if cls_ids is not None and i < len(cls_ids) else 0
            label = names.get(cid, "bubble") if isinstance(names, dict) else "bubble"
            c = float(confs[i]) if confs is not None and i < len(confs) else 1.0
            detections.append(
                {
                    "points": np.round(poly).astype(np.int32).tolist(),
                    "shape_type": "polygon",
                    "label": str(label),
                    "conf": round(c, 3),
                }
            )

    if not detections and getattr(res, "boxes", None) is not None:
        boxes = res.boxes
        if getattr(boxes, "xyxy", None) is not None and len(boxes.xyxy) > 0:
            names = getattr(res, "names", {}) or {}
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
            cls_ids = (
                boxes.cls.cpu().numpy().astype(int)
                if getattr(boxes, "cls", None) is not None
                else [0] * len(xyxy)
            )
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                label = names.get(int(cls_ids[i]), "bubble") if isinstance(names, dict) else "bubble"
                c = float(confs[i]) if confs is not None and i < len(confs) else 1.0
                detections.append(
                    {
                        "points": [[x1, y1], [x2, y2]],
                        "shape_type": "rectangle",
                        "label": str(label),
                        "conf": round(c, 3),
                    }
                )

    return detections, (H, W)


def render_preview_and_masks(image_pil, detections, size):
    H, W = size
    overlay = image_pil.copy()
    draw = ImageDraw.Draw(overlay)

    res_mask = []
    labelme_shapes = []

    for det in detections:
        pts = det["points"]
        label = det["label"]
        c = det["conf"]
        stype = det["shape_type"]

        # Full-resolution binary mask for this detection.
        mask = np.zeros((H, W), dtype=np.uint8)
        if stype == "polygon":
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
            draw.line(pts + [pts[0]], fill=(255, 0, 0), width=3)
        else:
            (x1, y1), (x2, y2) = pts
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)

        res_mask.append(torch.from_numpy(mask).unsqueeze(0).float())  # [1,H,W]

        labelme_shapes.append(
            {
                "label": f"{label}:{c}",
                "points": pts,
                "group_id": None,
                "shape_type": stype,
                "flags": {},
            }
        )

    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": labelme_shapes,
        "imagePath": None,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }

    if len(res_mask) == 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        res_mask.append(torch.from_numpy(empty).unsqueeze(0).float())

    preview = torch.from_numpy(np.array(overlay).astype(np.float32) / 255.0)
    preview = preview.unsqueeze(0)  # [1,H,W,3]

    return preview, res_mask, labelme_data


def _device_str(gpu):
    if gpu and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class ApplyBubbleDetector:
    """YOLO-based speech/thought-bubble detector. Returns one mask per
    detected bubble (stacked along the batch dim), a preview image with the
    detections outlined, and a labelme-style JSON. Designed for inpainting:
    the masks are full-resolution segmentation polygons (or boxes as a
    fallback for detection-only weights)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (
                    get_yolo_model_list(),
                    {"default": DEFAULT_MODEL},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "gpu": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = BUBBLE_CATEGORY
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK", "JSON")

    def main(self, image, model_name, confidence, gpu):
        model_path = ensure_model(model_name)
        device = _device_str(gpu)

        res_images = []
        res_masks = []
        res_labels = []

        for item in image:
            arr = np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)
            image_pil = Image.fromarray(arr).convert("RGB")

            detections, size = detect_bubbles(arr, model_path, confidence, device)
            preview, masks, labelme_data = render_preview_and_masks(
                image_pil, detections, size
            )

            res_images.append(preview)
            res_masks.extend(masks)
            res_labels.append(labelme_data)

        return (
            torch.cat(res_images, dim=0),
            torch.cat(res_masks, dim=0),
            res_labels,
        )


class ApplyBubbleDetectorCombined:
    """Same detection as ApplyBubbleDetector, but returns a single union mask
    (element-wise max of all bubble masks) per input image. This is what most
    inpainting pipelines want: one region covering all text/bubbles.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (
                    get_yolo_model_list(),
                    {"default": DEFAULT_MODEL},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "gpu": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = BUBBLE_CATEGORY
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK", "JSON")

    def main(self, image, model_name, confidence, gpu):
        model_path = ensure_model(model_name)
        device = _device_str(gpu)

        res_images = []
        res_masks = []
        res_labels = []

        for item in image:
            arr = np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)
            image_pil = Image.fromarray(arr).convert("RGB")

            detections, size = detect_bubbles(arr, model_path, confidence, device)
            preview, masks, labelme_data = render_preview_and_masks(
                image_pil, detections, size
            )

            stacked = torch.cat(masks, dim=0)  # [N,H,W]
            combined = stacked.max(dim=0, keepdim=True)[0]  # [1,H,W]

            res_images.append(preview)
            res_masks.append(combined)
            res_labels.append(labelme_data)

        return (
            torch.cat(res_images, dim=0),
            torch.cat(res_masks, dim=0),
            res_labels,
        )
