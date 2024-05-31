import folder_paths
import json
import cv2
import easyocr
import os
import logging
from PIL import Image
import numpy as np
import torch

logger = logging.getLogger("ComfyUI-EasyOCR")
model_dir_name = "EasyOCR"

lang_list = {
    "English": "en",
    "简体中文": "ch_sim",
    "繁體中文": "ch_tra",
    "العربية": "ar",
    "Azərbaycan": "az",
    "Euskal": "eu",
    "Bosanski": "bs",
    "Български": "bg",
    "Català": "ca",
    "Hrvatski": "hr",
    "Čeština": "cs",
    "Dansk": "da",
    "Nederlands": "nl",
    "Eesti": "et",
    "Suomi": "fi",
    "Français": "fr",
    "Galego": "gl",
    "Deutsch": "de",
    "Ελληνικά": "el",
    "עברית": "he",
    "हिन्दी": "hi",
    "Magyar": "hu",
    "Íslenska": "is",
    "Indonesia": "id",
    "Italiano": "it",
    "日本語": "ja",
    "한국어": "ko",
    "Latviešu": "lv",
    "Lietuvių": "lt",
    "Македонски": "mk",
    "Norsk": "no",
    "Polski": "pl",
    "Português": "pt",
    "Română": "ro",
    "Русский": "ru",
    "Српски": "sr",
    "Slovenčina": "sk",
    "Slovenščina": "sl",
    "Español": "es",
    "Svenska": "sv",
    "ไทย": "th",
    "Türkçe": "tr",
    "Українська": "uk",
    "Tiếng Việt": "vi",
}

def get_lang_list():
    result = []
    for key, value in lang_list.items():
        result.append(key)
    return result


def get_classes(label):
    label = label.lower()
    labels = label.split(",")
    result = []
    for l in labels:
        for key, value in lang_list.items():
            if l == value:
                result.append(key)
                break
    return result


def get_classes2(label):
    label = label.lower()
    labels = label.split(",")
    result = []
    for l in labels:
        for key, value in lang_list.items():
            if l == key:
                result.append(value)
                break
    return result

def plot_boxes_to_image(image_pil, tgt):
    image_np = np.array(image_pil)
    H, W = tgt["size"]
    result = tgt["result"]

    res_mask = []
    res_image = []

    font_scale = 1
    box_color = (255, 0, 0)
    text_color = (255, 255, 255)
    image_np = image_np[..., :3]

    # Make a copy of the image to avoid modifying the original image
    image_with_boxes = np.copy(image_np)

    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": None,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }

    for item in result:
        formatted_points, label, threshold = item

        x1, y1 = formatted_points[0]
        x2, y2 = formatted_points[2]
        threshold = round(threshold, 2)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        points = [[x1, y1], [x2, y2]]

        # save labelme json
        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }
        labelme_data["shapes"].append(shape)

        # change lable
        label = label + ":" + str(threshold)
        shape["threshold"] = str(threshold)

        # Draw rectangle on the copied image
        cv2.rectangle(
            image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3
        )

        # Draw label on the copied image
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        label_ymin = max(y1, label_size[1] + 10)
        cv2.rectangle(
            image_with_boxes,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            box_color,
            -1,
        )
        cv2.putText(
            image_with_boxes,
            label,
            (x1, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            2,
            cv2.LINE_AA,
            bottomLeftOrigin=False,
        )

        # Draw mask
        mask = np.zeros((H, W, 1), dtype=np.uint8)
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        res_mask.append(mask_tensor)

    if len(res_mask) == 0:
        mask = np.zeros((H, W, 1), dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        res_mask.append(mask_tensor)

    # Convert the modified image to a torch tensor
    image_with_boxes_tensor = torch.from_numpy(
        image_with_boxes.astype(np.float32) / 255.0
    )
    image_with_boxes_tensor = torch.unsqueeze(image_with_boxes_tensor, 0)
    res_image.append(image_with_boxes_tensor)

    return res_image, res_mask, labelme_data

class ApplyEasyOCR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gpu": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "detect": (
                    ["choose", "input"],
                    {"default": "choose"},
                ),
                "language_list": (
                    get_lang_list(),
                    {"default": "简体中文"},
                ),
                "language_name": (
                    "STRING",
                    {"default": "ch_sim,en", "multiline": False},
                ),
            },
        }

    CATEGORY = "ComfyUI-EasyOCR"
    FUNCTION = "main"
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "JSON",
    )

    def main(self, image, gpu, detect, language_list, language_name):
        res_images = []
        res_masks = []
        res_labels = []

        for item in image:
            image_pil = Image.fromarray(np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")

            language = None
            if detect == "choose":
                language = get_classes2(language_list)
            else:
                language = get_classes(language_name)

            model_storage_directory = os.path.join(folder_paths.models_dir, model_dir_name)
            if not os.path.exists(model_storage_directory):
                os.makedirs(model_storage_directory)

            reader  = easyocr.Reader(language, model_storage_directory=model_storage_directory,gpu=gpu)
            result = reader.readtext(np.array(image_pil))

            size = image_pil.size
            pred_dict = {
                "size": [size[1], size[0]],
                "result":result
            }

            image_tensor, mask_tensor, labelme_data = plot_boxes_to_image(image_pil, pred_dict)

            res_images.extend(image_tensor)
            res_masks.extend(mask_tensor)
            res_labels.append(labelme_data)

            if len(res_images) == 0:
                res_images.extend(item)
            if len(res_masks) == 0:
                mask = np.zeros((height, width, 1), dtype=np.uint8)
                empty_mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
                res_masks.extend(empty_mask)

        print(res_labels)
        return (
            torch.cat(res_images, dim=0),
            torch.cat(res_masks, dim=0),
            res_labels,
        )