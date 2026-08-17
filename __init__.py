from .node import *
from .bubble_node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    "Apply EasyOCR": ApplyEasyOCR,
    "Apply EasyOCR (Combined Mask)": ApplyEasyOCRCombined,
    "Apply Bubble Detector": ApplyBubbleDetector,
    "Apply Bubble Detector (Combined Mask)": ApplyBubbleDetectorCombined,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Apply EasyOCR": "Apply EasyOCR",
    "Apply EasyOCR (Combined Mask)": "Apply EasyOCR (Combined Mask)",
    "Apply Bubble Detector": "Apply Bubble Detector",
    "Apply Bubble Detector (Combined Mask)": "Apply Bubble Detector (Combined Mask)",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
