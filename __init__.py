from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    "Apply EasyOCR": ApplyEasyOCR,
    "Apply EasyOCR (Combined Mask)": ApplyEasyOCRCombined,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Apply EasyOCR": "Apply EasyOCR",
    "Apply EasyOCR (Combined Mask)": "Apply EasyOCR (Combined Mask)",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
