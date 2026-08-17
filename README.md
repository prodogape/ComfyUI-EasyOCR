# ComfyUI-EasyOCR
This node is primarily based on Easy-OCR to implement OCR text recognition functionality.
![image](/docs/workflow.png)

# README
- [English](README.md)
- [简体中文](readme/README.zh_CN.md)

# NODES
|name                          |description                                                                |
|------------------------------|---------------------------------------------------------------------------|
|Apply EasyOCR                 |the OCR model will be used, and the model will be automatically downloaded.|
|Apply EasyOCR (Combined Mask) |same as Apply EasyOCR, but returns a single union mask of all text regions.|
|Apply Bubble Detector         |YOLO-based speech/thought-bubble segmentation; auto-downloads the model.  |
|Apply Bubble Detector (Combined Mask) |same as Apply Bubble Detector, but returns one union mask per image.|

# INSTALL
This node calls the official Python packages. Dependencies are installed automatically via `requirements.txt`:

```
pip install -r requirements.txt
# i.e. easyocr + ultralytics
```

# MODEL
## EasyOCR
This node will automatically download the corresponding model based on the language you select.
```
ComfyUI
    models
        EasyOCR
            latin_g2.pth
            zh_sim_g2.pth
            craft_mlt_25k.pth
```

## Bubble Detector
The default `comic-speech-bubble-detector.pt` is auto-downloaded on first use into `models/YOLO/`. Drop any custom YOLO `.pt`/`.pth`/`.onnx` (seg or detect) in that folder to pick it from the dropdown.
```
ComfyUI
    models
        YOLO
            comic-speech-bubble-detector.pt
```
