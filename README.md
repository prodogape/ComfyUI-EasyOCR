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

# INSTALL
This node calls the official Python package, and you also need to install the following dependencies:

```
pip install easyocr
```

# MODEL
This node will automatically download the corresponding model based on the language you select.
```
ComfyUI
    models
        EasyOCR
            latin_g2.pth
            zh_sim_g2.pth
            craft_mlt_25k.pth
```
