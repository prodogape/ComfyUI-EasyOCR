# ComfyUI-EasyOCR
本节点主要是基于Easy-OCR，实现OCR文本识别功能
![image](/docs/workflow.png)

# 切换语言
- [English](README.md)
- [简体中文](readme/README.zh_CN.md)

# 节点
|名称                          |描述                             |
|------------------------------|--------------------------------|
|Apply EasyOCR                 |默认，使用OCR模型，自动下载模型    |
|Apply EasyOCR (Combined Mask) |同上，返回所有文本区域的合并蒙版    |
|Apply Bubble Detector         |基于 YOLO 的对话气泡分割，自动下载模型 |
|Apply Bubble Detector (Combined Mask) |同上，返回单张合并蒙版 |

# 需要安装的依赖
本节点调用的是官方提供的python包，依赖通过 `requirements.txt` 自动安装：

```
pip install -r requirements.txt
# 即 easyocr + ultralytics
```

# 模型
## EasyOCR
本节点会自动根据你选择的语言下载对应模型
```
ComfyUI
    models
        EasyOCR
            latin_g2.pth
            zh_sim_g2.pth
            craft_mlt_25k.pth
```

## 气泡检测
默认的 `comic-speech-bubble-detector.pt` 会在首次使用时自动下载到 `models/YOLO/`。也可将任意自定义 YOLO `.pt`/`.pth`/`.onnx`（分割或检测）放入该目录后从下拉菜单选择。
```
ComfyUI
    models
        YOLO
            comic-speech-bubble-detector.pt
```
