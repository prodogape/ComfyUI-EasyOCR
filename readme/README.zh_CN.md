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

# 需要安装的依赖
本节点调用的是官方提供的python包,你还需要安装下面的依赖

```
pip install easyocr
```

# 模型
本节点会自动根据你选择的语言下载对应模型
```
ComfyUI
    models
        EasyOCR
            latin_g2.pth
            zh_sim_g2.pth
            craft_mlt_25k.pth
```
