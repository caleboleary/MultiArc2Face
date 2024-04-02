---
title: Arc2Face
emoji: 🔥
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.23.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference





To run in google colab:

```
%cd /content
!GIT_LFS_SKIP_SMUDGE=1 git clone -b dev https://github.com/caleboleary/MultiArc2Face
%cd /content/Arc2Face-hf

!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/freddie.png -O /content/Arc2Face-hf/assets/examples/freddie.png
!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/jackie.png -O /content/Arc2Face-hf/assets/examples/jackie.png
!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/joacquin.png -O /content/Arc2Face-hf/assets/examples/joacquin.png
!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/scrfd_10g_bnkps.onnx -O /content/Arc2Face-hf/models/antelopev2/scrfd_10g_bnkps.onnx
!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/arcface.onnx -O /content/Arc2Face-hf/models/antelopev2/arcface.onnx

!pip install -q diffusers==0.22.0 transformers==4.34.1 accelerate onnxruntime-gpu gradio insightface

!python app.py
```
