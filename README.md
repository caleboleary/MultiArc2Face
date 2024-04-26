Originally forked from [this](https://github.com/camenduru/Arc2Face-hf), thanks camenduru!

# MultiArc2Face

This repo builds upon the work done to train the [Arc2Face](https://github.com/foivospar/Arc2Face) model by foivospar.

Specifically, this epxlores getting the embeddings of multiple of the same face and averaging them in different ways to gain more, and hopefully more correct, information about the face one is trying to reproduce with the Arc2Face model.

To run in google colab:

```
%cd /content
!GIT_LFS_SKIP_SMUDGE=1 git clone -b main https://github.com/caleboleary/MultiArc2Face
%cd /content/MultiArc2Face

!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/freddie.png -O /content/MultiArc2Face/assets/examples/freddie.png
!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/jackie.png -O /content/MultiArc2Face/assets/examples/jackie.png
!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/joacquin.png -O /content/MultiArc2Face/assets/examples/joacquin.png
!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/scrfd_10g_bnkps.onnx -O /content/MultiArc2Face/models/antelopev2/scrfd_10g_bnkps.onnx
!wget https://huggingface.co/camenduru/Arc2Face/resolve/main/arcface.onnx -O /content/MultiArc2Face/models/antelopev2/arcface.onnx

!pip install -r requirements.txt

!python app.py
```

Sample using median option, non-cherry picked

![sample of different face counts](assets/sample.png)
