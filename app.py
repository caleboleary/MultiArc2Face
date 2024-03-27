import sys
sys.path.append('./')

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from arc2face import CLIPTextModelWrapper, project_face_embs

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import random

import gradio as gr
import spaces

# global variable
MAX_SEED = np.iinfo(np.int32).max
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32


# download models
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/config.json", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/diffusion_pytorch_model.safetensors", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/config.json", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/pytorch_model.bin", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir="./models/antelopev2")

# Load face detection and recognition package
if device=="cuda":
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
else:
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load pipeline
base_model = 'runwayml/stable-diffusion-v1-5'
encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=dtype
)
unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=dtype
)
pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None
    )
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def get_example():
    case = [
        [
            './assets/examples/freeman.jpg',
        ],
        [
            './assets/examples/lily.png',
        ],
        [
            './assets/examples/joacquin.png',
        ],
        [
            './assets/examples/jackie.png',
        ], 
        [
            './assets/examples/freddie.png',
        ],
        [
            './assets/examples/hepburn.png',
        ],
    ]
    return case

def run_example(img_file):
    return generate_image(img_file, 25, 3, 23, 2)

@spaces.GPU
def generate_image(image_path, num_steps, guidance_scale, seed, num_images, progress=gr.Progress(track_tqdm=True)):

    if image_path is None:
        raise gr.Error(f"Cannot find any input face image! Please upload a face image.")
    
    img = np.array(Image.open(image_path))[:,:,::-1]

    # Face detection and ID-embedding extraction
    faces = app.get(img)
    
    if len(faces) == 0:
        raise gr.Error(f"Face detection failed! Please try with another image.")
    
    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
    id_emb = torch.tensor(faces['embedding'], dtype=dtype)[None].to(device)
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
    id_emb = project_face_embs(pipeline, id_emb)    # pass throught the encoder
                    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("Start inference...")        
    images = pipeline(
        prompt_embeds=id_emb,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale, 
        num_images_per_prompt=num_images,
        generator=generator
    ).images

    return images

### Description
title = r"""
<h1>Arc2Face: A Foundation Model of Human Faces</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://arc2face.github.io/' target='_blank'><b>Arc2Face: A Foundation Model of Human Faces</b></a>.<br>

Steps:<br>
1. Upload an image with a face. If multiple faces are detected, we use the largest one. For images with already tightly cropped faces, detection may fail, try images with a larger margin.
2. Click <b>Submit</b> to generate new images of the subject.
"""

Footer = r"""
---
📝 **Citation**
<br>
If you find Arc2Face helpful for your research, please consider citing our paper:
```bibtex
@misc{paraperas2024arc2face,
      title={Arc2Face: A Foundation Model of Human Faces}, 
      author={Foivos Paraperas Papantoniou and Alexandros Lattas and Stylianos Moschoglou and Jiankang Deng and Bernhard Kainz and Stefanos Zafeiriou},
      year={2024},
      eprint={2403.11641},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
"""

css = '''
.gradio-container {width: 85% !important}
'''
with gr.Blocks(css=css) as demo:

    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            
            # upload face image
            img_file = gr.Image(label="Upload a photo with a face", type="filepath")
            
            submit = gr.Button("Submit", variant="primary")
            
            with gr.Accordion(open=False, label="Advanced Options"):
                num_steps = gr.Slider( 
                    label="Number of sample steps",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=25,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=3,
                )
                num_images = gr.Slider(
                    label="Number of output images",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=2,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")

        submit.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[img_file, num_steps, guidance_scale, seed, num_images],
            outputs=[gallery]
        )
    
    
    gr.Examples(
        examples=get_example(),
        inputs=[img_file],
        run_on_click=True,
        fn=run_example,
        outputs=[gallery],
    )
    
    gr.Markdown(Footer)

demo.launch()