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

import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import gradio as gr

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
    return generate_image(img_file, 25, 3, 23, 2, "average")


def average_embeddings(embeddings, method="average"):
    """
    Averages embeddings based on the specified method.
    """
    if method == "average":
        # Straight Average (as previously implemented)
        return torch.mean(torch.stack(embeddings), dim=0)
    elif method == "median":
        # Median of Embeddings
        return torch.median(torch.stack(embeddings), dim=0).values
    elif method == "max_pooling":
        # Max Pooling
        return torch.max(torch.stack(embeddings), dim=0).values
    elif method == "min_pooling":
        # Min Pooling
        return torch.min(torch.stack(embeddings), dim=0).values
    else:
        raise ValueError("Unsupported averaging method.")
    
    return None  # Fallback

def remove_outliers(embeddings, n_outliers):
    """
    Removes the n embeddings farthest from the centroid of the embeddings.
    
    Args:
        embeddings (list of torch.Tensor): The embeddings from which to remove outliers.
        n_outliers (int): The number of outliers to remove.
    
    Returns:
        list of torch.Tensor: The embeddings with outliers removed.
    """
    if n_outliers == 0 or n_outliers >= len(embeddings):
        # No outliers to remove, or trying to remove too many
        return embeddings

    # Calculate the centroid of the embeddings
    centroid = torch.mean(torch.stack(embeddings), dim=0)
    
    # Calculate the distance of each embedding from the centroid
    distances = [torch.norm(e - centroid, p=2).item() for e in embeddings]
    
    # Identify the indices of the n farthest embeddings
    outlier_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)[:n_outliers]
    
    # Remove the outliers
    embeddings = [e for i, e in enumerate(embeddings) if i not in outlier_indices]
    
    return embeddings

def generate_image(image_paths, num_steps, guidance_scale, seed, num_images, average_method, n_outliers, negative_prompt, progress=gr.Progress(track_tqdm=True)):
    all_embeddings = []
    for image_data in image_paths:
        if image_data is None:
            continue
        
        image_path_or_data = image_data[0]

        # Open the image using PIL and ensure it is in RGB format
        img = Image.open(image_path_or_data).convert('RGB')
        img = np.array(img)[:, :, ::-1]  # Convert to BGR format if necessary for your model

        faces = app.get(img)
        
        if len(faces) > 0:
            embeddings = [torch.tensor(face['embedding'], dtype=dtype).to(device) for face in faces]
            all_embeddings.extend(embeddings)


    if len(all_embeddings) == 0:
        raise gr.Error("No faces detected in the uploaded images. Please upload different images.")

    all_embeddings = remove_outliers(all_embeddings, n_outliers)
    avg_embedding = average_embeddings(all_embeddings, method=average_method)
    
    # Normalize the averaged embedding
    avg_embedding = avg_embedding / torch.norm(avg_embedding, dim=0, keepdim=True)
    avg_embedding = avg_embedding.unsqueeze(0)  # Ensure it has the batch dimension
    id_emb = project_face_embs(pipeline, avg_embedding)
                    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("Start inference...")        
    images = pipeline(
        prompt_embeds=id_emb,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale, 
        num_images_per_prompt=num_images,
        generator=generator,
        negative_prompt=negative_prompt
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
            # img_file = gr.Image(label="Upload a photo with a face", type="filepath")
            img_files = gr.Gallery(label="Upload photos with faces", show_label=True)

            
            submit = gr.Button("Submit", variant="primary")
            
            with gr.Accordion(open=False, label="Advanced Options"):
                num_steps = gr.Slider( 
                    label="Number of sample steps",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=30,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=2.7,
                )
                num_images = gr.Slider(
                    label="Number of output images",
                    minimum=1,
                    maximum=6,
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

                average_method = gr.Radio(
                    label="Embedding Averaging Method",
                    choices=["average", "median", "max_pooling", "min_pooling"],
                    value="average",
                )

                n_outliers = gr.Slider(
                    label="Number of outliers to remove",
                    minimum=0,
                    maximum=10,  
                    step=1,
                    value=0,
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="ugly, deformed, crossed eyes, sunglasses", # Default value or leave empty
                    placeholder="Enter negative prompts separated by commas",
                )

    
   

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
            inputs=[img_files, num_steps, guidance_scale, seed, num_images, average_method, n_outliers, negative_prompt],
            outputs=[gallery]
        )       
    
    
    # gr.Examples(
    #     examples=get_example(),
    #     inputs=[img_files],
    #     run_on_click=True,
    #     fn=run_example,
    #     outputs=[gallery],
    # )
    
    gr.Markdown(Footer)

demo.launch(share=True)
