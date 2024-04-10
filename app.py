import sys
sys.path.append('./')

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting
)

from diffusers.utils import load_image

from arc2face import CLIPTextModelWrapper, project_face_embs

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import random
import os
import datetime

import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import csv

def save_embeddings_to_csv(embeddings, filename):
    """
    Saves embeddings to a CSV file.

    Args:
        embeddings (torch.Tensor): The embeddings tensor to save.
        filename (str): The filename for the CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Convert each embedding to a list and write to the CSV file
        for embedding in embeddings:
            writer.writerow(embedding.tolist())


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


pipeline_img2img = AutoPipelineForImage2Image.from_pretrained( 
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None
)
pipeline_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipeline_img2img.scheduler.config)
pipeline_img2img.to(device)

pipeline_inpainting = AutoPipelineForInpainting.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None
)
pipeline_inpainting.scheduler = DPMSolverMultistepScheduler.from_config(pipeline_inpainting.scheduler.config)
pipeline_inpainting.to(device)

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

     # Stack the embeddings into a single tensor
    embeddings_stack = torch.stack(embeddings)

    # log number of faces being averaged
    logging.info(f"Number of faces being averaged: {len(embeddings)}")

    # Generate a timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    embeddings_filename = f"./embeddings_{timestamp}_in.csv"
    averaged_filename = f"./embeddings_{timestamp}_out.csv"
    
    # Save all incoming face embeddings
    save_embeddings_to_csv(embeddings_stack, embeddings_filename)
    logging.info(f"All embeddings saved to {embeddings_filename}")

    if method == "average":
        # Straight Average (as previously implemented)
        averaged_embedding = torch.mean(embeddings_stack, dim=0)
    elif method == "median":
        # Median of Embeddings
        averaged_embedding = torch.median(embeddings_stack, dim=0).values
    elif method == "trimmed_mean":
        # Calculate the trimmed mean for each feature
        lower_bound, upper_bound = int(0.15 * len(embeddings)), int(0.85 * len(embeddings))
        sorted_embeddings = embeddings_stack.sort(dim=0).values
        trimmed_embeddings = sorted_embeddings[lower_bound:upper_bound, :]
        averaged_embedding = torch.mean(trimmed_embeddings, dim=0)
    elif method == "max_pooling":
        # Max Pooling
        averaged_embedding = torch.max(embeddings_stack, dim=0).values
    elif method == "min_pooling":
        # Min Pooling
        averaged_embedding = torch.min(embeddings_stack, dim=0).values
    elif method == "rounded_mode":
        # Initialize an empty tensor to hold the mode values for each dimension
        mode_embeddings = torch.zeros_like(embeddings_stack[0])

        # Iterate over each dimension
        for dim in range(embeddings_stack.size(1)):
            # Extract the current dimension across all embeddings
            current_dim_values = embeddings_stack[:, dim]
            # Round the values in the current dimension to 2 decimal places
            rounded_dim_values = torch.round(current_dim_values * 100) / 100
            # Calculate the mode of the rounded values
            values, counts = rounded_dim_values.unique(return_counts=True)
            mode_value = values[counts.argmax()]
            # Assign the mode value to the corresponding dimension in the result tensor
            mode_embeddings[dim] = mode_value

        averaged_embedding = mode_embeddings

    elif method == "rounded_mode_averaging":
        # Initialize an empty tensor to hold the averaged values for each dimension
        averaged_mode_embeddings = torch.zeros_like(embeddings_stack[0])
        
        # Iterate over each dimension
        for dim in range(embeddings_stack.size(1)):
            # Extract the current dimension across all embeddings
            current_dim_values = embeddings_stack[:, dim]
            # Round the values in the current dimension to 2 decimal places
            rounded_dim_values = torch.round(current_dim_values * 100) / 100
            # Calculate the mode of the rounded values
            mode_value, mode_count = rounded_dim_values.mode()
            # Find the indices of embeddings that contributed to the mode in the current dimension
            contributing_indices = (rounded_dim_values == mode_value).nonzero(as_tuple=True)[0]
            # Use these indices to select the original, full-precision values that contributed to the mode
            contributing_values = current_dim_values[contributing_indices]
            # Average these contributing values
            averaged_mode_value = torch.mean(contributing_values)
            # Assign the averaged value to the corresponding dimension in the result tensor
            averaged_mode_embeddings[dim] = averaged_mode_value
        
        averaged_embedding = averaged_mode_embeddings
    elif method == "ensemble_average":
        # Compute individual averages: mean, median, and trimmed mean
        mean_embedding = torch.mean(embeddings_stack, dim=0)
        median_embedding = torch.median(embeddings_stack, dim=0).values
        
        # For trimmed mean, assuming the same 15% trim as before
        lower_bound, upper_bound = int(0.15 * len(embeddings)), int(0.85 * len(embeddings))
        sorted_embeddings = embeddings_stack.sort(dim=0).values
        trimmed_embeddings = sorted_embeddings[lower_bound:upper_bound, :]
        trimmed_mean_embedding = torch.mean(trimmed_embeddings, dim=0)
        
        # Stack the individual averages and then compute the final ensemble average
        all_averages = torch.stack([mean_embedding, median_embedding, trimmed_mean_embedding])
        
        # Choose here if you want the mean or median of the averages
        # For mean of averages:
        averaged_embedding = torch.mean(all_averages, dim=0)
        # For median of averages (uncomment the following line if you prefer median):
        # averaged_embedding = torch.median(all_averages, dim=0).values
    elif method == "ensemble_median":
        # Compute individual averages: mean, median, and trimmed mean
        mean_embedding = torch.mean(embeddings_stack, dim=0)
        median_embedding = torch.median(embeddings_stack, dim=0).values
        
        # For trimmed mean, assuming the same 15% trim as before
        lower_bound, upper_bound = int(0.15 * len(embeddings)), int(0.85 * len(embeddings))
        sorted_embeddings = embeddings_stack.sort(dim=0).values
        trimmed_embeddings = sorted_embeddings[lower_bound:upper_bound, :]
        trimmed_mean_embedding = torch.mean(trimmed_embeddings, dim=0)
        
        # Stack the individual averages and then compute the final ensemble average
        all_averages = torch.stack([mean_embedding, median_embedding, trimmed_mean_embedding])
        
        # Choose here if you want the mean or median of the averages
        # For mean of averages:
        # averaged_embedding = torch.mean(all_averages, dim=0)
        # For median of averages (uncomment the following line if you prefer median):
        averaged_embedding = torch.median(all_averages, dim=0).values
    elif method == "random_sampling":
        # Initialize an empty tensor for the randomly sampled embedding
        randomly_sampled_embedding = torch.empty(embeddings_stack.shape[1], dtype=embeddings_stack.dtype).to(embeddings_stack.device)
        
        # For each dimension, randomly select an embedding and take its value for that dimension
        for dim in range(embeddings_stack.shape[1]):  # Iterate over each dimension
            random_index = torch.randint(0, embeddings_stack.shape[0], (1,)).item()  # Randomly select an embedding index
            randomly_sampled_embedding[dim] = embeddings_stack[random_index, dim]  # Assign the randomly selected value
        
        # Use the randomly sampled embedding as the averaged embedding
        averaged_embedding = randomly_sampled_embedding

    else:
        raise ValueError("Unsupported averaging method.")
    
    # Save the averaged embedding as CSV
    save_embeddings_to_csv(averaged_embedding.unsqueeze(0), averaged_filename)  # Unsqueeze to make it 2D for consistency
    logging.info(f"Averaged embedding saved to {averaged_filename}")
    
    return averaged_embedding

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

def create_face_mask(image): 
    img = Image.open(image).convert('RGB')
    img = np.array(img)[:, :, ::-1]  # Convert to BGR format if necessary for your model
    faces = app.get(img)
    
    if len(faces) > 0:
        # Calculate the area of the bounding box for each face and select the largest face
        largest_face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
        bbox = largest_face['bbox']
        
        # Create a black image with the same size as the input image
        mask = np.zeros_like(img)
        
        # Set the region of the face to white in the mask (note: bbox values need to be integers)
        mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255
        
        # Convert the mask to PIL Image format
        mask = Image.fromarray(mask[:, :, ::-1])  # Convert back to RGB format if necessary
        
        return mask
    else:
        return None



def generate_image(initial_image_path, image_paths, num_steps, guidance_scale, seed, num_images, average_method, n_outliers, negative_prompt, stren, face_only, progress=gr.Progress(track_tqdm=True)):

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
    
    if initial_image_path is not None and face_only:
        initial_image = load_image(initial_image_path)
        mask = create_face_mask(initial_image_path)

        if initial_image is None:
            raise ValueError("Failed to load initial image. Please check the image path and format.")

        if mask is None:
            raise ValueError("Failed to create face mask. Please check the image path and format.")

        logging.info(f"Initial image loaded successfully with size {initial_image.size}")

        initial_image = [initial_image] * num_images

        images = pipeline_inpainting(
            strength=(stren / 100),
            prompt_embeds=id_emb,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale, 
            num_images_per_prompt=num_images,
            generator=generator,
            image=initial_image,
            mask_image=mask,
            negative_prompt=negative_prompt,
            padding_mask_crop=32
        ).images
    elif initial_image_path is not None:
        initial_image = load_image(initial_image_path)
        
        if initial_image is None:
            raise ValueError("Failed to load initial image. Please check the image path and format.")

        logging.info(f"Initial image loaded successfully with size {initial_image.size}")

        initial_image = [initial_image] * num_images

        images = pipeline_img2img(
            strength=(stren / 100),
            prompt_embeds=id_emb,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale, 
            num_images_per_prompt=num_images,
            generator=generator,
            image=initial_image,
            negative_prompt=negative_prompt
        ).images
    else:
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

            img2img_file = gr.Image(label="Upload a photo for initial image (optional for img2img)", type="filepath")

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
                    choices=["average", "median", "trimmed_mean", "ensemble_average", "ensemble_median", "max_pooling", "min_pooling", "rounded_mode", "rounded_mode_averaging", "random_sampling"],
                    value="median",
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

                stren = gr.Slider(
                    label="Strength for img2img (only used if img2img file is present)",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=50,
                )

                face_only = gr.Checkbox(label="Face Only Inpainting", value=False)

    
   

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
            inputs=[img2img_file, img_files, num_steps, guidance_scale, seed, num_images, average_method, n_outliers, negative_prompt, stren, face_only],
            outputs=[gallery]
        )       
    
    
    gr.Markdown(Footer)

demo.launch(share=True)
