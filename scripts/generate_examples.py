import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = "/home/jonathan/dreambooth/ryct-working/output"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# set the scheduler to the better one
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda")

batchSize = 1

pipe = pipe.to("cuda")

# Open the file in read mode
with open('scripts/artist_list_fixup.txt', 'r') as file:
  # Initialize an empty string
  
  # Iterate through the lines of the file
  for line in file:

    prompt = "Portrait of ryct woman by " + line
    print("Prompt: " + prompt)
    image = pipe(     
        prompt=prompt,
        num_images_per_prompt=batchSize, 
        num_inference_steps=50,
        guidance_scale=7.0).images[0]  
    
    # Make the string all lower case
    outputFileName = prompt.lower()

    # Replace spaces with '-'
    outputFileName = outputFileName.replace(" ", "-")

    directory_name = 'scripts/ryct_artist_images/'
    os.makedirs(directory_name, exist_ok=True)
    image.save(directory_name + outputFileName + ".jpg")

# Open the file in read mode
with open('scripts/aestitics_list.txt', 'r') as file:
  # Initialize an empty string
  
  # Iterate through the lines of the file
  for line in file:

    line = line.strip()

    prompt = "Photo of a " + line + " ryct"
    print("Prompt: " + prompt)
    image = pipe(prompt).images[0]  
    
    # Make the string all lower case
    outputFileName = prompt.lower()

    # Replace spaces with '-'
    outputFileName = outputFileName.replace(" ", "-")
    directory_name = 'scripts/ryct_aestitics/'
    os.makedirs(directory_name, exist_ok=True)
    image.save(directory_name + line + ".jpg" )
