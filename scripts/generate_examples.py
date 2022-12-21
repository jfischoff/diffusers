from diffusers import StableDiffusionPipeline
import torch

model_id = "/home/jonathan/stable-diffusion-webui/models/dreambooth/jonathan-relight-1/working"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# set the scheduler to the better one
pipe = pipe.to("cuda")

# Open the file in read mode
with open('scripts/artist_list_fixup.txt', 'r') as file:
  # Initialize an empty string
  
  # Iterate through the lines of the file
  for line in file:

    prompt = "Portrait of byct by " + line
    print("Prompt: " + prompt")
    image = pipe(prompt).images[0]  
    
    # Make the string all lower case
    outputFileName = prompt.lower()

    # Replace spaces with '-'
    outputFileName = outputFileName.replace(" ", "-")

    image.save("scripts/byct_images/" + outputFileName + ".jpg")

# Open the file in read mode
with open('scripts/aestitics_list.txt', 'r') as file:
  # Initialize an empty string
  
  # Iterate through the lines of the file
  for line in file:

    line = line.strip()

    prompt = "Photo of a " + line + " byct"
    print("Prompt: " + prompt")
    image = pipe(prompt).images[0]  
    
    # Make the string all lower case
    outputFileName = prompt.lower()

    # Replace spaces with '-'
    outputFileName = outputFileName.replace(" ", "-")

    image.save("scripts/byct_aestitics/" + line + ".jpg" )
