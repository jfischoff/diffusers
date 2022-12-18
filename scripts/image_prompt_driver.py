from diffusers import StableDiffusionImageVariationTextPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
import PIL
import torch

model_id = "/home/jonathan/models/sd-image-variations-diffusers"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionImageVariationTextPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# Set the scheduler to Euler 
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

# open the image /home/jonathan/dreambooth/byct/byct-(48).jpg and run the pipeline on it


with PIL.Image.open("/home/jonathan/Downloads/example1.jpg") as initialImage:
    # run the pipeline on the image and produce a batch of 50 images and set the steps to 1000
    output = pipe(image=initialImage, num_images_per_prompt=4, num_inference_steps=50)
    # save all the images to the current directory
    for i, image in enumerate(output.images):
        image.save(f"image-{i}.png")


    