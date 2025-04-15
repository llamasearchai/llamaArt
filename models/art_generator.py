from diffusers import StableDiffusionPipeline


def generate_art(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
    image = pipe(prompt).images[0]
    return image
