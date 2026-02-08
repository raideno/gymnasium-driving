import io
from io import BytesIO

from PIL import Image as PILImage
from IPython.display import display, clear_output

# NOTE: render_mode must be set to rgb_array

def preview(
    environment,
    clear: bool = True,
):
    image = environment.render()
    if clear:
        clear_output(wait=True)
    
    pil_img = PILImage.fromarray(image)
    
    display(pil_img)