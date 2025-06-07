from pathlib import Path
import base64
import io
from PIL import Image

class Utilities:
    def __init__(self):
        pass

    def convert_image_to_bytes(self, image_path: Path) -> bytes:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return image_bytes
    
    def base64_to_image(self, base64_string: str) -> Image.Image:
        """
        Converts a base64 string back to a PIL Image.

        Args:
            base64_string (str): The base64 encoded image string.

        Returns:
            Image.Image: A PIL Image object.
        """
        image_bytes = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_bytes))