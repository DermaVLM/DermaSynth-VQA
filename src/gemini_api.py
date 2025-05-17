import logging

import base64
from google import genai
from google.genai import types

LOGGER = logging.getLogger(__name__)


class GeminiAPIHandler:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        """
        Initialize the Gemini API Client.

        Parameters:
        - api_key: str - Gemini API key.
        - model_name: str - The model to use (default: "gemini-2.0-flash").
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

        # Configure default safety settings to minimize blocks
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
        ]

    def reconfigure_api(self, api_key, model_name="gemini-2.0-flash"):
        """
        Reconfigure the API client with new credentials.

        Parameters:
        - api_key: str - Gemini API key.
        - model_name: str - The model to use (default: "gemini-2.0-flash").
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        LOGGER.info("Reconfigured Gemini API client.")

    def generate_text(self, prompt):
        """
        Generate text content from a prompt.

        Parameters:
        - prompt: str - The text prompt to send to the model.

        Returns:
        - str - The generated text.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.safety_settings
                ),
            )
            return response.text
        except Exception as e:
            LOGGER.error(f"Error generating text: {e}")
            return None

    def generate_from_image(self, image_path, prompt):
        """
        Generate content based on an image and a text prompt.

        Parameters:
        - image_path (str): Path to the image file.
        - prompt (str): The text prompt to guide the generation.

        Returns:
        - str: The generated response text.
        """
        with open(image_path, "rb") as img_file:
            image_content = img_file.read()

        image_data = {
            "mime_type": "image/jpg",
            "data": base64.b64encode(image_content).decode("utf-8"),
        }

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image_data, prompt],
            config=types.GenerateContentConfig(safety_settings=self.safety_settings),
        )
        return response.text

    def generate_from_pil_image(self, pil_image, prompt):
        """
        Generate content based on a single PIL Image object and a text prompt.

        Parameters:
        - pil_image (PIL.Image.Image): A PIL Image object.
        - prompt (str): The text prompt to guide the generation.

        Returns:
        - str: The generated response text.
        """
        # Ensure the image is in RGB mode (convert from RGBA if necessary)
        if pil_image.mode == "RGBA":
            LOGGER.warning(
                "Image is in RGBA mode, converting to RGB for JPEG compatibility."
            )
            pil_image = pil_image.convert("RGB")
        elif pil_image.mode != "RGB":
            LOGGER.warning(
                f"Image is in {pil_image.mode} mode, converting to RGB for compatibility."
            )
            pil_image = pil_image.convert("RGB")

        # Api handles image data directly
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, pil_image],
            config=types.GenerateContentConfig(safety_settings=self.safety_settings),
        )

        return response.text


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    gemini = GeminiAPIHandler(api_key=os.getenv("GEMINI_API_KEY"))

    text_prompt = "Do you know da wae?"
    print("Text Response:", gemini.generate_text(text_prompt))
