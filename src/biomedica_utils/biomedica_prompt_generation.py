from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import logging

import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class BIOMEDICARequestGenerator:
    def __init__(self, dataset_path: str, is_eval: bool = False):
        """
        Initialize the request generator with dataset path.

        Args:
            dataset_path: Path to the dataset directory containing 'images' and 'metadata' folders
            is_eval: Whether the dataset is for evaluation or not. If eval, another prompt will be used.
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.metadata_path = self.dataset_path / "metadata"
        self.is_eval = is_eval

    def load_metadata(self, json_path: str) -> Dict[str, Any]:
        """
        Load and parse a single metadata JSON file.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_qa_generation_prompt(self, caption: str, image_context: str) -> str:
        """
        Prepares a prompt for generating QA pairs from BIOMEDICA dataset entries.

        Args:
            caption: The image caption from the dataset
            image_context: The specific image context/inline mention from the paper

        Returns:
            str: Formatted prompt for QA generation
        """
        prompt = (
            f"***** INSTRUCTIONS *****\n"
            f"You are an AI assistant tasked with generating question-answer pairs about a medical image from research papers.\n"
            f"Generate three question-answer pairs about the visible characteristics shown in the image.\n"
            f"While figure caption and context are provided for understanding the image type and medical context, \n"
            f"your questions and answers must focus on describing what can be directly observed in the image.\n\n"
            f"Note: The image context provided below contains all mentions and references to this image throughout the research paper concatenated together.\n"
            f"While this may contain duplicate or overlapping information, it provides the complete context of how the image was discussed in the paper.\n\n"
            f"***** CONTEXT *****\n"
            f"Figure Caption: {caption}\n"
            f"Image Context: {image_context}\n\n"
            f"***** RESPONSE FORMAT *****\n"
            f"{{\n"
            f'  "qa_pairs": [\n'
            f"    {{\n"
            f'      "question_1": "<first_question>",\n'
            f'      "answer_1": "<first_answer>"\n'
            f"    }},\n"
            f"    {{\n"
            f'      "question_2": "<second_question>",\n'
            f'      "answer_2": "<second_answer>"\n'
            f"    }},\n"
            f"    {{\n"
            f'      "question_3": "<third_question>",\n'
            f'      "answer_3": "<third_answer>"\n'
            f"    }}\n"
            f"  ]\n"
            f"}}"
        )

        return prompt

    def prepare_eval_qa_generation_prompt(
        self, caption: str, image_context: str
    ) -> str:
        """
        Prepares a specialized prompt for generating a single difficult multiple-choice
        question-answer pair specifically for dermatology image evaluation benchmarks.

        Args:
            caption: The image caption from the dataset
            image_context: The specific image context/inline mention from the paper

        Returns:
            str: Formatted prompt for difficult dermatology multiple-choice QA generation
        """
        prompt = (
            f"***** INSTRUCTIONS *****\n"
            f"You are an AI assistant tasked with generating a challenging multiple-choice evaluation question about a dermatological image from a research paper.\n"
            f"Generate ONE very difficult multiple-choice question with four options (A, B, C, D) that requires detailed understanding of dermatological features, patterns, or diagnostic elements visible in the image.\n"
            f"The question should be challenging enough to serve as a benchmark for evaluating advanced medical image understanding capabilities.\n\n"
            f"Your question should:\n"
            f"- Focus on specific visual characteristics that require expertise to identify\n"
            f"- Test the ability to recognize subtle diagnostic features\n"
            f"- Include plausible distractors that test fine-grained understanding\n"
            f"- Potentially require integration of visual elements with dermatological knowledge\n\n"
            f"Note: The image context provided below contains all mentions and references to this image throughout the research paper concatenated together.\n"
            f"While this may contain duplicate or overlapping information, it provides the complete context of how the image was discussed in the paper.\n\n"
            f"***** CONTEXT *****\n"
            f"Figure Caption: {caption}\n"
            f"Image Context: {image_context}\n\n"
            f"***** RESPONSE FORMAT *****\n"
            f"{{\n"
            f'  "qa_pair": {{\n'
            f'    "question": "<difficult_dermatology_question>",\n'
            f'    "options": {{\n'
            f'      "A": "<option_A>",\n'
            f'      "B": "<option_B>",\n'
            f'      "C": "<option_C>",\n'
            f'      "D": "<option_D>"\n'
            f"    }},\n"
            f'    "correct_answer": "<A, B, C, or D>"\n'
            f"  }}\n"
            f"}}"
        )

        return prompt

    def prepare_eval_qa_generation_prompt_open_ended(
        self, caption: str, image_context: str
    ) -> str:
        """
        Prepares a specialized prompt for generating a single difficult question-answer pair
        specifically for dermatology image evaluation benchmarks.
        Question should be open-ended and require detailed understanding of dermatological features.

        Args:
            caption: The image caption from the dataset
            image_context: The specific image context/inline mention from the paper

        Returns:
            str: Formatted prompt for difficult dermatology QA generation
        """
        prompt = (
            f"***** INSTRUCTIONS *****\n"
            f"You are an AI assistant tasked with generating a challenging evaluation question about a dermatological image from a research paper.\n"
            f"Generate ONE very difficult question-answer pair that requires detailed understanding of dermatological features, patterns, or diagnostic elements visible in the image.\n"
            f"The question should be challenging enough to serve as a benchmark for evaluating advanced medical image understanding capabilities.\n\n"
            f"Your question should:\n"
            f"- Focus on specific visual characteristics that require expertise to identify\n"
            f"- Test the ability to recognize subtle diagnostic features\n"
            f"- Potentially require integration of visual elements with dermatological knowledge\n\n"
            f"Note: The image context provided below contains all mentions and references to this image throughout the research paper concatenated together.\n"
            f"While this may contain duplicate or overlapping information, it provides the complete context of how the image was discussed in the paper.\n\n"
            f"***** CONTEXT *****\n"
            f"Figure Caption: {caption}\n"
            f"Image Context: {image_context}\n\n"
            f"***** RESPONSE FORMAT *****\n"
            f"{{\n"
            f'  "qa_pair": {{\n'
            f'    "question": "<difficult_dermatology_question>",\n'
            f'    "answer": "<detailed_answer>"\n'
            f"  }}\n"
            f"}}"
        )

        return prompt

    def create_request(
        self, metadata: Dict[str, Any], image_file_name: str
    ) -> Dict[str, Any]:
        """
        Create an API request for a single image.
        """
        # Extract caption and context from metadata
        caption = metadata.get("caption", "")
        image_contexts = metadata.get("metadata").get("image_context")
        image_hash = metadata.get("metadata").get("image_hash")

        image_file_name = image_file_name.replace(".jpg", "")

        # Get specific context for this image
        context = [" ".join(v) for k, v in image_contexts.items()]

        # Combine all contexts into a single string
        context = "\n".join(context)

        # Create the prompt based on whether this is for evaluation or not
        if self.is_eval:
            prompt = self.prepare_eval_qa_generation_prompt(caption, context)
        else:
            prompt = self.prepare_qa_generation_prompt(caption, context)

        # Create the full request object
        request = {
            "image_id": image_file_name,
            "image_path": str(self.images_path / f"{image_hash}.jpg"),
            "metadata_path": str(self.metadata_path / f"{image_hash}.json"),
            "prompt": prompt,
            "caption": caption,
            "context": context,
            "image_primary_label": metadata.get("metadata").get("image_primary_label"),
            "image_secondary_label": metadata.get("metadata").get(
                "image_secondary_label"
            ),
            "is_eval": self.is_eval,
        }

        return request

    def generate_all_requests(self, output_path: str):
        """
        Generate API requests for all images in the dataset and save to a single JSON file.

        Args:
            output_path: Path where to save the generated requests JSON file
        """
        all_requests = []

        # Get list of all JSON files
        json_files = list(self.metadata_path.glob("*.json"))

        # Process each file
        for json_file in tqdm(json_files, desc="Generating requests"):
            # Load metadata
            metadata = self.load_metadata(str(json_file))

            image_file_name = metadata.get("metadata").get("image_file_name")

            # Create request
            request = self.create_request(metadata, image_file_name)

            # Add to list
            all_requests.append(request)

        # Save all requests to a single JSON file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"total_requests": len(all_requests), "requests": all_requests},
                f,
                indent=2,
            )

        print(f"Generated {len(all_requests)} requests and saved to {output_path}")
        return all_requests
