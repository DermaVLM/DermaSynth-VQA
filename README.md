# DermaSynth-VQA: Medical Image VQA Dataset Generator

A comprehensive tool for generating Visual Question Answering (VQA) datasets for medical images, in this study, we generated VQA for PMC paper figures and open-access dermatology datasets.

## Overview

This project provides utilities to analyze medical image datasets, generate medically relevant questions and prompts, and create API requests for Large Language Models (Gemini) to generate question-answer pairs for medical images. The primary focus is on skin lesion datasets and dermatological images from PMC.

## Features

- **Dataset Analysis**: Analyze medical image datasets for statistics like label distribution, image sizes, and caption information
- **Prompt Generation**: Generate diverse medical prompts for use with LLMs
- **API Request Generation**: Create structured API requests for Gemini models
- **Knowledge Base Integration**: Vector store for medical knowledge with RAG capabilities (not used for this study)
- **Multi-threaded API Call Handling**: Efficient processing with thread-safe API key management

## Project Structure

```
├── config/
│   └── prompt_samples.json     # Sample prompts (for open-access datasets) for answer generation
├── src/
│   ├── __init__.py
│   ├── gemini_api.py           # Handler for Gemini API calls
│   ├── prompt_generation.py    # Prompt generation utilities
│   ├── biomedica_utils/
│   │   ├── __init__.py
│   │   ├── biomedica_analyzer.py        # Dataset analysis tools
│   │   └── biomedica_prompt_generation.py  # BIOMEDICA-specific prompt generation
│   └── knowledge_base/
│       ├── section_mapper.py
│       ├── vector_store.py     # Vector storage for medical knowledge
│       └── wikipedia_api_scraper.py
├── generate_api_inputs.py      # Script to generate API requests for models
├── generate_VQA.py             # Main script to generate VQA pairs
└── README.md
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DermaSynth-VQA.git
cd DermaSynth-VQA
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your Gemini API key(s)
GEMINI_API_KEY=your_api_key_here
```

## Usage

First we generate prompt in a json file, then use the json file for generation with Gemini API.

### Generating API Requests

```python
from src import BIOMEDICARequestGenerator

# Initialize generator (for eval type prompt)
generator = BIOMEDICARequestGenerator("datasets/biomedica_eval", is_eval=True)

# Generate and save all requests
requests = generator.generate_all_requests(
    output_path="api_requests/api_requests_biomedica_eval.json"
)

# Print statistics
print(f"Total requests generated: {len(requests)}")
```

### Generating VQA Pairs

```python
# Configure the environment variables first (API keys)
from dotenv import load_dotenv
load_dotenv()

# Use the generate_VQA.py script
python generate_VQA.py
```

## Prompt Customization

The system uses various prompt templates stored in `config/prompt_samples.json` for different question types:

- Observational Overview
- Diagnostic Questions
- Clinical Assessment
- Differential Diagnosis
- Treatment Options
- And many more...

You can customize these prompts by editing the JSON file.
