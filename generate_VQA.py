# %%
import os
import json
from pathlib import Path
import logging
import time
import random
import threading
from queue import Queue

from PIL import Image
from dotenv import load_dotenv

from src import GeminiAPIHandler


# %%
class ThreadSafeCounter:
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

    def get(self):
        with self.lock:
            return self.value


class APIKeyManager:
    def __init__(self, api_keys: list):
        self.api_keys = api_keys
        self.api_key = api_keys[0]  # Get the first key TEMPorarily
        self.key_index = 0
        self.lock = threading.Lock()
        # self.model_name = "gemini-2.5-flash-preview-04-17"
        self.model_name = "gemini-2.0-flash"

    @property
    def model_name(self):
        with self.lock:
            return self._model_name

    @model_name.setter
    def model_name(self, value):
        with self.lock:
            self._model_name = value

    def get_next_api_key(self):
        with self.lock:
            key = self.api_keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(self.api_keys)
            return key

    def create_api_handler(self):
        key = self.api_key
        return GeminiAPIHandler(api_key=key, model_name=self.model_name)


def setup_logging(output_dir):
    log_filename = os.path.join(
        output_dir, f"-api_results_multithreaded_{time.time()}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
        filename=log_filename,
    )
    # Also add a console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return logging.getLogger(__name__)


def call_gemini_api(api_handler, request):
    """
    Call the Gemini API for a given request and dataset image path.
    """
    image_id = request.get("image_id")
    image_path = request.get("image_path")
    prompt = request.get("prompt")
    image_primary_label = request.get("image_primary_label")
    image_secondary_label = request.get("image_secondary_label")

    # Load the image
    with Image.open(image_path) as pil_image:
        api_response = api_handler.generate_from_pil_image(pil_image, prompt)

    # Save the responses
    result = {
        "image_id": image_id,
        "image_path": image_path,
        "image_primary_label": image_primary_label,
        "image_secondary_label": image_secondary_label,
        "api_response": api_response,
        "prompt": prompt,
        "model_name": api_handler.model_name,
    }

    return result


def worker_task(
    worker_id, request_queue, key_manager, output_dir, logger, completed_counter
):
    # Create a thread-specific API handler
    api_handler = key_manager.create_api_handler()
    logger.info(f"Worker {worker_id} started with API key: {api_handler.api_key}")

    possible_errors = [
        "429 Quota exceeded for quota metric",
        "429 Resource has been exhausted",
        "429 RESOURCE_EXHAUSTED",
    ]
    consecutive_errors = 0
    max_consecutive_errors = 100

    while True:
        try:
            # Get a request from the queue with a timeout
            request = request_queue.get(timeout=5)
        except Exception:
            # If the queue is empty for too long, exit the thread
            logger.info(f"Worker {worker_id} exiting - no more tasks in queue")
            break

        # Check if this is a signal to stop
        if request is None:
            logger.info(f"Worker {worker_id} received stop signal")
            request_queue.task_done()
            break

        try:
            image_id = os.path.basename(request.get("image_path")).replace(".jpg", "")
            generated_response_file = Path(output_dir) / f"{image_id}_response.json"

            # Skip if already processed
            if os.path.exists(generated_response_file):
                logger.info(
                    f"Worker {worker_id} skipping image_id {image_id} - response already exists"
                )
                request_queue.task_done()
                continue

            # Process the request
            result = call_gemini_api(api_handler, request)

            # Save result to file
            with open(generated_response_file, "w") as out_file:
                json.dump(result, out_file, indent=4)

            completed = completed_counter.increment()
            logger.info(
                f"Worker {worker_id} processed image_id {image_id} - {completed} completed"
            )
            consecutive_errors = 0

        except Exception as e:
            error_msg = str(e)
            logger.warning(
                f"Worker {worker_id} error processing image_id {image_id}: {error_msg}"
            )

            # If API key quota exceeded, get a new API key
            if any(error in error_msg for error in possible_errors):
                logger.warning(
                    f"Worker {worker_id} API key quota exceeded. Getting new API key."
                )
                api_handler = key_manager.create_api_handler()
                logger.info(
                    f"Worker {worker_id} switched to API key: {api_handler.api_key}"
                )

                # Put the request back in the queue
                request_queue.put(request)
            else:
                # For other errors, increment consecutive errors
                consecutive_errors += 1

                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"Worker {worker_id} stopping after {consecutive_errors} consecutive errors"
                    )
                    request_queue.task_done()
                    break

                # Add the request back to the queue for retry
                request_queue.put(request)

            time.sleep(4 + 4 * random.random())

        finally:
            request_queue.task_done()
            # Add a small random delay between requests to avoid hammering the API
            time.sleep(0.05 + 0.05 * random.random())


# %%

# Configuration
dataset_name = "biomedica_500_750"
api_requests_file = f"api_requests/api_requests_{dataset_name}.json"
assert os.path.exists(api_requests_file), f"File {api_requests_file} does not exist"
output_dir = os.path.join("api_results", dataset_name)
os.makedirs(output_dir, exist_ok=True)

# Set up logging
logger = setup_logging(output_dir)
logger.info("Starting multithreaded API processing")

# Load environment variables and API keys
load_dotenv()
api_keys = os.getenv("GEMINI_API_KEY").split(",")
logger.info(f"Loaded {len(api_keys)} API keys")
# %%
# Load API requests
with open(api_requests_file, "r") as file:
    api_requests_data = json.load(file)

total_requests = api_requests_data["total_requests"]
api_requests = api_requests_data["requests"]
logger.info(f"Loaded {total_requests} API requests")

# Shuffle the requests for better load balancing
random.seed(time.time())
random.shuffle(api_requests)

# %% Create a thread-safe queue with the requests
no_of_skipped = 0
request_queue = Queue()
for req in api_requests:
    image_id = os.path.basename(req.get("image_path")).replace(".jpg", "")
    generated_response_file = Path(output_dir) / f"{image_id}_response.json"
    if not os.path.exists(generated_response_file):
        request_queue.put(req)

    else:
        no_of_skipped += 1

logger.info(f"Skipped {no_of_skipped} requests that were already processed")

# %%
# Initialize the API key manager and completed counter
key_manager = APIKeyManager(api_keys)
completed_counter = ThreadSafeCounter()

# Determine number of worker threads
num_workers = 3
logger.info(f"Starting {num_workers} worker threads")

# Start worker threads
threads = []
for i in range(num_workers):
    thread = threading.Thread(
        target=worker_task,
        args=(i, request_queue, key_manager, output_dir, logger, completed_counter),
        name=f"Worker-{i}",
    )
    thread.daemon = True
    thread.start()
    threads.append(thread)

try:
    # Wait for all tasks to be processed
    request_queue.join()
    logger.info("All requests have been processed")
except KeyboardInterrupt:
    logger.info("Keyboard interrupt detected, stopping workers")
finally:
    # Signal all threads to stop
    for _ in range(num_workers):
        request_queue.put(None)

    # Wait for threads to finish
    for thread in threads:
        thread.join(timeout=2)

    logger.info(f"Processing completed. Processed {completed_counter.get()} requests.")

# %%
