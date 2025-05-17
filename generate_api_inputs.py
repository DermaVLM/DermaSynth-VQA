# %%
import os
import json

import pandas as pd

from src import BIOMEDICAAnalyzer, BIOMEDICARequestGenerator


pd.set_option("display.max_columns", 100)

# %%
start = 500
end = 750
dataset_path = f"datasets/biomedica_clinical_samples_{start}_{end}"

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

# %% Initialize analyzer
analyzer = BIOMEDICAAnalyzer(dataset_path)

# Create DataFrame
df = analyzer.create_dataframe()
print("DataFrame shape:", df.shape)

stats = analyzer.get_basic_stats()
print("\nBasic stats:", json.dumps(stats, indent=2))

caption_stats = analyzer.analyze_captions()
print("\nCaption statistics:", caption_stats)

analyzer.plot_label_distribution(label_column="image_primary_label")

analyzer.plot_label_distribution(label_column="image_secondary_label")

size_stats = analyzer.analyze_image_sizes()
print("\nImage size statistics:", size_stats)

# %%
# Initialize generator
generator = BIOMEDICARequestGenerator(dataset_path, is_eval=False)

# Generate and save all requests
output_path = f"api_requests/api_requests_biomedica_{start}_{end}.json"
requests = generator.generate_all_requests(output_path=output_path)

# Print some statistics
print(f"Total requests generated: {len(requests)}")
print(f"First request preview:")
print(json.dumps(requests[0], indent=2))

# %%
