import os
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd


class BIOMEDICAAnalyzer:
    def __init__(self, dataset_path: str):
        """
        Initialize the analyzer with the dataset path.

        Args:
            dataset_path: Path to the dataset directory containing 'images' and 'metadata' folders
        """
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, "images")
        self.metadata_path = os.path.join(dataset_path, "metadata")
        self.df = None

    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame from all JSON files in the metadata directory.
        """
        data_list = []

        # Get all JSON files
        json_files = [f for f in os.listdir(self.metadata_path) if f.endswith(".json")]

        for json_file in json_files:
            with open(os.path.join(self.metadata_path, json_file), "r") as f:
                data = json.load(f)

                # Flatten the nested structure
                flat_dict = {
                    "image_file": os.path.basename(data.get("image_path", "")),
                    "caption": data.get("caption", ""),
                }

                # Add metadata fields
                metadata = data.get("metadata", {})
                for key, value in metadata.items():
                    flat_dict[key] = value

                data_list.append(flat_dict)

        # Create DataFrame
        self.df = pd.DataFrame(data_list)
        return self.df

    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset.
        """
        if self.df is None:
            self.create_dataframe()

        # Helper function to get unique values from list columns
        def get_unique_from_lists(series):
            # Flatten the lists and get unique values
            unique_values = set()
            for item in series:
                if isinstance(item, list):
                    unique_values.update(item)
                else:
                    unique_values.add(item)
            return list(unique_values)

        # Helper function to count values in list columns
        def get_value_counts_from_lists(series):
            counter = Counter()
            for item in series:
                if isinstance(item, list):
                    counter.update(item)
                else:
                    counter[item] += 1
            return dict(counter)

        stats = {
            "total_samples": len(self.df),
            "unique_primary_labels": get_unique_from_lists(
                self.df["image_primary_label"]
            ),
            "unique_secondary_labels": get_unique_from_lists(
                self.df["image_secondary_label"]
            ),
            "panel_types": get_value_counts_from_lists(self.df["image_panel_type"]),
            "panel_subtypes": get_value_counts_from_lists(
                self.df["image_panel_subtype"]
            ),
        }

        # Add counts to the stats
        stats["num_unique_primary_labels"] = len(stats["unique_primary_labels"])
        stats["num_unique_secondary_labels"] = len(stats["unique_secondary_labels"])

        return stats

    def analyze_captions(self) -> Dict[str, Any]:
        """
        Analyze the image captions.
        """
        if self.df is None:
            self.create_dataframe()

        caption_stats = {
            "avg_caption_length": self.df["caption"].str.len().mean(),
            "min_caption_length": self.df["caption"].str.len().min(),
            "max_caption_length": self.df["caption"].str.len().max(),
        }

        return caption_stats

    def plot_label_distribution(
        self, label_column: str = "image_primary_label", top_n: int = 10
    ) -> None:
        """
        Plot distribution of labels, handling list-type values.

        Args:
            label_column: Column name to analyze
            top_n: Number of top categories to show
        """
        if self.df is None:
            self.create_dataframe()

        # Count occurrences of each label
        counter = Counter()
        for items in self.df[label_column]:
            if isinstance(items, list):
                counter.update(items)
            else:
                counter[items] += 1

        # Get top N items
        top_items = dict(counter.most_common(top_n))

        plt.figure(figsize=(12, 6))
        plt.barh(list(top_items.keys()), list(top_items.values()))
        plt.title(f"Top {top_n} {label_column} Distribution")
        plt.xlabel("Count")
        plt.ylabel(label_column)
        plt.tight_layout()
        plt.show()

        return counter

    def analyze_image_sizes(self) -> Dict[str, Any]:
        """
        Analyze the distribution of image sizes.
        """
        if self.df is None:
            self.create_dataframe()

        # Convert string representation of image size to tuple
        sizes = self.df["image_size"]
        widths = sizes.apply(lambda x: x[0])
        heights = sizes.apply(lambda x: x[1])

        size_stats = {
            "avg_width": widths.mean(),
            "avg_height": heights.mean(),
            "min_width": widths.min(),
            "min_height": heights.min(),
            "max_width": widths.max(),
            "max_height": heights.max(),
            "aspect_ratios": (widths / heights).describe().to_dict(),
        }

        return size_stats
