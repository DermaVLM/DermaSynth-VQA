import json
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class MedicalSectionMapper:
    def __init__(self):
        # Core medical section mappings
        self.section_groups = {
            "overview": {
                "overview",
                "definition",
                "description",
                "general",
                "about",
                "introduction",
                "background",
                "general description",
            },
            "clinical_presentation": {
                "symptoms",
                "signs",
                "presentation",
                "clinical features",
                "manifestations",
                "clinical presentation",
                "characteristics",
                "clinical manifestations",
                "signs and symptoms",
                "symptomatology",
                "clinical signs",
                "features",
            },
            "diagnosis": {
                "diagnosis",
                "diagnostic",
                "testing",
                "investigations",
                "diagnostic criteria",
                "tests",
                "examination",
                "evaluation",
                "assessment",
                "screening",
            },
            "differential_diagnosis": {
                "differential diagnosis",
            },
            "laboratory_findings": {
                "laboratory",
                "laboratory tests",
                "biochemical tests",
                "blood test",
                "serology",
                "cytogenetics",
                "blood count",
                "blood testing",
                "reference ranges",
            },
            "imaging": {
                "imaging",
                "imaging findings",
                "radiography",
                "radiographs",
                "x-ray",
                "ultrasound",
                "ct scan",
                "mri",
                "thermography",
                "nuclear medicine",
            },
            "treatment": {
                "treatment",
                "therapy",
                "management",
                "therapeutic",
                "medications",
                "interventions",
                "prevention",
                "medication",
                "drugs",
                "clinical management",
                "therapeutic options",
            },
            "surgery": {
                "surgery",
                "surgical",
                "surgical resection",
                "surgical removal",
                "surgical techniques",
                "surgical treatment",
            },
            "pathophysiology": {
                "pathophysiology",
                "mechanism",
                "pathology",
                "pathogenesis",
                "pathogenic",
            },
            "causes_and_risk_factors": {
                "etiology",
                "causes",
                "risk factors",
                "predisposing factors",
                "triggers",
                "triggering factors",
            },
            "epidemiology": {
                "epidemiology",
                "prevalence",
                "incidence",
                "demographics",
                "statistics",
                "frequency",
                "occurrence",
                "distribution",
                "population",
            },
            "complications": {
                "complications",
                "prognosis",
                "outcomes",
                "side effects",
                "adverse effects",
                "sequelae",
                "consequences",
                "risks",
                "adverse reactions",
                "effects",
            },
            "subtypes": {
                "types",
                "classification",
                "variants",
                "forms",
                "subtypes",
                "categories",
                "classes",
                "variations",
            },
            "history": {
                "history",
                "historical",
                "etymology",
            },
        }

        # Create reverse mapping for quick lookups
        self.section_lookup = {}
        for main_category, related_terms in self.section_groups.items():
            for term in related_terms:
                self.section_lookup[term] = main_category

    def _contains_year(self, text: str) -> bool:
        """Helper function to check for year patterns (e.g., "1995", "20th century")."""
        return bool(
            re.search(r"\b(1[89]\d{2}|20\d{2}|1[89]\d{1}th|2[01]\d{1}th)\b", text)
        )

    def _contains_name(self, text: str) -> bool:
        """
        Helper to check for name patterns, very basic, only capitalized words.
        It does not look for specific format for names, we just want something simple
        """
        return bool(re.search(r"\b([A-Z][a-z]+)\b", text))

    def find_best_category(self, section_name: str) -> str:
        """Find the best matching category for a section name"""
        # Clean the section name
        clean_name = section_name.lower().strip()

        # Direct match
        if clean_name in self.section_lookup:
            return self.section_lookup[clean_name]

        # Check for year/century
        if self._contains_year(clean_name):
            return "history"

        # Check for names:
        if self._contains_name(clean_name):
            return "other"

        # Partial match
        for word in clean_name.split():
            if word in self.section_lookup:
                return self.section_lookup[word]

        # Check if section contains any of our known terms
        for term, category in self.section_lookup.items():
            if term in clean_name:
                return category

        return "other"

    def map_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Map sections to standardized categories"""
        mapped_sections = defaultdict(str)

        for section_name, content in sections.items():
            if not content.strip():  # Skip empty sections
                continue

            category = self.find_best_category(section_name)
            if mapped_sections[category]:
                mapped_sections[category] += f"\n\n{content}"
            else:
                mapped_sections[category] = content

        return dict(mapped_sections)


def process_medical_content(input_file: str, output_file: str):
    """Process and standardize medical content from JSON file"""
    mapper = MedicalSectionMapper()

    # Read input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each condition
    processed_data = []
    for entry in data:
        if not entry["sections"]:  # Skip empty entries
            continue

        processed_entry = {
            "name": entry["name"],
            "url": entry["url"],
            "sections": mapper.map_sections(entry["sections"]),
            "categories": entry["categories"],
        }
        processed_data.append(processed_entry)

    # Save processed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import os

    dir_path = "src/knowledge_base"
    input_file = os.path.join(dir_path, "dermatology_knowledge_base.json")
    output_file = os.path.join(dir_path, "processed_dermatology_kb.json")
    process_medical_content(input_file=input_file, output_file=output_file)
