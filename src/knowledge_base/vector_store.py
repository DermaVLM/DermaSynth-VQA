import os
import json
from typing import List, Dict

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class DermatologyVectorStore:
    def __init__(self, persist_dir: str = "./dermatology_index"):
        self.persist_dir = persist_dir
        # Initialize the text splitter
        self.chunk_size = 1024
        self.chunk_overlap = 100

        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def create_document_from_entry(self, entry: Dict) -> Document:
        """Create a structured document from a medical entry"""
        content_parts = [f"# {entry['name']}\n"]

        metadata = {
            "title": entry["name"],
            "url": entry["url"],
            "categories": entry.get("categories", []),
        }

        section_order = [
            "overview",
            "clinical_presentation",
            "diagnosis",
            "differential_diagnosis",
            "laboratory_findings",
            "imaging",
            "pathophysiology",
            "causes_and_risk_factors",
            "treatment",
            "surgery",
            "epidemiology",
            "complications",
            "subtypes",
            "history",
            "other",
        ]

        for section in section_order:
            section_key = section.lower()
            if (
                section_key in entry["sections"]
                and entry["sections"][section_key].strip()
            ):
                content_parts.extend(
                    [
                        f"\n## {section.replace('_', ' ').title()}",
                        entry["sections"][section_key],
                    ]
                )

        full_content = "\n".join(content_parts)
        return Document(text=full_content, metadata=metadata)

    def build_index(self, input_file: str):
        """Build and save the vector store index"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Load and process documents
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = [
            self.create_document_from_entry(entry)
            for entry in data
            if entry["sections"]
        ]

        # Create node parser with sentence splitting
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(documents)

        # Create index with embedding model
        index = VectorStoreIndex.from_documents(documents, embed_model=self.embed_model)

        # Save index
        index.storage_context.persist(persist_dir=self.persist_dir)
        return index

    def load_index(self):
        """Load existing vector store index"""
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(f"No index found at {self.persist_dir}")

        storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        return load_index_from_storage(storage_context, embed_model=self.embed_model)

    def get_relevant_chunks(self, query_text: str, num_chunks: int = 5):
        """Retrieve relevant text chunks for a query without LLM processing"""
        index = self.load_index()
        retriever = index.as_retriever(similarity_top_k=num_chunks)
        nodes = retriever.retrieve(query_text)

        return [
            {
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score if hasattr(node, "score") else None,
            }
            for node in nodes
        ]


def main():
    source_dir = "src\\knowledge_base"
    vector_store = DermatologyVectorStore(persist_dir=f"{source_dir}/dermatology_index")

    # build index only if it does not exist
    if not os.path.exists(os.path.join(source_dir, "dermatology_index")):
        vector_store.build_index(
            os.path.join(source_dir, "processed_dermatology_kb.json")
        )

    # Example retrieval
    # query = "What are the symptoms of psoriasis?"
    queries = [
        # Instruction queries
        "Identify the presence of fluid-filled lesions on the skin.",
        "Look for signs of hair loss, scaly scalp, or other scalp lesions.",
        # Input queries
        "Procedure: Excision Biopsy",
        "Erythematous, possibly exudative patch; uncertain if fungal or bacterial",
        # Output queries
        "Appearance changed from a pigmented papule to a healing site with a scab after biopsy.",
        "Body dysmorphic disorder involves an obsessive focus on minor or imagined defects; excoriation disorder entails compulsive skin picking, both needing psychological support.",
    ]
    for query in queries:
        print(f"\n\n\nQuery: {query}")
        relevant_chunks = vector_store.get_relevant_chunks(query)

        for i, chunk in enumerate(relevant_chunks, 1):
            print(
                f"\nChunk {i} (Score: {chunk['score'] if chunk['score'] else 'N/A'}):"
            )
            print(f"From: {chunk['metadata'].get('title', 'Unknown')}")
            print(f"Text: {chunk['text'][:200]}...")
            break  # Only show the top result


if __name__ == "__main__":
    main()
