import logging
import json
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel

system_logger = logging.getLogger(__name__)

class ChromaDBEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model: AzureOpenAIModel):
        self.model = model

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return self.model.embedding_model.embed_documents(input)

class KnowledgeBaseService:
    def __init__(self, model: AzureOpenAIModel):
        self.model = model
        self.client = chromadb.Client()
        self.embedding_function = ChromaDBEmbeddingFunction(model)
        try:
            self.collection = self.client.get_collection(
                name="medical_data",
                embedding_function=self.embedding_function
            )
            system_logger.info("Successfully connected to existing ChromaDB collection 'medical_data'.")
        except ValueError:
            system_logger.info("Collection 'medical_data' did not exist. Creating a new one.")
            self.collection = self.client.create_collection(
                name="medical_data",
                embedding_function=self.embedding_function
            )
            self._index_data()

    def _load_and_prepare_data(self, file_path: str, root_key: str, doc_type: str, fields_to_include: List[str], id_field: str) -> List[Dict[str, Any]]:
        system_logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items_list = data.get(root_key, [])
        if not isinstance(items_list, list):
            system_logger.error(f"Expected a list for root_key '{root_key}' in {file_path}, but got {type(items_list)}.")
            return []

        prepared_data = []
        for item in items_list:
            item_id = item.get(id_field)
            if not item_id:
                system_logger.warning(f"Skipping item in {file_path} because it is missing the required id_field '{id_field}'. Item: {item}")
                continue

            # Improved document content formatting
            if doc_type == "doctor":
                document_content = f"Thông tin bác sĩ (Mã: {item.get('id')}). Tên: {item.get('name')}, Bằng cấp: {item.get('degree')}, Chức vụ: {item.get('position')}, Chuyên khoa: {item.get('specialty')}."
            else:
                document_content = f"Loại tài liệu: {doc_type}. " + ". ".join([f"{field}: {item.get(field, 'N/A')}" for field in fields_to_include])

            metadata = {"doc_type": doc_type, "source": file_path}
            for key, value in item.items():
                if isinstance(value, (str, int, float, bool)):
                     metadata[key] = value
            prepared_data.append({"document": document_content, "metadata": metadata, "id": str(item_id)})
        return prepared_data

    def _index_data(self):
        system_logger.info("Starting data indexing...")
        all_data = []
        
        personnel_data = self._load_and_prepare_data("data/medical_personnel.json", "medical_personnel", "doctor", ["id", "name", "degree", "position", "specialty"], id_field="id")
        patient_data = self._load_and_prepare_data("data/patient_data.json", "patients", "patient", ["patient_id", "full_name", "date_of_birth", "gender", "medical_history"], id_field="patient_id")
        symptom_data = self._load_and_prepare_data("data/symptoms.json", "symptoms", "symptom", ["symptom_name", "details"], id_field="symptom_name")

        all_data.extend(personnel_data)
        all_data.extend(patient_data)
        all_data.extend(symptom_data)

        documents = [item["document"] for item in all_data]
        metadatas = [item["metadata"] for item in all_data]
        ids = [item["id"] for item in all_data]

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            system_logger.info(f"Successfully indexed {len(documents)} documents into ChromaDB.")

    def search_semantic(self, query: str, n_results: int = 3, where: Dict[str, Any] = None) -> List[str]:
        system_logger.info(f"Performing semantic search for query: '{query}' with where filter: {where}")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results['documents'][0] if results['documents'] else []
