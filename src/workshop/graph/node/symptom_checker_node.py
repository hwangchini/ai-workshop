import logging
from langchain_core.output_parsers.string import StrOutputParser

from src.workshop.graph.state import AgentState
from src.workshop.graph.prompts import SYMPTOM_CHECKER_PROMPT, RETRIEVAL_QUERY_GEN_PROMPT
from src.workshop.model.llm import LLM
from src.workshop.rag.knowledge_base import KnowledgeBase

system_logger = logging.getLogger(__name__)

class SymptomCheckerNode:
    def __init__(self, model: LLM, knowledge_base: KnowledgeBase):
        self.model = model
        self.knowledge_base = knowledge_base

    def execute(self, state: AgentState) -> dict:
        system_logger.info("---NODE: SYMPTOM_CHECKER---")
        question = state["user_question"]
        chat_history = state["chat_history"]
        
        # Generate a concise retrieval query optimized for semantic search, just like in RagNode
        retrieval_query_chain = RETRIEVAL_QUERY_GEN_PROMPT | self.model.llm | StrOutputParser()
        retrieval_query_raw = retrieval_query_chain.invoke({"question": question, "chat_history": chat_history})
        retrieval_query = retrieval_query_raw.strip().replace('"', '')

        system_logger.info(f"Original question: {question}")
        system_logger.info(f"Optimized retrieval query for symptoms: {retrieval_query}")

        # Perform semantic search with the cleaned, optimized query
        context_docs = self.knowledge_base.search_semantic(retrieval_query, n_results=3)
        context_str = "\n---\n".join(context_docs)
        
        # Generate the final answer
        checker_chain = SYMPTOM_CHECKER_PROMPT | self.model.llm | StrOutputParser()
        response_text = checker_chain.invoke({
            "context": context_str,
            "question": question,
            "chat_history": chat_history
        })
        
        return {"answer": response_text}
