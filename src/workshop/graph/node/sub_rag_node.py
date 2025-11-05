import logging
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import AIMessage

from src.workshop.graph.state import BookingState
from src.workshop.graph.prompts import RAG_SECURITY_PROMPT, RETRIEVAL_QUERY_GEN_PROMPT
from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel
from src.workshop.service.KnowledgeBaseService import KnowledgeBaseService

system_logger = logging.getLogger(__name__)

class SubRagNode:
    def __init__(self, model: AzureOpenAIModel, knowledge_base: KnowledgeBaseService):
        self.model = model
        self.knowledge_base = knowledge_base

    def execute(self, state: BookingState) -> dict:
        system_logger.info("---SUB-RAG NODE---")
        question = state["messages"][-1].content
        chat_history = state["messages"][:-1]

        retrieval_query_chain = RETRIEVAL_QUERY_GEN_PROMPT | self.model.llm | StrOutputParser()
        retrieval_query_raw = retrieval_query_chain.invoke({"question": question, "chat_history": chat_history})
        retrieval_query = retrieval_query_raw.strip().replace('"', '')

        system_logger.info(f"Sub-RAG optimized query: {retrieval_query}")

        context_docs = self.knowledge_base.search_semantic(retrieval_query)
        context_str = "\n".join(context_docs)

        rag_chain = RAG_SECURITY_PROMPT | self.model.llm | StrOutputParser()
        response_text = rag_chain.invoke({
            "context": context_str,
            "chat_history": chat_history,
            "question": question,
            "user_role": "guest", # Assume guest role for sub-queries
            "current_patient_name": state.get("patient_name")
        })

        # Add the RAG response, but keep the flow going
        return {**state, "messages": state["messages"] + [AIMessage(content=response_text)]}
