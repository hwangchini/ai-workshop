import logging
import re
from langchain_core.messages import AIMessage

from src.workshop.graph.state import BookingState
from src.workshop.service.KnowledgeBaseService import KnowledgeBaseService

system_logger = logging.getLogger(__name__)

class AskDoctorNode:
    def __init__(self, knowledge_base: KnowledgeBaseService):
        self.knowledge_base = knowledge_base

    def execute(self, state: BookingState) -> dict:
        system_logger.info("---ASKING NODE: DOCTOR---")
        ask_message = "Bạn muốn đặt lịch khám với bác sĩ cụ thể nào không?"
        symptoms = state.get("symptoms")
        
        if symptoms:
            system_logger.info(f"Finding specialty for symptom: '{symptoms}'")
            symptom_docs = self.knowledge_base.search_semantic(symptoms, n_results=1, where={"doc_type": "symptom"})
            
            recommended_specialty = None
            if symptom_docs:
                match = re.search(r"bác sĩ ([\w\s-]+)", symptom_docs[0])
                if match:
                    recommended_specialty = match.group(1).strip()
                    system_logger.info(f"Recommended specialty found: '{recommended_specialty}'")

            if recommended_specialty:
                search_query = f"bác sĩ chuyên khoa {recommended_specialty}"
                system_logger.info(f"Suggesting doctors with targeted query: '{search_query}'")
                doctors = self.knowledge_base.search_semantic(search_query, n_results=3, where={"doc_type": "doctor"})
                
                if doctors:
                    doctor_suggestions = "\nTrong lúc đó, bạn có thể tham khảo một số bác sĩ chuyên khoa sau:\n" + "\n".join(doctors)
                    ask_message += doctor_suggestions
        
        return {**state, "messages": state["messages"] + [AIMessage(content=ask_message)], "next_step": "ask_missing_info"}
