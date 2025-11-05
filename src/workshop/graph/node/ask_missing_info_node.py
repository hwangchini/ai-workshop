import logging
import re
from langchain_core.messages import AIMessage

from src.workshop.graph.state import BookingState
from src.workshop.service.KnowledgeBaseService import KnowledgeBaseService

system_logger = logging.getLogger(__name__)

class AskMissingInfoNode:
    def __init__(self, knowledge_base: KnowledgeBaseService):
        self.knowledge_base = knowledge_base

    def execute(self, state: BookingState) -> dict:
        system_logger.info("---BOOKING NODE: ASK MISSING INFO (SEQUENTIAL)---")
        
        # Priority 1: Ask for Patient Name
        if not state.get("patient_name"):
            ask_message = "Để bắt đầu, bạn vui lòng cho tôi biết tên của bạn là gì?"
            return {**state, "messages": state["messages"] + [AIMessage(content=ask_message)], "next_step": "ask_missing_info"}

        # Priority 2: Ask for Symptoms
        if not state.get("symptoms"):
            ask_message = f"Chào {state['patient_name']}, bạn đang gặp phải triệu chứng hay vấn đề sức khỏe gì?"
            return {**state, "messages": state["messages"] + [AIMessage(content=ask_message)], "next_step": "ask_missing_info"}

        # Priority 3: Ask for Doctor (and suggest if possible)
        if not state.get("doctor_name"):
            ask_message = "Bạn muốn đặt lịch khám với bác sĩ cụ thể nào không?"
            symptoms = state.get("symptoms")
            
            # If symptoms are known, try to suggest relevant doctors.
            if symptoms:
                # Step 1: Find the symptom details to identify the recommended specialty.
                system_logger.info(f"Finding specialty for symptom: '{symptoms}'")
                symptom_docs = self.knowledge_base.search_semantic(symptoms, n_results=1, where={"doc_type": "symptom"})
                
                recommended_specialty = None
                if symptom_docs:
                    # FIX: Corrected the regular expression. The previous version had an extra backslash.
                    # This regex now correctly finds the specialty name after "bác sĩ".
                    match = re.search(r"bác sĩ ([\w\s-]+)", symptom_docs[0])
                    if match:
                        recommended_specialty = match.group(1).strip()
                        system_logger.info(f"Recommended specialty found: '{recommended_specialty}'")

                # Step 2: Find doctors of that specialty.
                if recommended_specialty:
                    search_query = f"bác sĩ chuyên khoa {recommended_specialty}"
                    system_logger.info(f"Suggesting doctors with targeted query: '{search_query}'")
                    doctors = self.knowledge_base.search_semantic(search_query, n_results=3, where={"doc_type": "doctor"})
                    
                    if doctors:
                        doctor_suggestions = "\nTrong lúc đó, bạn có thể tham khảo một số bác sĩ chuyên khoa sau:\n" + "\n".join(doctors)
                        ask_message += doctor_suggestions
            
            return {**state, "messages": state["messages"] + [AIMessage(content=ask_message)], "next_step": "ask_missing_info"}

        # Priority 4: Ask for Phone Number
        if not state.get("phone_number"):
            ask_message = "Cảm ơn bạn. Cuối cùng, bạn vui lòng cung cấp số điện thoại để chúng tôi có thể liên hệ xác nhận."
            return {**state, "messages": state["messages"] + [AIMessage(content=ask_message)], "next_step": "ask_missing_info"}

        # Fallback in case this node is called unexpectedly (all info is already complete)
        return {**state}
