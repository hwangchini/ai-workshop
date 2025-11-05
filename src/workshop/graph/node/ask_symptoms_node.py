import logging
from langchain_core.messages import AIMessage
from src.workshop.graph.state import BookingState

system_logger = logging.getLogger(__name__)

class AskSymptomsNode:
    def execute(self, state: BookingState) -> dict:
        system_logger.info("---ASKING NODE: SYMPTOMS---")
        patient_name = state.get("patient_name", "bạn")
        ask_message = f"Chào {patient_name}, bạn đang gặp phải triệu chứng hay vấn đề sức khỏe gì?"
        return {**state, "messages": state["messages"] + [AIMessage(content=ask_message)], "next_step": "ask_missing_info"}
