import logging
from langchain_core.messages import AIMessage

from src.workshop.graph.state import BookingState

system_logger = logging.getLogger(__name__)

class ProcessConfirmationNode:
    def execute(self, state: BookingState) -> dict:
        system_logger.info("---BOOKING NODE: PROCESS CONFIRMATION---")
        user_response = state["messages"][-1].content.lower()

        if "đúng" in user_response or "chính xác" in user_response:
            system_logger.info("User confirmed booking info. Proceeding to book.")
            return {**state, "next_step": "book_appointment"}
        else:
            system_logger.info("User denied booking info. Restarting information gathering.")
            # Resetting the state to ask again
            state["patient_name"] = None
            state["phone_number"] = None
            state["symptoms"] = None
            state["doctor_name"] = None
            ask_message = "Rất xin lỗi vì sự nhầm lẫn. Chúng ta hãy bắt đầu lại nhé."
            return {**state, "messages": state["messages"] + [AIMessage(content=ask_message)], "next_step": "ask_missing_info"}
