import logging
from langchain_core.messages import AIMessage
from src.workshop.graph.state import BookingState

system_logger = logging.getLogger(__name__)

class CancelNode:
    def execute(self, state: BookingState) -> dict:
        system_logger.info("---CANCEL NODE---")
        cancel_message = "Đã hủy yêu cầu của bạn. Tôi có thể giúp gì khác không?"
        
        # Clear the booking state and signal to the parent graph to end
        return {
            "messages": state["messages"] + [AIMessage(content=cancel_message)],
            "patient_name": None,
            "phone_number": None,
            "symptoms": None,
            "doctor_name": None,
            "next_step": "end_booking"
        }
