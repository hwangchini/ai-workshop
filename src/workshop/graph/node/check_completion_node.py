import logging
from src.workshop.graph.state import BookingState

system_logger = logging.getLogger(__name__)

class CheckCompletionNode:
    def execute(self, state: BookingState) -> dict:
        system_logger.info("---BOOKING NODE: CHECK COMPLETION---")
        
        # Check if the last message from AI was a confirmation question
        last_ai_message = ""
        if len(state["messages"]) > 1:
            if state["messages"][-2].type == "ai":
                last_ai_message = state["messages"][-2].content

        if "Thông tin trên đã chính xác chưa ạ?" in last_ai_message:
            system_logger.info("User is responding to a confirmation. Routing to process_confirmation.")
            return {**state, "next_step": "process_confirmation"}

        # Original logic if it's not a confirmation response
        missing_fields = []
        if not state.get("patient_name"): missing_fields.append("tên của bạn")
        if not state.get("phone_number"): missing_fields.append("số điện thoại")
        if not state.get("symptoms"): missing_fields.append("triệu chứng bạn đang gặp phải")
        if not state.get("doctor_name"): missing_fields.append("bác sĩ bạn muốn khám")
        
        if missing_fields:
            system_logger.info(f"Booking information is incomplete. Missing: {missing_fields}")
            return {**state, "next_step": "ask_missing_info"}
        else:
            system_logger.info("Booking information is complete. Proceeding to confirmation.")
            return {**state, "next_step": "confirm_info"}
