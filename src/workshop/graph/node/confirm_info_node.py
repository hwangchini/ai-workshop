import logging
from langchain_core.messages import AIMessage

from src.workshop.graph.state import BookingState

system_logger = logging.getLogger(__name__)

class ConfirmInfoNode:
    def execute(self, state: BookingState) -> dict:
        system_logger.info("---BOOKING NODE: CONFIRM INFO---")
        
        patient_name = state.get("patient_name")
        symptoms = state.get("symptoms")
        doctor_name = state.get("doctor_name")
        phone_number = state.get("phone_number")

        confirmation_message = (
            f"Tuyệt vời! Chúng ta hãy cùng xác nhận lại thông tin nhé:\n"
            f"- Tên bệnh nhân: {patient_name}\n"
            f"- Triệu chứng: {symptoms}\n"
            f"- Bác sĩ: {doctor_name}\n"
            f"- Số điện thoại: {phone_number}\n\n"
            f"Thông tin trên đã chính xác chưa ạ? (Vui lòng trả lời 'đúng' hoặc 'sai')"
        )

        return {**state, "messages": state["messages"] + [AIMessage(content=confirmation_message)], "next_step": "confirm_info"}
