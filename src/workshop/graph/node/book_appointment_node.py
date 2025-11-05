import logging
from datetime import datetime, timedelta
from langchain_core.messages import AIMessage

from src.workshop.graph.state import BookingState
from src.workshop.service.AppointmentService import AppointmentService

system_logger = logging.getLogger(__name__)

class BookAppointmentNode:
    def __init__(self, appointment_service: AppointmentService):
        self.appointment_service = appointment_service

    def execute(self, state: BookingState) -> dict:
        system_logger.info("---BOOKING NODE: BOOK APPOINTMENT---")
        patient_id = "BN_" + state["patient_name"].split(" ")[-1].upper() if state.get("patient_name") else "BN_UNKNOWN"
        doctor_id = "BS_" + state["doctor_name"].split(" ")[-1].upper() if state.get("doctor_name") else "BS_UNKNOWN"
        appointment_time = (datetime.now() + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)

        success = self.appointment_service.book_appointment(patient_id, doctor_id, appointment_time)
        if success:
            response_text = f"Đã đặt lịch thành công cho bệnh nhân {state['patient_name']} vào 8:00 sáng mai với bác sĩ {state['doctor_name']}."
        else:
            response_text = "Rất tiếc, đã có lỗi xảy ra khi đặt lịch. Vui lòng thử lại sau."
        return {**state, "messages": state["messages"] + [AIMessage(content=response_text)], "next_step": "completed"}
