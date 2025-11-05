import logging
from src.workshop.graph.state import BookingState

system_logger = logging.getLogger(__name__)

class SequentialAskerNode:
    def execute(self, state: BookingState) -> dict:
        system_logger.info("---ASKER NODE: DECIDING WHICH QUESTION TO ASK---")
        
        if not state.get("patient_name"):
            return {"next_step": "ask_name"}

        if not state.get("symptoms"):
            return {"next_step": "ask_symptoms"}

        if not state.get("doctor_name"):
            return {"next_step": "ask_doctor"}

        if not state.get("phone_number"):
            return {"next_step": "ask_phone"}

        # Should not be reached if called correctly
        return {"next_step": "end"}
