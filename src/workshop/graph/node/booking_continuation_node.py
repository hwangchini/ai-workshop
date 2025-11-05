import logging
from src.workshop.graph.state import AgentState

system_logger = logging.getLogger(__name__)

class BookingContinuationNode:
    def execute(self, state: AgentState) -> dict:
        system_logger.info("---NODE: BOOKING CONTINUATION---")
        if state.get("booking_status") == "ask_missing_info":
            system_logger.info("Booking sub-graph needs more information. Looping back.")
            return {"destination": "appointment_booking"}
        else:
            system_logger.info("Booking sub-graph completed or ended.")
            return {"destination": "END"}
