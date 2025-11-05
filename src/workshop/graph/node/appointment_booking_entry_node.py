import logging
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

from src.workshop.graph.state import AgentState, BookingState

system_logger = logging.getLogger(__name__)

class AppointmentBookingEntryNode:
    def __init__(self, booking_graph: StateGraph):
        self.booking_graph = booking_graph

    def execute(self, state: AgentState) -> dict:
        system_logger.info("---ENTERING BOOKING SUB-GRAPH---")
        
        # Set the booking flow flag to True
        state["is_in_booking_flow"] = True

        # Construct the message history for the sub-graph to process
        sub_graph_messages = state["chat_history"] + [HumanMessage(content=state["user_question"])]

        # Prepare the initial state for the booking sub-graph
        booking_state: BookingState = {
            "messages": sub_graph_messages,
            "patient_name": state.get("current_patient_name"),
            "phone_number": None, 
            "symptoms": None, 
            "doctor_name": None,
            "next_step": None
        }
        # Carry over any previously gathered info from the main state
        if state.get("booking_state"):
            for key, value in state["booking_state"].items():
                if key != "messages" and value is not None:
                    booking_state[key] = value

        # Invoke the booking sub-graph
        final_booking_state = self.booking_graph.invoke(booking_state)
        
        # The final message from the sub-graph is the answer for this turn
        final_answer = final_booking_state["messages"][-1].content

        # Prepare the state update for the main graph
        updated_agent_state = {
            "answer": final_answer,
            "booking_state": final_booking_state
        }
        if final_booking_state.get("patient_name"):
            updated_agent_state["current_patient_name"] = final_booking_state["patient_name"]

        # If the booking process is finished (e.g., cancelled or completed), lower the flag.
        # The sub-graph signals this with a specific next_step.
        if final_booking_state.get("next_step") == "end_booking":
            system_logger.info("---BOOKING FLOW CONCLUDED---")
            updated_agent_state["is_in_booking_flow"] = False
        
        return updated_agent_state
