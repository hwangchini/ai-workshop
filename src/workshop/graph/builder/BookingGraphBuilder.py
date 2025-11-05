from langgraph.graph import StateGraph, END

from src.workshop.graph.state import BookingState
from src.workshop.graph.node.book_appointment_node import BookAppointmentNode
from src.workshop.service.AppointmentService import AppointmentService

class BookingGraphBuilder:
    def __init__(self, information_gathering_graph: StateGraph, appointment_service: AppointmentService):
        self.information_gathering_graph = information_gathering_graph
        self.appointment_service = appointment_service

    def build(self) -> StateGraph:
        book_appointment_node = BookAppointmentNode(self.appointment_service)

        # This node will call the information gathering subgraph
        def gather_info_entry_node(state: BookingState):
            return self.information_gathering_graph.invoke(state)

        workflow = StateGraph(BookingState)
        workflow.add_node("gather_info", gather_info_entry_node)
        workflow.add_node("book_appointment", book_appointment_node.execute)

        workflow.set_entry_point("gather_info")

        # After gathering info, decide if we can book or if the process was interrupted
        workflow.add_conditional_edges(
            "gather_info",
            lambda x: "book_appointment" if x.get("next_step") == "book_appointment" else END,
            {
                "book_appointment": "book_appointment",
                END: END
            }
        )

        workflow.add_edge("book_appointment", END)
        
        return workflow.compile()
