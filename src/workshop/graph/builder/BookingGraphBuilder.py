from langgraph.graph import StateGraph, END

from src.workshop.graph.state import BookingState
from src.workshop.graph.node.book_appointment_node import BookAppointmentNode
from src.workshop.model.llm import LLM
from src.workshop.rag.knowledge_base import KnowledgeBase
from src.workshop.service.AppointmentService import AppointmentService

class BookingGraphBuilder:
    def __init__(self, model: LLM, knowledge_base: KnowledgeBase, appointment_service: AppointmentService):
        self.model = model
        self.knowledge_base = knowledge_base
        self.appointment_service = appointment_service

    def build(self) -> StateGraph:
        # Initialize nodes
        book_appointment_node = BookAppointmentNode(self.appointment_service)

        # Define the workflow
        workflow = StateGraph(BookingState)
        workflow.add_node("book_appointment", book_appointment_node.execute)

        # Set entry and build the graph structure
        workflow.set_entry_point("book_appointment")
        workflow.add_edge("book_appointment", END)
        
        return workflow.compile()
