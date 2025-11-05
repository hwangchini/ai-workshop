from langgraph.graph import StateGraph, END

from src.workshop.graph.state import AgentState
from src.workshop.graph.node.router_node import RouterNode
from src.workshop.graph.node.rag_node import RagNode
from src.workshop.graph.node.conversation_node import ConversationNode
from src.workshop.graph.node.symptom_checker_node import SymptomCheckerNode
from src.workshop.graph.node.appointment_booking_entry_node import AppointmentBookingEntryNode
from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel
from src.workshop.service.KnowledgeBaseService import KnowledgeBaseService

class MainGraphBuilder:
    def __init__(self, model: AzureOpenAIModel, knowledge_base: KnowledgeBaseService, booking_graph: StateGraph):
        self.model = model
        self.knowledge_base = knowledge_base
        self.booking_graph = booking_graph

    def build(self) -> StateGraph:
        router_node = RouterNode(self.model)
        rag_node = RagNode(self.model, self.knowledge_base)
        conversation_node = ConversationNode(self.model)
        symptom_checker_node = SymptomCheckerNode(self.model, self.knowledge_base)
        appointment_booking_entry_node = AppointmentBookingEntryNode(self.booking_graph)

        main_workflow = StateGraph(AgentState)
        main_workflow.add_node("router", router_node.execute)
        main_workflow.add_node("rag", rag_node.execute)
        main_workflow.add_node("conversation", conversation_node.execute)
        main_workflow.add_node("symptom_checker", symptom_checker_node.execute)
        main_workflow.add_node("appointment_booking", appointment_booking_entry_node.execute)
        main_workflow.add_node("process_confirmation", appointment_booking_entry_node.execute) # Re-use the entry node

        main_workflow.set_entry_point("router")
        main_workflow.add_conditional_edges(
            "router",
            lambda x: x["destination"].lower().strip(),
            {
                "rag_query": "rag",
                "symptom_checker": "symptom_checker",
                "appointment_booking": "appointment_booking",
                "process_confirmation": "process_confirmation",
                "general_conversation": "conversation"
            }
        )
        main_workflow.add_edge("rag", END)
        main_workflow.add_edge("conversation", END)
        main_workflow.add_edge("symptom_checker", END)
        main_workflow.add_edge("appointment_booking", END)
        main_workflow.add_edge("process_confirmation", END)

        return main_workflow.compile()
