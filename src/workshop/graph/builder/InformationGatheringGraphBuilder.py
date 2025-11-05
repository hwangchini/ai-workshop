from langgraph.graph import StateGraph, END

from src.workshop.graph.state import BookingState
from src.workshop.graph.node.information_extraction_node import InformationExtractionNode
from src.workshop.graph.node.sub_router_node import SubRouterNode
from src.workshop.graph.node.sub_rag_node import SubRagNode
from src.workshop.graph.node.cancel_node import CancelNode
from src.workshop.graph.node.check_completion_node import CheckCompletionNode
# FIX: Corrected the import path from .node to .builder
from src.workshop.graph.builder.AskMissingInfoGraphBuilder import AskMissingInfoGraphBuilder
from src.workshop.graph.node.confirm_info_node import ConfirmInfoNode
from src.workshop.graph.node.process_confirmation_node import ProcessConfirmationNode
from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel
from src.workshop.service.KnowledgeBaseService import KnowledgeBaseService

class InformationGatheringGraphBuilder:
    def __init__(self, model: AzureOpenAIModel, knowledge_base: KnowledgeBaseService, ask_missing_info_graph: StateGraph):
        self.model = model
        self.knowledge_base = knowledge_base
        self.ask_missing_info_graph = ask_missing_info_graph

    def build(self) -> StateGraph:
        # Initialize nodes
        info_extraction_node = InformationExtractionNode(self.model)
        sub_router_node = SubRouterNode(self.model)
        sub_rag_node = SubRagNode(self.model, self.knowledge_base)
        cancel_node = CancelNode()
        check_completion_node = CheckCompletionNode()
        confirm_info_node = ConfirmInfoNode()
        process_confirmation_node = ProcessConfirmationNode()

        def ask_missing_info_entry_node(state: BookingState):
            return self.ask_missing_info_graph.invoke(state)

        # Define the workflow
        workflow = StateGraph(BookingState)
        workflow.add_node("information_extraction", info_extraction_node.execute)
        workflow.add_node("sub_router", sub_router_node.execute)
        workflow.add_node("sub_rag", sub_rag_node.execute)
        workflow.add_node("cancel", cancel_node.execute)
        workflow.add_node("check_completion", check_completion_node.execute)
        workflow.add_node("ask_missing_info", ask_missing_info_entry_node)
        workflow.add_node("confirm_info", confirm_info_node.execute)
        workflow.add_node("process_confirmation", process_confirmation_node.execute)

        # Set entry and build the graph structure
        workflow.set_entry_point("information_extraction")
        workflow.add_edge("information_extraction", "sub_router")

        # The sub-router decides the next step
        workflow.add_conditional_edges(
            "sub_router",
            lambda x: x["next_step"],
            {
                "provide_info": "check_completion",
                "ask_rag_question": "sub_rag",
                "cancel": "cancel"
            }
        )

        # If the user asks a side question, answer it and then re-ask the original question
        workflow.add_edge("sub_rag", "ask_missing_info")

        # If the user cancels, end this sub-process
        workflow.add_edge("cancel", END)

        # The rest of the logic remains the same as before
        workflow.add_conditional_edges(
            "check_completion",
            lambda x: x["next_step"],
            {
                "ask_missing_info": "ask_missing_info",
                "confirm_info": "confirm_info",
                "process_confirmation": "process_confirmation"
            }
        )

        workflow.add_edge("ask_missing_info", END)
        workflow.add_edge("confirm_info", END)

        workflow.add_conditional_edges(
            "process_confirmation",
            lambda x: x["next_step"],
            {
                "book_appointment": END,
                "ask_missing_info": "ask_missing_info"
            }
        )
        
        return workflow.compile()
