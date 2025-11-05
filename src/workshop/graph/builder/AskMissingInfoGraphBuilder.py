from langgraph.graph import StateGraph, END

from src.workshop.graph.state import BookingState
from src.workshop.graph.node.sequential_asker_node import SequentialAskerNode
from src.workshop.graph.node.ask_name_node import AskNameNode
from src.workshop.graph.node.ask_symptoms_node import AskSymptomsNode
from src.workshop.graph.node.ask_doctor_node import AskDoctorNode
from src.workshop.graph.node.ask_phone_node import AskPhoneNode
from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel
from src.workshop.service.KnowledgeBaseService import KnowledgeBaseService

class AskMissingInfoGraphBuilder:
    def __init__(self, model: AzureOpenAIModel, knowledge_base: KnowledgeBaseService):
        self.model = model
        self.knowledge_base = knowledge_base

    def build(self) -> StateGraph:
        sequential_asker_node = SequentialAskerNode()
        ask_name_node = AskNameNode()
        ask_symptoms_node = AskSymptomsNode()
        ask_doctor_node = AskDoctorNode(self.knowledge_base)
        ask_phone_node = AskPhoneNode()

        workflow = StateGraph(BookingState)
        workflow.add_node("sequential_asker", sequential_asker_node.execute)
        workflow.add_node("ask_name", ask_name_node.execute)
        workflow.add_node("ask_symptoms", ask_symptoms_node.execute)
        workflow.add_node("ask_doctor", ask_doctor_node.execute)
        workflow.add_node("ask_phone", ask_phone_node.execute)

        workflow.set_entry_point("sequential_asker")

        # Based on the decision of the sequential asker, go to the specific question node
        workflow.add_conditional_edges(
            "sequential_asker",
            lambda x: x["next_step"],
            {
                "ask_name": "ask_name",
                "ask_symptoms": "ask_symptoms",
                "ask_doctor": "ask_doctor",
                "ask_phone": "ask_phone",
                "end": END # If no more questions, end the subgraph
            }
        )

        # After asking any question, the subgraph's turn is over
        workflow.add_edge("ask_name", END)
        workflow.add_edge("ask_symptoms", END)
        workflow.add_edge("ask_doctor", END)
        workflow.add_edge("ask_phone", END)
        
        return workflow.compile()
