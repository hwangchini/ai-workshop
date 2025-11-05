"""Service layer that uses LangGraph to create a stateful, routing agent."""

import logging
from typing import Generator, List, Dict, Any

from langchain_core.messages import HumanMessage, BaseMessage
from langchain.memory import ConversationSummaryBufferMemory

from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel
from src.workshop.util.LoggingUtil import log_activity
from src.workshop.service.KnowledgeBaseService import KnowledgeBaseService
from src.workshop.service.AppointmentService import AppointmentService
from src.workshop.graph.builder.AskMissingInfoGraphBuilder import AskMissingInfoGraphBuilder
from src.workshop.graph.builder.InformationGatheringGraphBuilder import InformationGatheringGraphBuilder
from src.workshop.graph.builder.BookingGraphBuilder import BookingGraphBuilder
from src.workshop.graph.builder.MainGraphBuilder import MainGraphBuilder

system_logger = logging.getLogger(__name__)

class ConversationService:
    @log_activity
    def __init__(self, model: AzureOpenAIModel, knowledge_base: KnowledgeBaseService):
        self.model = model
        self.knowledge_base = knowledge_base
        self.appointment_service = AppointmentService()
        self.agent_state: Dict[str, Any] = {} # Persistent state for the agent

        # Initialize the intelligent memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.model.llm,
            max_token_limit=1000,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer"
        )

        # Build the dependency chain of graphs
        ask_missing_info_builder = AskMissingInfoGraphBuilder(self.model, self.knowledge_base)
        ask_missing_info_graph = ask_missing_info_builder.build()

        info_gathering_builder = InformationGatheringGraphBuilder(self.model, self.knowledge_base, ask_missing_info_graph)
        info_gathering_graph = info_gathering_builder.build()

        booking_graph_builder = BookingGraphBuilder(info_gathering_graph, self.appointment_service)
        booking_graph = booking_graph_builder.build()

        main_graph_builder = MainGraphBuilder(self.model, self.knowledge_base, booking_graph)
        self.main_graph = main_graph_builder.build()

        system_logger.info("ConversationService initialized with a deeply nested graph architecture.")

    @log_activity
    def generate_chat_response(self, user_prompt: str, history: List[BaseMessage], user_role: str) -> Generator[str, None, None]:
        try:
            # Load chat history from memory
            memory_variables = self.memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])

            # Load the persistent state from the previous turn and update it
            current_state = self.agent_state.copy()
            current_state.update({
                "user_question": user_prompt,
                "chat_history": chat_history,
                "user_role": user_role,
            })

            # Invoke the main graph
            final_state = self.main_graph.invoke(current_state)
            ai_response = final_state.get("answer", "I'm sorry, I encountered an error.")

            # Save the entire final state to be used in the next turn
            self.agent_state = final_state

            # Save the context of this turn to memory
            self.memory.save_context(
                {"input": user_prompt},
                {"answer": ai_response}
            )

            for char in ai_response:
                yield char
        except Exception as e:
            system_logger.error(f"An error occurred in the LangGraph agent: {e}", exc_info=True)
            yield "I'm sorry, but I encountered an error. Please try again."
