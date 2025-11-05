import logging
from datetime import datetime
from langchain_core.output_parsers.json import JsonOutputParser

from src.workshop.graph.state import BookingState
from src.workshop.graph.prompts import APPOINTMENT_EXTRACTOR_PROMPT
from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel

system_logger = logging.getLogger(__name__)

class InformationExtractionNode:
    def __init__(self, model: AzureOpenAIModel):
        self.model = model

    def execute(self, state: BookingState) -> dict:
        system_logger.info("---BOOKING NODE: EXTRACT INFO---")
        request = state["messages"][-1].content
        chat_history_for_extractor = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"][:-1]])
        
        extractor_chain = APPOINTMENT_EXTRACTOR_PROMPT | self.model.llm | JsonOutputParser()
        newly_extracted_info = extractor_chain.invoke({
            "request": request,
            "today": datetime.now().strftime("%Y-%m-%d"),
            "chat_history": chat_history_for_extractor
        })
        
        current_booking_state = state.copy()

        for key, value in newly_extracted_info.items():
            if value is not None:
                current_booking_state[key] = value
        
        current_booking_state["messages"] = state["messages"]

        system_logger.info(f"Extracted booking info: {current_booking_state}")
        return current_booking_state
