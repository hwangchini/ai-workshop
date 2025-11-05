import logging
from langchain_core.messages import AIMessage
from src.workshop.graph.state import BookingState

system_logger = logging.getLogger(__name__)

class AskNameNode:
    def execute(self, state: BookingState) -> dict:
        system_logger.info("---ASKING NODE: NAME---")
        ask_message = "Để bắt đầu, bạn vui lòng cho tôi biết tên của bạn là gì?"
        return {**state, "messages": state["messages"] + [AIMessage(content=ask_message)], "next_step": "ask_missing_info"}
