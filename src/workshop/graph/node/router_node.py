import logging
from langchain_core.output_parsers.string import StrOutputParser
from src.workshop.graph.state import AgentState
from src.workshop.graph.prompts import ROUTER_PROMPT
from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel

system_logger = logging.getLogger(__name__)

class RouterNode:
    def __init__(self, model: AzureOpenAIModel):
        self.model = model

    def execute(self, state: AgentState) -> dict:
        system_logger.info("---NODE: ROUTER---")
        question = state["user_question"]
        is_in_booking_flow = state.get("is_in_booking_flow", False)
        
        router_chain = ROUTER_PROMPT | self.model.llm | StrOutputParser()
        destination = router_chain.invoke({
            "question": question,
            "is_in_booking_flow": is_in_booking_flow
        })
        destination = destination.strip().replace("`", "") # Clean the destination string
        system_logger.info(f"Router destination: {destination}")
        return {"destination": destination}
