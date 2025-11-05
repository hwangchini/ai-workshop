import logging
from langchain_core.output_parsers.string import StrOutputParser
from src.workshop.graph.state import BookingState
from src.workshop.graph.prompts import SUB_ROUTER_PROMPT
from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel

system_logger = logging.getLogger(__name__)

class SubRouterNode:
    def __init__(self, model: AzureOpenAIModel):
        self.model = model

    def execute(self, state: BookingState) -> dict:
        system_logger.info("---SUB-ROUTER NODE---")
        question = state["messages"][-1].content
        chat_history = state["messages"][:-1]
        
        router_chain = SUB_ROUTER_PROMPT | self.model.llm | StrOutputParser()
        destination = router_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        destination = destination.strip().replace("`", "")
        system_logger.info(f"Sub-Router destination: {destination}")
        return {"next_step": destination}
