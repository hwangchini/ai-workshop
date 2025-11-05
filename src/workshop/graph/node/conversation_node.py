import logging
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser

from src.workshop.config import MAIN_SYSTEM_PROMPT
from src.workshop.graph.state import AgentState
from src.workshop.model.llm import LLM

system_logger = logging.getLogger(__name__)

class ConversationNode:
    def __init__(self, model: LLM):
        self.model = model

    def execute(self, state: AgentState) -> dict:
        system_logger.info("---NODE: CONVERSATION---")
        
        conversation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=MAIN_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{user_question}"),
        ])
        
        conversation_chain = conversation_prompt | self.model.llm | StrOutputParser()
        response_text = conversation_chain.invoke({
            "chat_history": state["chat_history"],
            "user_question": state["user_question"]
        })
        
        return {"answer": response_text}
