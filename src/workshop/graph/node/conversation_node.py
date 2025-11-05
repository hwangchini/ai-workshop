import logging
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser

from src.workshop.graph.state import AgentState
from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel
from src.workshop.util.PropertiesUtil import load_properties

system_logger = logging.getLogger(__name__)

class ConversationNode:
    def __init__(self, model: AzureOpenAIModel):
        self.model = model

    def execute(self, state: AgentState) -> dict:
        system_logger.info("---NODE: CONVERSATION---")
        prompts = load_properties("src/workshop/prompt.properties")
        system_message = prompts.get("prompt.main", "You are a helpful AI assistant.")
        
        conversation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{user_question}"),
        ])
        
        conversation_chain = conversation_prompt | self.model.llm | StrOutputParser()
        response_text = conversation_chain.invoke({
            "chat_history": state["chat_history"],
            "user_question": state["user_question"]
        })
        
        return {"answer": response_text}
