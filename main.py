"""Main entry point for the interactive AI chat application."""

import sys
import logging
from typing import List

from src.workshop.config import APP_CONFIG
from src.workshop.model.llm import LLM
from src.workshop.service.ConversationService import ConversationService
from src.workshop.service.ConsoleMakeupService import Colors
from src.workshop.rag.knowledge_base import KnowledgeBase

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

# Basic logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress verbose logs from third-party libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    """Initializes all services and runs the main interactive chat loop."""
    logging.info("Session started.")

    try:
        model_config = LLM()
        knowledge_base_service = KnowledgeBase(model=model_config)
        chat_service = ConversationService(model=model_config, knowledge_base=knowledge_base_service)

        print(f"--- Interactive Chat Session ---")
        print("Type 'exit' to end the conversation.")

        chat_history: List[BaseMessage] = []

        while True:
            user_prefix = f"{Colors.GREEN}{Colors.UNDERLINE}You:{Colors.RESET}{Colors.GREEN}"
            user_input = input(f"{user_prefix} ")

            if user_input.lower() == 'exit':
                break
            
            ai_prefix = f"{Colors.YELLOW}{Colors.UNDERLINE}AI:{Colors.RESET}{Colors.YELLOW}"
            print(f"{ai_prefix} ", end="")
            
            full_response = ""
            try:
                for chunk in chat_service.generate_chat_response(user_prompt=user_input, history=chat_history):
                    print(chunk, end="", flush=True)
                    full_response += chunk
            except Exception as e:
                logging.error("An error occurred during chat response generation: %s", e, exc_info=True)
                print(f"\n{Colors.YELLOW}ERROR: An unexpected error occurred. Please check the logs for details.{Colors.RESET}")
                continue
            
            print(f"{Colors.RESET}")
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=full_response))

    except Exception as e:
        logging.error("An unexpected error occurred in the main loop: %s", e, exc_info=True)
        print(f"An unexpected error occurred. Please check the logs for details.")
    finally:
        logging.info("AI project finished.")

if __name__ == "__main__":
    main()
