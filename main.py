"""Main entry point for the interactive AI chat application."""

import logging
import sys
import shutil
from typing import List

from src.workshop.util.LoggingUtil import setup_logging
from src.workshop.util.TTSUtil import speak_text
from src.workshop.util.PropertiesUtil import load_properties
from src.workshop.model.AzureOpenAIModel import AzureOpenAIModel
from src.workshop.service.ConversationService import ConversationService
from src.workshop.service.ConsoleMakeupService import Colors
from src.workshop.service.KnowledgeBaseService import KnowledgeBaseService

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

setup_logging()

system_logger = logging.getLogger(__name__)
conversation_logger = logging.getLogger('conversation')

def main():
    """Initializes all services and runs the main interactive chat loop."""
    # --- Simulated Login: Determine user role ---
    role_input = input("Enter your role (manager/guest): ").lower()
    user_role = "manager" if role_input == "manager" else "guest"
    print(f"Session started with role: {user_role.upper()}\n")
    # -------------------------------------------

    app_config = load_properties("config/config.properties")
    tts_enabled = app_config.get("tts.enabled", "true").lower() == "true"
    system_logger.info(f"TTS functionality is {'enabled' if tts_enabled else 'disabled'}.")

    if shutil.which("ffmpeg"):
        system_logger.info("ffmpeg found in system PATH.")
    else:
        system_logger.critical("ffmpeg NOT found in system PATH. TTS will likely fail.")

    try:
        model_config = AzureOpenAIModel()
        knowledge_base_service = KnowledgeBaseService(model=model_config)
        chat_service = ConversationService(model=model_config, knowledge_base=knowledge_base_service)

        print(f"--- Interactive Chat Session (Role: {user_role.upper()}) ---")
        print("Type 'exit' to end the conversation.")

        chat_history: List[BaseMessage] = []

        while True:
            user_prefix = f"{Colors.GREEN}{Colors.UNDERLINE}You:{Colors.RESET}{Colors.GREEN}"
            user_input = input(f"{user_prefix} ")
            
            conversation_logger.info(f"[User][{user_role}]: {user_input}")

            if user_input.lower() == 'exit':
                break
            
            ai_prefix = f"{Colors.YELLOW}{Colors.UNDERLINE}AI:{Colors.RESET}{Colors.YELLOW}"
            print(f"{ai_prefix} ", end="")
            
            full_response = ""
            # Pass the user role to the service
            for chunk in chat_service.generate_chat_response(user_prompt=user_input, history=chat_history, user_role=user_role):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print(f"{Colors.RESET}")
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=full_response))
            
            conversation_logger.info(f"[Assistant]: {full_response}")
            
            if tts_enabled:
                speak_text(full_response, lang='vi')

    except Exception as e:
        system_logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        system_logger.info("AI project finished. Shutting down logging.")
        logging.shutdown()

if __name__ == "__main__":
    main()
