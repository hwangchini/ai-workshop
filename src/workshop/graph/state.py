from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    user_question: str
    chat_history: List[BaseMessage]
    answer: str
    user_role: str
    destination: str
    booking_state: Optional[dict]
    current_patient_name: Optional[str]
    is_in_booking_flow: bool # Flag to track if we are in the booking process

class BookingState(TypedDict):
    messages: List[BaseMessage]
    patient_name: Optional[str]
    phone_number: Optional[str]
    symptoms: Optional[str]
    doctor_name: Optional[str]
    next_step: Optional[str]
