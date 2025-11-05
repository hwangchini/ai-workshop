"""Service for managing patient appointments."""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
APPOINTMENTS_FILE = "data/patient_appointments.json"

class AppointmentService:
    """Handles reading, writing, and checking appointment data."""

    def _load_appointments(self) -> list:
        """Loads all appointments from the JSON file."""
        try:
            with open(APPOINTMENTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("appointments", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_appointments(self, appointments: list):
        """Saves the list of appointments back to the JSON file."""
        with open(APPOINTMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"appointments": appointments}, f, indent=2, ensure_ascii=False)

    def is_slot_available(self, doctor_id: str, appointment_time: datetime) -> bool:
        """Checks if a specific time slot is available for a given doctor."""
        appointments = self._load_appointments()
        for appt in appointments:
            # Check for appointments with the same doctor at the exact same time
            if appt['doctor_id'] == doctor_id and datetime.fromisoformat(appt['appointment_datetime']) == appointment_time:
                logger.warning(f"Time slot {appointment_time} is already booked with doctor {doctor_id}.")
                return False
        return True

    def book_appointment(self, patient_id: str, doctor_id: str, appointment_time: datetime) -> bool:
        """Books a new appointment and saves it."""
        if not self.is_slot_available(doctor_id, appointment_time):
            return False
        
        appointments = self._load_appointments()
        new_appointment = {
            "appointment_id": f"APP_{int(datetime.now().timestamp())}",
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "appointment_datetime": appointment_time.isoformat(),
            "status": "confirmed"
        }
        appointments.append(new_appointment)
        self._save_appointments(appointments)
        logger.info(f"Successfully booked appointment: {new_appointment}")
        return True
