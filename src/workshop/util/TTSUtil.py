import os
import logging
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play # pydub's play function uses simpleaudio or pyaudio

# Get the dedicated app_activity logger to record TTS actions
logger = logging.getLogger('app_activity')

def speak_text(text: str, lang: str = 'vi', slow: bool = False):
    """Converts text to speech using Google TTS and plays the audio using pydub/simpleaudio."""
    if not text.strip():
        logger.info("No text to speak.")
        return

    temp_audio_file = "temp_speech.mp3"
    try:
        # Create a gTTS object
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        # Save the audio to a temporary file
        tts.save(temp_audio_file)
        logger.info(f"Generated speech for text (first 30 chars): '{text[:30]}...'")
        
        # Load and play the audio file using pydub's play function
        audio = AudioSegment.from_mp3(temp_audio_file)
        play(audio)
        logger.info("Played speech audio.")
        
    except Exception as e:
        # Errors from here should go to the system logger
        system_logger = logging.getLogger(__name__) # Get the root logger for errors
        system_logger.error(f"Error during text-to-speech or playing audio: {e}", exc_info=True)
    finally:
        # Clean up the temporary audio file
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
            logger.debug(f"Removed temporary audio file: {temp_audio_file}")
