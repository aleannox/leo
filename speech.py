from loguru import logger
import pyttsx3


VOICE_ID = 2
RATE = 150
VOLUME = 1


PHRASES = {
    'human': 'Hello human, how are you today.',
    'default': 'I don\'t know what to say.',
}


class Speech:
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[VOICE_ID].id)
        self.engine.setProperty('rate', RATE)

    def say_phrase(self, phrase):
        text = PHRASES.get(phrase, PHRASES['default'])
        self.say(text)

    def say(self, text):
        logger.info(f"Saying: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
