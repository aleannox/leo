"""Speak using text to speech or prerecorded phrases.

Prerecorded phrases are collected from subfolders of ./phrases.
If ./phrases/<phrase_name> contains several files, one is played at random
each time the phrase is used.
"""

import pathlib
import random

from loguru import logger
import pydub
import pydub.playback
import pyttsx3


# Only for text to speech.
VOICE_ID = 2
RATE = 150
VOLUME = 1

# For prerecorded phrases.
PHRASES_PATH = pathlib.Path(__file__).parent / 'phrases'
PHRASES_FILE_PATTERN = '*.[wWmMaA][aApPiI][vV3fF]'  # .wav, .mp3, .aif


class Speech:
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[VOICE_ID].id)
        self.engine.setProperty('rate', RATE)
        self.phrases = {}
        for phrase_dir in PHRASES_PATH.iterdir():
            if phrase_dir.is_dir():
                self.phrases[phrase_dir.name] = list(
                    phrase_dir.glob(PHRASES_FILE_PATTERN)
                )
        logger.info(
            f'I have found the following prerecorded phrases: {self.phrases}'
        )

    def say_phrase(self, phrase):
        file_to_play = random.sample(self.phrases[phrase], 1)[0]
        logger.info(f"Playing: {file_to_play}")
        pydub.playback.play(
            pydub.AudioSegment.from_wav(file_to_play)
        )

    def say(self, text):
        logger.info(f"Saying: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
