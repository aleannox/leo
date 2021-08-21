import json
import pathlib
import random
import socket
import time

from loguru import logger
import requests
import ruamel.yaml

import speech
import vision

yaml = ruamel.yaml.YAML()
logger.add(pathlib.Path(__file__).parent / 'run.log', rotation='100 MB')


class TankController:
    API_HEADER = {'X-Api-Key': 'A3003FAA8A6748CDA5ADCBC54A44244E'}
    if socket.gethostname() == 'octopi':
        # Use localhost if on octopi because octopi.local may not be available
        # if octopi is not connected to WiFi
        URL_BASE = 'http://localhost/api/'
    else:
        URL_BASE = 'http://octopi.local/api/'
    
    def __init__(self, dry_run):
        self.dry_run = dry_run
        if self.dry_run:
            logger.info("Dry run, skipping api connection.")
        else:
            TankController._wait_for_api()
            TankController._connect_api()
        self._init_gun()
        self._init_turret()
        self._init_chains()
        self.config = yaml.load(pathlib.Path(__file__).parent / 'config.yaml')
        logger.info(f"Using config: {self.config}")
        self.person_detector = vision.PersonDetector()
        self.speech = speech.Speech()

    def run_look_at_person(self):
        while True:
            person_x = self.person_detector.get_largest_person_relative_x()
            self.move_gun()
            if person_x:
                self.speech.say_phrase('human')
                # 1000 corresponds to 90 degrees
                # we use a linear scale such that
                # person_x = +-0.5 -> +-45 degrees -> +-500
                # recall that positive = ccw for chains, need sign flip
                chain_target = round(
                    max(
                        self.config['chain_min'],
                        min(
                            self.config['chain_max'],
                            self.chain_position - person_x * 1000,
                        ),
                    )
                )
                self.move_chains(chain_target, self.config['chain_speed'])
                self.move_gun()
            elif (
                time.time()
                - self.person_detector.camera_last_seen
                > self.config['wait_for_camera_time']
            ):
                self._random_move()
                time.sleep(self.config['time_scale'])
        
    def run_random(self):
        while True:
            self._random_move()
            time.sleep(self.config['time_scale'])

    def _random_move(self):
        self.move_gun()
        chain_target = random.randint(
            self.config['chain_min'], self.config['chain_max']
        )
        self.move_chains(chain_target, self.config['chain_speed'])
        self.move_gun()
            
    def move_chains(self, position, speed):
        logger.info(f"Moving chains to E{position} with F{speed}.")
        self._send_gcode(f'G0 E{position} F{speed}')
        # Expected duration for this move, we sleep for this duration.
        # Currently we do not know of any way to query whether the printer is actually moving.
        duration = abs(position - self.chain_position) / 1000 * self.config['chain_time_per_1000']
        logger.info(f"Expected move duration {duration:.01f}s")
        time.sleep(duration)
        self.chain_position = position
    
    def move_gun(self):
        return  # Skip for now.
        # Gun movement is so slow that it is barely visible.
        # We only use it to add sound-flavor to the other movements.
        # For this a self-contained random movement is sufficient.
        # Clip gun movement so it does not take too long.
        diff = random.randint(
            -self.config['gun_max_per_move'], self.config['gun_max_per_move']
        )
        target = max(
            min(
                self.gun_position + diff,
                self.config['gun_max']
            ),
            self.config['gun_min']
        )
        diff = target - self.gun_position
        duration = abs(diff) / 1000 * self.config['gun_time_per_1000']
        logger.info(
            f"Moving gun to E{target} with F{self.config['gun_speed']}."
        )
        self._send_gcode(f'G0 X{target} F{self.config["gun_speed"]}')
        logger.info(f"Expected move duration {duration:.01f}s")
        time.sleep(duration)
        self.gun_position = target
        
    def _init_gun(self):
        # TODO: homing, currently homing sensor is not at 0 but max ~= 2000
        logger.info("Initializing gun.")
        self._send_gcode('G92 X0')
        self.gun_position = 0

    def _init_turret(self):
        # just disable, hopefully preventing thermal runaway
        logger.info("Initializing turrent.")
        self._send_gcode('M18 Z')
        
    def _init_chains(self):
        logger.info("Initializing chains.")
        self._send_gcode(
            [
                'M163 S0 P0.5',
                'M163 S1 P0.5',
                'M164 S3',
                'T3',
                'G92 E0',
            ]
        )
        self.chain_position = 0

    def _send_gcode(self, commands):
        if isinstance(commands, str):
            commands = [commands]
        if self.dry_run:
            logger.info(f"[Dry run] Sending GCODE commands: {commands}")
        else:
            logger.info(f"Sending GCODE commands: {commands}")
            result = TankController._api_post(
                'printer/command', json_={'commands': commands}
            )
            assert result.status_code == 204, f"Sent wrong GCODE command {commands}."
        
    @staticmethod
    def _wait_for_api():
        logger.info("Waiting for octoprint API to become alive.")
        while True:
            try:
                TankController._api_get('version')
                break
            except (socket.gaierror, requests.ConnectionError):
                pass
            time.sleep(10)
            
    @staticmethod
    def _connect_api():
        logger.info("Connecting octoprint API to printer.")
        result = TankController._api_post(
            'connection',
            json_={
                "command": "connect",
                "port": "/dev/ttyACM0",
                "baudrate": 250000,
                "printerProfile": "_default",
                "save": True,
                "autoconnect": True,
            }
        )
        assert result.status_code == 204, "Connection error."
        logger.info("Connected octoprint API to printer.")
        
    @staticmethod
    def _api_get(url):
        result = requests.get(
            TankController.URL_BASE + url,
            headers=TankController.API_HEADER,
        )
        return result

    @staticmethod
    def _api_post(url, json_):
        result = requests.post(
            TankController.URL_BASE + url,
            headers=TankController.API_HEADER,
            json=json_,
        )
        return result
