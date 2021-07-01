import json
import logging
import pathlib
import random
import socket
import time

import requests
import ruamel.yaml

yaml = ruamel.yaml.YAML()


class TankController:
    API_HEADER = {'X-Api-Key': 'A3003FAA8A6748CDA5ADCBC54A44244E'}
    if socket.gethostname() == 'octopi':
        # Use localhost if on octopi because octopi.local may not be available
        # if octopi is not connected to WiFi
        URL_BASE = 'http://localhost/api/'
    else:
        URL_BASE = 'http://octopi.local/api/'
    
    def __init__(self):
        TankController._wait_for_api()
        TankController._connect_api()
        self._init_chains()
        self.config = yaml.load(pathlib.Path('config.yaml'))
        logging.info(f"Using config: {self.config}")
        
    def run_random(self):
        while True:
            chain_target = random.randint(
                self.config['chain_min'], self.config['chain_max']
            )
            duration = self.move_chains(chain_target, self.config['chain_speed'])
            time.sleep(duration + self.config['time_scale'])
            
    def move_chains(self, position, speed):
        logging.info(f"Moving chains to E{position} with F{speed}.")
        TankController._send_gcode(f'G0 E{position} F{speed}')
        duration = abs(position - self.chain_position) / 1000 * self.config['chain_time_per_1000']
        logging.info(f"Expected move duration {duration:.01f}s")
        self.chain_position = position
        return duration
        
    def _init_chains(self):
        logging.info("Initializing chains.")
        TankController._send_gcode(
            [
                'M163 S0 P0.5',
                'M163 S1 P0.5',
                'M164 S3',
                'T3',
                'G92 E0',
            ]
        )
        self.chain_position = 0
        
    @staticmethod
    def _send_gcode(commands):
        if isinstance(commands, str):
            commands = [commands]
        logging.info(f"Sending GCODE commands: {commands}")
        result = TankController._api_post(
            'printer/command', json_={'commands': commands}
        )
        assert result.status_code == 204, f"Sent wrong GCODE command {commands}."
        
    @staticmethod
    def _wait_for_api():
        logging.info("Waiting for octoprint API to become alive.")
        while True:
            try:
                TankController._api_get('version')
                break
            except (socket.gaierror, requests.ConnectionError):
                pass
            time.sleep(10)
            
    @staticmethod
    def _connect_api():
        logging.info("Connecting octoprint API to printer.")
        result = TankController._api_post(
            'connection',
            json_={
                'command': 'connect',
                'port': '/dev/ttyACM0',
                'baudrate': 250000,
                'printerProfile': '_default',
                'save': True,
                'autoconnect': True,
            }
        )
        assert result.status_code == 204, "Connection error."
        logging.info("Connected octoprint API to printer.")
        
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
