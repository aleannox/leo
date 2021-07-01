import json
import logging
import socket
import time

import requests


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
        logging.info("Connecting to octoprint API.")
        connection_result = TankController._api_post(
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
        assert connection_result.status_code == 204, "Connection error."
        logging.info("Connected to octoprint API.")
        
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
        try:
            return result
        except json.JSONDecodeError:
            pass
