import asyncio
import os
import sys
from time import sleep
# from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field
from pyftg.socket.aio.gateway import Gateway
from src.test.DQNAgentTest import DQNAgent
async def run_test(game_num: int, port: int):
    character = "ZEN"
    host = os.environ.get("SERVER_HOST", "127.0.0.1")
    ai_name = "DQNAI"
    gateway = Gateway(host, port)
    blind_ai = DQNAgent(gateway=gateway)
    p2 = "MctsAi23i"
    gateway.register_ai(ai_name, blind_ai)
    await gateway.run_game([character, character], [ai_name, p2], game_num)
