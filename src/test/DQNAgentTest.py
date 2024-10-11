from loguru import logger
from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.screen_data import ScreenData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult


import tensorflow as tf
import keras
import time
import numpy as np

import keras.api.backend as k
import keras.api.layers as kl
import keras.api.activations as ka
import keras.api.optimizers as ko

import os
import csv
import shutil
from src.model import Model

# check the gpu
logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# check the seed
np.random.seed(1)
tf.random.set_seed(1)

class DQNAgent(AIInterface):
    def __init__(self, gateway):
        self.action_list = ["STAND", "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP", "BACK_JUMP", "STAND_GUARD",
                            "CROUCH_GUARD", "AIR_GUARD", "THROW_A", "THROW_B", "STAND_A", "STAND_B", "CROUCH_A",
                            "CROUCH_B", "AIR_A", "AIR_B", "AIR_DA", "AIR_DB", "STAND_FA", "STAND_FB", "CROUCH_FA",
                            "CROUCH_FB", "AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "STAND_D_DF_FA", "STAND_D_DF_FB",
                            "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_D_DB_BA", "STAND_D_DB_BB", "AIR_D_DF_FA",
                            "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB", "STAND_D_DF_FC"]
        try:
            # self.model = keras.models.load_model("DQN01/checkpoints/2.weights.h5")
            self.model = Model(len(self.action_list))
            self.model.load_weights("DQN01/checkpoints/2.weights.h5")
            # self.model = keras.models.load_model("D:\Research\FTG4.50\python\DQNAgent")
            logger.info("Loaded model:", self.model)

        except:
            logger.info("Can't Load")

        self.gateway = gateway
        self.raw_audio_memory = None
        self.audio_data = None
        self.n_frame = 1

    def close(self):
        pass
        
    def get_information(self, frameData: FrameData, isControl: bool):
        # Getting the frame data of the current frame
        self.frameData = frameData
        self.cc.set_frame_data(self.frameData, self.player)
    

    # please define this method when you use FightingICE version 3.20 or later
    def round_end(self, round_result: RoundResult):
        logger.info(round_result.remaining_hps[0])
        logger.info(round_result.remaining_hps[1])
        logger.info(round_result.elapsed_frame)


    # please define this method when you use FightingICE version 4.00 or later
    def get_screen_data(self, sd: ScreenData):
        pass


    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things        
        self.inputKey = Key()
        self.frameData = FrameData()
        self.cc = CommandCenter()
            
        self.player = player
        self.gameData = gameData
        return 0


    def predict(self, state):
        q_values = self.model.predict(state[None])
        best_action = np.argmax(q_values, axis=-1)
        logger.info("best_action: ",best_action)
        action_name = self.action_list[best_action[0]]
        logger.info(action_name)
        return action_name

    
    def input(self):
        # Return the input for the current frame
        return self.inputKey
        
    def processing(self):
        # Just compute the input for the current frame
        if self.frameData.empty_flag or self.frameData.current_frame_number <= 0:
                self.isGameJustStarted = True
                return
                
        if self.cc.get_skill_flag():
                self.inputKey = self.cc.get_skill_key()
                return
            
        self.inputKey.empty()
        self.cc.skill_cancel()     

        state = self.raw_audio_memory
        action = self.predict(state)
        # logger.info("action: ",action)
        self.cc.command_call(action)
        # self.cc.commandCall("B")

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data
        # process audio
        try:
            byte_data = self.audio_data.raw_data_bytes
            np_array = np.frombuffer(byte_data, dtype=np.float32)
            raw_audio = np_array.reshape((2, 1024))
            # raw_audio = raw_audio.T
            raw_audio = raw_audio[:, :800]
            # raw_audio = raw_audio.T
        except Exception as ex:
            raw_audio = np.zeros((2, 800))
        if self.raw_audio_memory is None:
            # self.logger.info('raw_audio_memory none {}'.format(raw_audio.shape))
            self.raw_audio_memory = raw_audio
        else:
            self.raw_audio_memory = np.hstack((raw_audio, self.raw_audio_memory))
            # self.raw_audio_memory = self.raw_audio_memory[:4, :, :]
            self.raw_audio_memory = self.raw_audio_memory[:, :800 * self.n_frame]

        # append so that audio memory has the first shape of n_frame
        increase = (800 * self.n_frame - self.raw_audio_memory.shape[1]) // 800
        for _ in range(increase):
            self.raw_audio_memory = np.hstack((np.zeros((2, 800)), self.raw_audio_memory))

    def game_end(self):
        pass

    def name(self) -> str:
        return self.__class__.__name__
    
    def is_blind(self) -> bool:
        return True
    
    def get_non_delay_frame_data(self, non_delay: FrameData):
        pass