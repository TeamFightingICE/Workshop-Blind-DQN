"""
A simple version of Deep Q-Network(DQN) including the main tactics mentioned in DeepMind's original paper:
- Experience Replay
- Target Network
To play CartPole-v0.
> Note: DQN can only handle discrete-env which have a discrete action space, like up, down, left, right.
        As for the CartPole-v0 environment, its state(the agent's observation) is a 1-D vector not a 3-D image like
        Atari, so in that simple example, there is no need to use the convolutional layer, just fully-connected layer.
Using:
TensorFlow 2.17.0
Numpy 1.17.2
tf-keras~=2.16

original code is from here:
- https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning/blob/master/01_dqn.py
- applied to FightingICE by Yoshina Takano from Ritsumeikan University
- applied to DareFightingICE Blind AI by Nguyen Van Thai from Ritsumeikan University
- updated to DareFightingICE 7.0 by Nguyen Van Thai from Ritsumeikan University (2024/10/11)
"""

# plese check the code
# clipvalue=10.0
# how to give the reward at the end

# how to save and load the model
# # 重みの保存
# model.save_weights('./checkpoints/my_checkpoint')

# # 重みの復元
# model = Model()
# model.load_weights('./checkpoints/my_checkpoint')


# to avoiding memory leak
# 1
# model and memory initialize in def initialize(self, gameData, player):
# -> no effect
# 2
# refresh the model and optimizer with clean_session(only model, but not target model)
# the initial memory usage is 3.1~3.7GB
# -> good! but there is still memory leak. lets refresh also target model!
# --> the effect of the different refresh interval is nothing!
# -> after 1000 rounds trainings, the memory usage is 17.9GB (PC5810 cpu) (save interval = 10)
# --> drop box usage is 5.6 GB. python3 usage is 9.6 GB  
# -> drop box use 4.6GB after 800 rounds (when save interval == 1) (MyPC cpu)
# --> maybe python3 use about 9GB, because my PC memory is 16GB
# 3
# refresh model & target model & oprimizer with clean_session
# -> no obvious effect.
# 
# after 20 rounds, java use 1.6 GB, python3 use 2.5 GB, dropbox use 0.35 GB (My cpu)Machete
# after 46 rounds, java use 1.6 GB, python3 use 3.1 GB, dropbox use 0.41 GB (My cpu)
# after 73 rounds, java use 1.7 GB, python3 use 3.2 GB, dropbox use 0.43 GB (My cpu)
# after 290 rounds, java use 5.6 GB, python3 use 5.1 GB, dropbox use 1.2 GB (My cpu)
# after 355 rounds, java use 5.6 GB, python3 use 5.5 GB, dropbox use 1.6 GB (My cpu)
# after 420 rounds, java use 5.6 GB, python3 use 6.0 GB, dropbox use 1.9 GB (My cpu)
# after 600 rounds, java use 5.2 GB, python3 use None GB, dropbox use 3.0 GB (My cpu)
# -> (base) Exception in thread "Thread-2" py4j.Py4JException: Error while sending a command
# -> (tf2) /home/yoshina/Dropbox/Tensorflow2/train.sh: 8 行:  5543 強制終了

# after 42 rounds, java use 1.4 GB, python3 use 3.1 GB, dropbox use 0.39 GB (5810 cpu)Machete
# after 63 rounds, java use 1.2 GB, python3 use 3.3 GB, dropbox use 0.40 GB (5810 cpu)
# after 320 rounds, java use 1.2 GB, python3 use 5.1 GB, dropbox use 1.5 GB (5810 cpu)
# after 390 rounds, java use 1.3 GB, python3 use 5.7 GB, dropbox use 1.9 GB (5810 cpu)
# after 460 rounds, java use 1.1 GB, python3 use 6.2 GB, dropbox use 2.3 GB (5810 cpu)
# after 940 rounds, java use 1.2 GB, python3 use 9.7 GB, dropbox use 5.1 GB (5810 cpu)
# after 1000 rounds, java use 1.5 GB, python3 use None, dropbox use 5.5 GB (5810 cpu)
# -> (base) game over
# -> (tf2) WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. 

# after 230 rounds, java use 1.2 GB, python3 use 5.1 GB, dropbox use 1.1 GB (T3810 cpu)Machete
# after 280 rounds, java use 1.4 GB, python3 use 5.4 GB, dropbox use 1.4 GB (T3810 cpu)Machete
# after 350 rounds, java use 1.2 GB, python3 use 5.8 GB, dropbox use 1.8 GB (T3810 cpu)Machete
# after 800 rounds, java use 0.8 GB, python3 use 8.7 GB, dropbox use 4.3 GB (T3810 cpu)Machete
# after 837 rounds, java use None GB, python3 use 8.2 GB, dropbox use 4.7 GB (T3810 cpu)Machete
# -> (base) Java HotSpot(TM) 64-Bit Server VM warning: INFO: os::commit_memory(0x00000007a9980000, 198705152, 0) failed; error='メモリを確保できません' (errno=12)
# -> (tf2) py4j.protocol.Py4JError: An error occurred while calling t.runGame

# after 150 rounds, java use 1.3 GB, python3 use 6.0 GB, dropbox use 0.7 GB (T3810 cpu)MctsAi
# after 190 rounds, java use 1.5 GB, python3 use 6.5 GB, dropbox use 0.8 GB (T3810 cpu)MctsAi
# after 220 rounds, java use 1.4 GB, python3 use 7.5 GB, dropbox use 1.0 GB (T3810 cpu)MctsAi
# after 460 rounds, java use 2.0 GB, python3 use None GB, dropbox use 2.2 GB (T3810 cpu)MctsAi
# -> (base) Exception in thread "Thread-4" py4j.Py4JException: Error while sending a command
# -> (tf2) /home/icelab-pc3/Dropbox/Tensorflow2/train.sh: 8 行:  3752 強制終了 

# 4
# only clear only session without model & target model & optimizer. 
# -> not yet
# after 1 rounds, java use 1.5 GB, python3 use 0.38 GB, dropbox use 0.45 GB (T3810 cpu)Machete
# after 27 rounds, java use 1.3 GB, python3 use 3.3 GB, dropbox use 0.46 GB (T3810 cpu)Machete
# after 110 rounds, java use 1.3 GB, python3 use 6.0 GB, dropbox use 0.6 GB (T3810 cpu)Machete
# 5
# reconstruct only the model without clear_session()
# after 1 rounds, java use 1.4 GB, python3 use 0.3 GB, dropbox use 0.45 GB (T3810 cpu)Machete
# after 20 rounds, java use 2.4 GB, python3 use 2.6 GB, dropbox use 0.45 GB (T3810 cpu)Machete
# after 20 rounds, java use 2.1 GB, python3 use 12.1 GB, dropbox use 0.45 GB (T3810 cpu)Machete

# 7 train with tensorflow1.14
# after 1 rounds, java use 1.0 GB, python3 use 0.11 GB, dropbox use 0.3 GB (T3810 cpu)Machete

# 8
# define the input shape in Class Model
# -> not yet

# 9
# same hyperparameter to DQN_tf (memory size is different 50000->10000)
# -> tf-nightly

# 10
# same hyperparameter to DQN_tf
# -> tf-nightly

# 20
# same hyperparameter to DQN_tf
# maintain moving action 30
# 42 actions
# k.ckeal_session()
# -> tf-nightly

# 30
# same hyperparameter to DQN_tf
# maintain moving action 15
# 42 actions
# make 3 type (change reward)
# -> tf-nightly

# 40
# same hyperparameter to DQN_tf
# maintain moving action 15
# 42 actions
# make 3 type (change reward)
# use terminated reward
# -> tf-nightly


# future work
# 1, change the hyperparameter according to the deepmind baseline:
# https://github.com/deepmind/bsuite/tree/master/bsuite/baselines
# 2, why oiptimizer is changed ?
# 3, make readme file about configuration  
# -> done
# 4, should we remove old device driver for gpu? is there any difference between pc and Deel Learning Box 4?
# 5, why FTG/log is refresh per sec? due to run by tensorflow2? run the tensorflow1 in FTG4.5 
# -> due to run in the dropbox
# 6 print the memory usase of the each instance
# -> dont need because memory leak is due to the tensorflow
# 7, dont show the dropbox pop up
# -> done

from loguru import logger
from src.model import Model
from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.screen_data import ScreenData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult

import tensorflow as tf
import time
import numpy as np

import os
import csv
import shutil

import keras.api.optimizers as ko

# check the gpu
logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# check the seed
np.random.seed(1)
tf.random.set_seed(1)


# Neural Network Model Defined at Here.

class DQNAgent(AIInterface):

    def __init__(self, gateway):
        self.gateway = gateway
        # len(self.action_list) -> 42 
        self.action_list = ["STAND", "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP", "BACK_JUMP",
                            "STAND_GUARD",
                            "CROUCH_GUARD", "AIR_GUARD", "THROW_A", "THROW_B", "STAND_A", "STAND_B", "CROUCH_A",
                            "CROUCH_B", "AIR_A", "AIR_B", "AIR_DA", "AIR_DB", "STAND_FA", "STAND_FB", "CROUCH_FA",
                            "CROUCH_FB", "AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "STAND_D_DF_FA", "STAND_D_DF_FB",
                            "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_D_DB_BA", "STAND_D_DB_BB", "AIR_D_DF_FA",
                            "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB", "STAND_D_DF_FC"]
        # len(self.maintain_action_list) -> 8
        self.maintain_action_list = ["STAND", "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP",
                                     "BACK_JUMP"]
        self.continuous_action_list = ["FORWARD_WALK", "DASH", "CROUCH"]

        # argment for brain
        self.training = True
        self.dir_name = "DQN01"
        self.MAX_REWARD = 40  # or 120
        self.scaling_rewards = False
        self.save_interval = 2
        self.n_frame = 1  # Number of frames to be used as the current state

        # game mode
        self.mode = "score"

        # agent type
        self.type = 1  # balanced -> 1, offensive -> 2, defensive -> 3

        # additional reward
        self.additional_reward = False

        # final reward
        self.train_with_done = True
        self.give_0_in_terminated = True
        self.final_reward_type = 1  # 1->when only win,give hp_diff. 2 -> only lose. 3 -> both

        # maintain moving
        self.maintain_moveing = True
        self.start_maintain = False
        self.start_maintain_frame = None
        self.maintain_interval_frame = 15  # -> 0.25 sec

        # maintain moving
        self.maintain_moveing = True
        self.start_maintain = False
        self.start_maintain_frame = None
        self.maintain_interval_frame = 15  # -> 0.5 sec

        self.input_size = [2, 800 * self.n_frame]
        self.epsilon = 1.0  # e-greedy when exploring
        self.lr = 0.001  # learning rate
        self.epsilon_decay = 0.9998  # epsilon decay rate
        self.min_epsilon = 0.1  # minimum epsilon
        self.gamma = 0.9  # discount rate
        self.batch_size = 32  # batch_size
        self.target_update_iter = 300  # target network update period
        # self.train_nums = 500 not used in here   # total training steps
        self.num_in_buffer = 0  # transition's num in buffer
        self.buffer_size = 50000  # replay buffer size
        self.start_learning = self.batch_size  # step to begin learning(no update before that step)
        self.t = 1  # current time step t

        self.epsilon_decrees_steps = 10000
        self.epsilon_decrement = (self.epsilon - self.min_epsilon) / self.epsilon_decrees_steps
        self.linear_epsilon = True

        # model & optimizer
        self.model = Model(len(self.action_list))
        self.load_model()
        self.target_model = Model(len(self.action_list))
        logger.info("model id", id(self.model),
              id(self.target_model))  # to make sure the two models don't update simultaneously
        # gradient cilp
        opt = ko.Adam(learning_rate=self.lr, clipvalue=10.0)  # do gradient clip
        self.model.compile(optimizer=opt, loss="mse")  # use mean squared error (mse)

        # replay buffer params [(s, a, r, ns, done), ...]
        # s, a, r, ns, done are <class 'numpy.ndarray'> 
        self.obs = np.empty([self.buffer_size] + self.input_size)  ######### whether tuple
        self.actions = np.empty((self.buffer_size), dtype=np.int8)
        self.rewards = np.empty((self.buffer_size), dtype=np.float32)
        self.dones = np.empty((self.buffer_size), dtype=np.bool_)
        self.next_states = np.empty([self.buffer_size] + self.input_size)  ######### whether tuple
        self.next_idx = 0
        self.raw_audio_memory = None  # current state
        self.raw_audio_memory_pre = None
        self.audio_data = None  # audio data at the current frame
        self.n_round = 1
        self.nonDelay = None
        
    def close(self):
        pass

    def get_non_delay_frame_data(self, non_delay: FrameData):
        self.pre_framedata = self.nonDelay if self.nonDelay is not None else non_delay
        self.nonDelay = non_delay

    def get_information(self, frameData: FrameData, inControl: bool):
        # Load the frame data every time getInformation gets called
        self.frameData = frameData
        self.cc.set_frame_data(self.frameData, self.player)
        # self.nonDelay = nonDelay
        self.isControl = inControl
        self.currentFrameNum = self.frameData.current_frame_number # first frame is 14

    # please define this method when you use FightingICE version 3.20 or later
    def round_end(self, round_result: RoundResult):
        logger.info(round_result.remaining_hps[0])
        logger.info(round_result.remaining_hps[1])
        logger.info(round_result.elapsed_frame)

        if self.training:

            if self.train_with_done:
                reward = self.make_final_reward()

                state_ = self.raw_audio_memory

                self.store_transition(self.state, self.action_index, reward, state_, True)
                self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

                # trainging
                # but we dont train in roundEnd to avoid unexpected error
                # if self.t > self.start_learning:  # start learning
                #     losses = self.train_step()
                # logger.info("losses:", losses)

                # update target network
                # if self.t % self.target_update_iter == 0:
                #     self.update_target_model()

                # self.t = self.t + 1
                # self.e_decay()

            self.processing_for_log()
            self.save_result()

            # save and load the model to avoid memory leak
            # if self.n_round % self.save_interval == 0:
            self.save_model()
            self.n_round += 1
        self.last_opp_hp = 400
        self.last_my_hp = 400
        self.nonDelay = None


    def make_final_reward(self):
        # final reward is 0, it mean that we give only imidiet reward
        if self.give_0_in_terminated:
            return 0

        if self.mode == "score":
            # get hp difference
            hp_diff = self.nonDelay.get_character(self.player).hp - self.nonDelay.get_character(not self.player).hp

            if self.final_reward_type == 3:
                return hp_diff

            # which is win?
            if hp_diff >= 0:
                win = True
            else:
                win = False

            if self.final_reward_type == 1:
                if win:
                    return hp_diff
                else:
                    return 0
            elif self.final_reward_type == 2:
                if win:
                    return 0
                else:
                    return hp_diff

        elif self.mode == "hp":
            return 0

    # save the model
    def save_model(self):
        model_path = "./" + self.dir_name + "/checkpoints/" + str(self.n_round) + ".weights.h5"
        self.model.save_weights(model_path)
        self.model.save(model_path)
        logger.info("saved the model:", model_path)

    def load_model(self):
        try:
            path_to_checkpoint = "./" + self.dir_name + "/checkpoints/"
            # path_to_checkpoint = "./NetworkContainer/" + restore_model_dir + "/"
            ckpt = tf.train.get_checkpoint_state(path_to_checkpoint)
            if ckpt:  # checkpointがある場合
                logger.info("found the check point")

                # choose the type
                model_path = ckpt.model_checkpoint_path  # 最後のcheckpointからロード
                # model_path = random.choice(ckpt.all_model_checkpoint_paths) #checkpointにあるmodelからランダムに選択
                # model_path = ckpt.all_model_checkpoint_paths[round_mum-1] # モデルを順番にロード

                self.model.load_weights(model_path)
                logger.info("load the model from:", model_path)
            else:
                logger.info("There is no check point")
        except Exception as e:
            logger.info(e)

    # please define this method when you use FightingICE version 4.00 or later
    def get_screen_data(self, screen_data: ScreenData):
        pass

    def make_result_file(self):
        # firt, if there is a directry, remove it
        # if os.path.exists("./" + self.dir_name):
        #     shutil.rmtree("./" + self.dir_name)
        #     logger.info("remove the directry")

        # make the directry
        os.makedirs("./" + self.dir_name + "/checkpoints", exist_ok=True)
        logger.info("make dir")

        csvList = []
        csvList.append("Round Number(EPOCH)")  # from environment
        csvList.append("Steps(t)")  # from brain
        csvList.append("Epsilon")  # from brain
        csvList.append("myHp")  # from environment
        csvList.append("oppHp")  # from environment
        csvList.append("score")  # from environment
        csvList.append("time_terminated")  # from environment
        csvList.append("win")  # from environment
        f = open("./" + self.dir_name + "/matchResult.csv", 'a')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(csvList)
        f.close()

    def make_readme_file(self):
        path_w = "./" + self.dir_name + "/readme.txt"
        with open(path_w, mode='w') as f:
            f.write("target_update_iter: " + str(self.target_update_iter))
            f.write("\n")
            f.write("buffer_size: " + str(self.buffer_size))
            f.write("\n")
            f.write("batch_size: " + str(self.batch_size))
            f.write("\n")
            f.write("training: " + str(self.training))
            f.write("\n")
            f.write("epsilon: " + str(self.epsilon))
            f.write("\n")
            f.write("epsilon_decay: " + str(self.epsilon_decay))
            f.write("\n")
            f.write("min_epsilon: " + str(self.min_epsilon))
            f.write("\n")
            f.write("learning rate: " + str(self.lr))
            f.write("\n")
            f.write("gamma: " + str(self.gamma))
            f.write("\n")
            f.write("input_size: " + str(self.input_size))
            f.write("\n")
            f.write("train_with_done: " + str(self.train_with_done))
            f.write("\n")
            f.write("start_learning: " + str(self.start_learning))
            f.write("\n")
            f.write("save_interval: " + str(self.save_interval))

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = Key()
        self.frameData = FrameData()
        self.cc = CommandCenter()
        self.player = player  # p1 == True, p2 == False
        self.gameData = gameData
        self.isGameJustStarted = True
        self.last_opp_hp = 400
        self.nonDelay = None
        self.last_my_hp = 400

        if self.training:
            self.make_result_file()
            self.make_readme_file()

        # self.brain.restore_model_from_GN()

        return 0

    def input(self):
        return self.inputKey

    def reset_argments_at_initial_frame(self):
        # self.n_round = self.frameData.getRound()
        self.start_maintain = False
        self.start_maintain_frame = None

    def make_reward(self):
        offence_reward = self.last_opp_hp - self.nonDelay.get_character(not self.player).hp
        defence_reward = self.nonDelay.get_character(self.player).hp - self.last_my_hp

        # balanced
        if self.type == 1:
            reward = (0.5 * offence_reward) + (0.5 * defence_reward)
        # offensive
        elif self.type == 2:
            reward = (0.6 * offence_reward) + (0.4 * defence_reward)
        # defensive
        elif self.type == 3:
            reward = (0.4 * offence_reward) + (0.6 * defence_reward)

        if self.scaling_rewards:
            reward = reward / self.MAX_REWARD

        logger.info("reward:", reward)

        # give additional reward
        if self.additional_reward:
            if self.mode == "score":
                # balanced
                if self.type == 1:
                    pass

                # offensive
                elif self.type == 2:
                    pass

                # defensive
                elif self.type == 3:
                    pass

            elif self.mode == "hp":
                # balanced
                if self.type == 1:
                    pass

                # offensive
                elif self.type == 2:
                    pass

                # defensive
                elif self.type == 3:
                    pass

            # logger.info("additional_reward:", additional_reward)
            # reward = reward + additional_reward
            return reward

        else:
            return reward

    def set_last_hp(self):
        self.last_my_hp = self.nonDelay.get_character(self.player).hp
        self.last_opp_hp = self.nonDelay.get_character(not self.player).hp

    def play_action(self, state):
        # best_action is the index of the q_values (numpy.int64) 
        # q_values (numpy.ndarray)
        best_action, q_values = self.model.action_value(state[None])  # input the obs to the network model
        action_index = self.get_action(best_action)  # get the real action

        action_name = self.action_list[action_index]
        logger.info(action_name)
        self.cc.command_call(action_name)
        self.action_index = action_index

        if self.maintain_moveing:
            for s in self.maintain_action_list:
                if s == action_name:
                    self.start_maintain = True
                    self.start_maintain_frame = self.frameData.current_frame_number
                    self.moving_name = action_name

    # e-greedy
    def get_action(self, best_action):
        sample = np.random.rand()
        if sample < self.epsilon:
            logger.info(f"{sample} < {self.epsilon}")
            logger.info("random")
            return np.random.randint(0, len(self.action_list), dtype=np.int64)
        logger.info(f"{sample} >= {self.epsilon}")
        logger.info("model")
        return best_action

    def processing_for_log(self):
        if self.nonDelay.get_character(not self.player).hp == 0:
            self.score = 0
        else:
            self.score = (self.nonDelay.get_character(not self.player).hp / (
                    self.nonDelay.get_character(not self.player).hp + self.nonDelay.get_character(
                self.player).hp)) * 1000

        if self.nonDelay.get_character(self.player).hp == self.nonDelay.get_character(not self.player).hp:
            if self.training:
                self.win = 0
                logger.info("drow but lose")
            else:
                self.win = 1
                logger.info("drow but win")
        elif self.nonDelay.get_character(self.player).hp > self.nonDelay.get_character(not self.player).hp:
            self.win = 1
            logger.info("win")
        else:
            self.win = 0
            logger.info("lose")

    def save_result(self):
        csvList = []
        csvList.append(self.n_round)
        csvList.append(self.t)
        csvList.append(self.epsilon)
        csvList.append(abs(self.nonDelay.get_character(self.player).hp))
        csvList.append(abs(self.nonDelay.get_character(not self.player).hp))
        csvList.append(self.score)
        csvList.append(self.win)
        f = open("./" + self.dir_name + "/matchResult.csv", 'a')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(csvList)
        f.close()

    # store transitions into replay butter
    def store_transition(self, obs, action, reward, next_state, done):
        n_idx = self.next_idx % self.buffer_size
        self.obs[n_idx] = obs
        self.actions[n_idx] = action
        self.rewards[n_idx] = reward
        self.next_states[n_idx] = next_state
        self.dones[n_idx] = done
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    # train the model
    def train_step(self):
        # make the batch
        idxes = self.sample(self.batch_size)
        s_batch = self.obs[idxes]
        a_batch = self.actions[idxes]
        r_batch = self.rewards[idxes]
        ns_batch = self.next_states[idxes]
        done_batch = self.dones[idxes]

        target_q = r_batch + self.gamma * np.amax(self.get_target_value(ns_batch), axis=1) * (1 - done_batch)
        target_f = self.model.predict(s_batch)

        for i, val in enumerate(a_batch):
            target_f[i][val] = target_q[i]

        losses = self.model.train_on_batch(s_batch, target_f)

        return losses

    # assign the current network parameters to target network
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_target_value(self, obs):
        return self.target_model.predict(obs)

    # sample n different indexes
    def sample(self, n):
        assert n < self.num_in_buffer
        res = []
        while True:
            num = np.random.randint(0, self.num_in_buffer)
            if num not in res:
                res.append(num)
            if len(res) == n:
                break

        return res

    # decrece the epsilon
    def e_decay(self):
        if self.linear_epsilon:
            self.epsilon = max(self.epsilon - self.epsilon_decrement, self.min_epsilon)
        else:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        # logger.info("epsilon", self.epsilon)

    def isMoving(self):
        if self.start_maintain:
            current_frame_num = self.frameData.current_frame_number
            if self.maintain_interval_frame <= (current_frame_num - self.start_maintain_frame):
                self.start_maintain = False
                return False
            else:
                for continuous_action in self.continuous_action_list:
                    if self.moving_name == continuous_action:
                        self.cc.command_call(self.moving_name)
                        return True
        else:
            return False

    def processing(self):
        # First we check whether we are at the end of the round
        if self.frameData.empty_flag or self.frameData.current_frame_number <= 0:
            self.isGameJustStarted = True
            return
        if not self.isGameJustStarted:
            pass
        else:
            # initialize the argment at 1st frame of round
            self.isGameJustStarted = False
            self.reset_argments_at_initial_frame()

        if self.cc.get_skill_flag():
            self.inputKey = self.cc.get_skill_key()

            return
        self.inputKey.empty()
        self.cc.skill_cancel()

        # --------------this is main process---------------     
        logger.info(f"Current frame num {self.currentFrameNum} {self.nonDelay.current_frame_number}")
        if self.currentFrameNum == 1:
            # state_ = self.get_observation()
            state_ = self.raw_audio_memory
            self.set_last_hp()
            self.play_action(state_)
            self.state = state_

        elif self.maintain_moveing and self.isMoving():
            # do nothing
            logger.info("maintaining moving")

        elif self.isControl:
            reward = self.make_reward()
            # state_ = self.get_observation()
            state_ = self.raw_audio_memory

            self.store_transition(self.state, self.action_index, reward, state_, False)
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

            # trainging
            if self.t > self.start_learning:  # start learning
                losses = self.train_step()
                logger.info("losses:", losses)

            # update target network
            if self.t % self.target_update_iter == 0:
                self.update_target_model()

            # Up to here is one stepc
            self.t = self.t + 1
            self.e_decay()
            self.set_last_hp()
            self.play_action(state_)
            self.state = state_
            # logger.info("memory counter", self.brain.memory_counter)

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
            self.raw_audio_memory_pre = self.raw_audio_memory.copy()
            self.raw_audio_memory = np.hstack((raw_audio, self.raw_audio_memory))
            # self.raw_audio_memory = self.raw_audio_memory[:4, :, :]
            self.raw_audio_memory = self.raw_audio_memory[:, :800 * self.n_frame]

        # append so that audio memory has the first shape of n_frame
        increase = (800 * self.n_frame - self.raw_audio_memory.shape[1]) // 800
        for _ in range(increase):
            self.raw_audio_memory = np.hstack((np.zeros((2, 800)), self.raw_audio_memory))
        pass

    def game_end(self):
        pass

    def name(self) -> str:
        return self.__class__.__name__
    
    def is_blind(self) -> bool:
        return False