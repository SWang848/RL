import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import numpy as np

class MoMarios():

    def __init__(self, life_done, single_stage, n_obj):
        self.env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-1-1-v0'), SIMPLE_MOVEMENT)

        self.reward = 0
        self.lives = 3
        self.coin = 0
        self.x_pos = 40
        self.time = 0
        self.score = 0
        self.stage_bonus = 0

        self.life_done = life_done
        self.single_stage = single_stage
        
        self.n_obj = n_obj

        self.reset()

    def reset(self):
        self.reward = 0
        self.lives = 3
        self.coin = 0
        self.x_pos = 0
        self.time = 0
        self.score = 0
        self.stage_bonus = 0
        self.m_reward = np.zeros(self.n_obj)
        state = self.env.reset()

        return state

    def step(self, action:int):
        state, reward, done, info = self.env.step(action)
        xpos_r = info["x_pos"] - self.x_pos
        self.x_pos = info["x_pos"]

        if xpos_r < -5:
            xpos_r = 0
        
        time_r = info["time"] - self.time
        self.time = info["time"]
        if time_r > 0:
            time_r = 0
        
        if done == True and info['flag_get'] == False:
            death_r = -15
        
        score_r = (info['score'] - self.score) / 100
        self.score = info['score']

        #no flag_get score 
        if score_r == 8:
            score_r = 0

        if score_r < 0:
            score_r = 0
        
        m_reward = np.zeros(self.n_obj)
        if self.n_obj == 2:
            m_reward[0] = reward
            m_reward[1] = score_r + death_r + xpos_r

        # if self.single_stage and info["flag_get"]:
        #     self.stage_bonus = 10000
        #     done = True
        
        # if self.life_done:
        #     if self.lives > info['life'] and info['life'] > 0:
        #         force_done = True
        #         self.lives = info['life']
        #     else:
        #         force_done = done
        #         self.lives = info['life']
        # else:
        #     force_done = done

        # xpos_r = info["x_pos"] - self.x_pos
        # self.x_pos = info["x_pos"]
        # # if mario dies, positions gap will be above 5
        # if xpos_r < -5:
        #     xpos_r = 0
        
        # time_r = info["time"] - self.time
        # self.time = info["time"]
        # # time always decreasing
        # if time_r > 0:
        #     time_r = 0
        
        # if self.lives-1 > info["life"]:
        #     death_r = -15
        #     self.lives -= 1
        # else:
        #     death_r = 0

        # coin_r = 100 * (info["coins"] - self.coin)
        # self.coin = info["coins"]

        # enemy_r = info["score"] - self.score
        # if coin_r > 0 or done:
        #     enemy_r = 0
        # self.score = info['score']

        # m_reward = np.zeros(self.n_obj)
        # if self.n_obj == 2:
        #     m_reward[0] = time_r
        #     m_reward[1] = info['score']
        # elif self.n_obj == 5:
        #     m_reward[0] = xpos_r
        #     m_reward[1] = time_r
        #     m_reward[2] = death_r
        #     m_reward[3] = coin_r
        #     m_reward[4] = enemy_r
        
        return state, m_reward, done, info

    def render(self):
        self.env.render()

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # done = True
    # for step in range(50000):
    #     if done:
    #         state = env.reset()
    #     state, reward, done, info = env.step(env.action_space.sample())
    #     print(reward, done, info)
    #     env.render()

    # env.close()

    test = MoMarios(True, True, 5)
    done = True
    for step in range(500):
        if done:
            state = test.reset()
        state, reward, done, info = test.step(env.action_space.sample(), True)
        print(reward, done, info)
        test.render()


