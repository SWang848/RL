import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import numpy as np

class MoMarios():

    def __init__(self, is_render, life_done, singel_stage):
        self.env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-v0'), SIMPLE_MOVEMENT)

        self.is_render = is_render
        self.steps = 0
        self.episode = 0
        self.reward = 0
        self.lives = 3
        self.coin = 0
        self.x_pos = 0
        self.time = 0
        self.score = 0
        self.stage_bonus = 0

        self.life_done = life_done
        self.single_stage = singel_stage
        
        self.n_obj = 5

        self.reset()

    def reset(self):
        self.steps = 0
        self.episode += 1
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

    def step(self, action:int, frame_skip):
        state, reward, done, info = self.env.step(action)
        m_reward = np.zeros(self.n_obj)

        if self.single_stage and info["flag_get"]:
            self.stage_bonus = 10000
            done = True
        
        # if self.life_done:
        #     if self.lives > info['life'] and info['life'] > 0:
        #         force_done = True
        #         self.lives = info['life']
        #     else:
        #         force_done = done
        #         self.lives = info['life']
        # else:
        #     force_done = done

        xpos_r = info["x_pos"] - self.x_pos
        self.x_pos = info["x_pos"]
        # if mario dies, positions gap will be above 5
        if xpos_r < -5:
            xpos_r = 0
        
        time_r = info["time"] - self.time
        self.time = info["time"]
        # time always decreasing
        if time_r > 0:
            time_r = 0
        
        if self.lives-1 > info["life"]:
            death_r = -15
            self.lives -= 1
        else:
            death_r = 0

        coin_r = 100 * (info["coins"] - self.coin)
        self.coin = info["coins"]

        enemy_r = info["score"] - self.score
        if coin_r > 0 or done:
            enemy_r = 0
        self.score = info['score']
        
        m_reward[0] = xpos_r
        m_reward[1] = time_r
        m_reward[2] = death_r
        m_reward[3] = coin_r
        m_reward[4] = enemy_r
        
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

    test = MoMarios(True, True, True)
    done = True
    for step in range(5000):
        if done:
            state = test.reset()
        state, reward, done, info = test.step(env.action_space.sample(), True)
        print(reward, done, info)
        test.render()


