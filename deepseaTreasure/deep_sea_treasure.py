"""Deep sea treasure envirnment

mostly compatible with OpenAI gym envs
"""

from __future__ import print_function

import sys
import math
import random
import itertools

import numpy as np
import scipy.stats

from utils import truncated_mean, compute_angle, pareto_filter

SEA_MAP = [ 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [18, 26, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 44, 48.2, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 56, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, 72, 76.3, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, 90, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, 100, 0, 0]]

HOME_POS = (0, 0)

ACT_LEFT = 0
ACT_RIGHT = 1
ACT_UP = 2
ACT_DOWN = 3

ACTIONS = ["Left", "Right", "Up", "Down"]
ACTION_COUNT = len(ACTIONS)


class DeepSeaTreasure():
    """Deep sea treasure environment
    """

    def __init__(self, sea_map=SEA_MAP, full=False, view=(5,5), scale=1):
        assert view[0] % 2 == 1
        assert view[1] % 2 == 1
        self.fully_observable=full
        self.map_height = len(sea_map)
        self.map_width = len(sea_map[0])
        self.view_height = view[0]
        self.view_width = view[1]
        self.scale = scale
        self.unit = 255

        self.observe_height = self.map_height if full else view[0]
        self.observe_width = self.map_width if full else view[1]

        self.observation_space = np.zeros((self.observe_width*scale, self.observe_height*scale, 3))
        self.nb_count = ACTION_COUNT
        self.action_space = np.arange(ACTION_COUNT)
        
        self.action_rand = False

        self.submarine_pos = list(HOME_POS)
        self.sea_map = sea_map
        self._static_map = self._create_map(sea_map)
        self.game_map = self.get_blank_map()

        self.end = False
        self.obj_cnt = 2

        self._margined_width = self.map_width + self.view_width - 1
        self._margined_height = self.map_height + self.view_height - 1

        self._w_min = int(self.view_width/2)
        self._w_max = int(self.view_width/2 + self.map_width)
        self._h_min = int(self.view_height/2)
        self._h_max = int(self.view_height/2 + self.map_height)


    def _create_map(self, sea_map):
        game_map = np.zeros((self.map_height, self.map_width, 3))
        for row in range(self.map_height):
            for col in range(self.map_width):
                if sea_map[row][col] == -1:
                    game_map[row][col][0] = self.unit # seafloor
                elif sea_map[row][col] != 0:
                    game_map[row][col][1] = self.unit # treasure
                else:
                    game_map[row][col][2] = self.unit # water
        return game_map

    def get_blank_map(self):
        return np.copy(self._static_map)

    def step(self, action, frame_skip=1, incremental_frame_skip=True):
        """Perform the given action `frame_skip` times
         ["Left", "Right", "Up", "Down"]
        Arguments:
            action {int} -- Action to perform, ACT_MINE (0), ACT_LEFT (1), ACT_RIGHT (2), ACT_ACCEL (3), ACT_BRAKE (4) or ACT_NONE (5)

        Keyword Arguments:
            frame_skip {int} -- Repeat the action this many times (default: {1})
            incremental_frame_skip {bool} -- If True, frame_skip actions are performed in succession, otherwise the repeated actions are performed simultaneously (e.g., 4 accelerations are performed and then the cart moves).

        Returns:
            tuple -- (observation, reward, terminal, info) tuple
        """

        reward = np.zeros(self.obj_cnt)
        reward[-1] = -1

        if self.action_rand:
            prob = np.array([.05]*self.nb_count)
            prob[action] += 0.8
            action = np.random.choice(self.action_space, p=prob)

        move = (0, 0)
        if action == ACT_LEFT:
            move = (0, -1)
        elif action == ACT_RIGHT:
            move = (0, 1)
        elif action == ACT_UP:
            move = (-1, 0)
        elif action == ACT_DOWN:
            move = (1, 0)
        
        changed = self.move_submarine(move)
        reward[0] = self.sea_map[self.submarine_pos[0]][self.submarine_pos[1]]

        if changed:
            self.render()

        info, observation = self.get_state(True)
        # observation (pixels), reward (list), done (boolean), info (dict)
        return observation, reward, self.end, info

    def move_submarine(self, move):
        target_y = self.submarine_pos[0] + move[0]
        target_x = self.submarine_pos[1] + move[1]
        
        if 0 <= target_x < self.map_width and 0 <= target_y < self.map_height:
            if self.game_map[target_y][target_x][0] != self.unit:
                self.submarine_pos[0] = target_y
                self.submarine_pos[1] = target_x
            
            if self.game_map[target_y][target_x][1] == self.unit:
                self.end = True
            return True
        return False


    def get_state(self, update=True):
        """Returns the environment's full state, including the cart's position,
        its speed, its orientation and its content, as well as the environment's
        pixels

        Keyword Arguments:
            update {bool} -- Whether to update the representation (default: {True})

        Returns:
            dict -- dict containing the aforementioned elements
        """

        info = {
            "position": self.submarine_pos,
            "pixels": self.game_map
        }

        observation = self.get_observation()

        return info, observation

    def get_observation(self):
        """Create a partially observable observation with the given state. 
        Half size of state["pixels"] with origin of state["position"], and the 
        overflow part is black
        
        Returns:
            array: 3d array as the same type of state["pixels"]
        """

        if self.fully_observable:
            self.observation_space = np.copy(self.game_map)
            return self.scale_map(self.game_map)

        margined_map = np.zeros((self._margined_height, self._margined_width, 3), dtype=np.uint8)
        margined_map[self._w_min : self._w_max, self._h_min : self._h_max, :] = self.game_map

        min_y = self.submarine_pos[0]
        min_x = self.submarine_pos[1]

        observe = margined_map[min_y : int(min_y + self.observe_height), min_x : int(min_x + self.observe_width), :]
        
        self.observation_space = self.scale_map(observe)
        return self.observation_space

    def reset(self):
        """Resets the environment to the start state

        Returns:
            [type] -- [description]
        """

        self.end = False
        self.submarine_pos = list(HOME_POS)
        self.render()
        _, observation = self.get_state()
        return observation

    def __str__(self):
        string = "Completed: {} ".format(self.end)
        string += "Position: {} ".format(self.submarine_pos)
        return string

    def render(self):
        """Update the environment's representation
        """

        self.game_map = self.get_blank_map()
        self.game_map[self.submarine_pos[0]][self.submarine_pos[1]] = self.unit

    def scale_map(self, frame):
        return frame.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

    def show(self, frame=None, plot=False):
        frame = frame or self.observation_space

        if not plot:
            for i in range(frame.shape[0]):
                print(frame[i,:,0], frame[i,:,1], frame[i,:,2])
        else:
            import matplotlib.pyplot as plt
            image = frame
            fig = plt.imshow(image)
            plt.show()


if __name__ == '__main__':

    env = DeepSeaTreasure(view=(5,5), full=False, scale=1)
    print(env.nb_count)
    print(env.observation_space.shape)
    print(env.action_space.shape)
    o_t = env.reset()
    terminal = False

    env.show(plot=True)
    
    while not terminal:
        a_t = int(input("0:left, 1:right, 2:up, 3:down | choose: "))
        o_t1, r_t, terminal, s_t1 = env.step(a_t)
        print(o_t1.shape)
        env.show(plot=True)

        o_t = o_t1
        print("Taking action", a_t, "at", s_t1["position"], "with reward", r_t)
    env.reset()