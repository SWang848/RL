from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(500):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    print(info)
    env.render()

# print(env.reset().shape)
# for i in range(5000):
#     state, reward, done, info = env.step(env.action_space.sample())
#     print(reward, done, info)
#     if done == True:
#         env.reset()
#     cv2.imshow('as', state)
#     cv2.waitKey(0)
# # env.render()
# env.close()