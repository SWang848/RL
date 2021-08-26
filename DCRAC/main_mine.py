import sys
import os
import time
import numpy as np
from optparse import OptionParser
import random

import gym
from gym.spaces import Box

from minecart import Minecart
from agent_mine import DCRACSAgent, DCRACAgent, DCRACSEAgent, DCRAC0Agent, CNAgent, CN0Agent
from utils import mkdir_p, get_weights_from_json
from stats import rebuild_log, print_stats, compute_log

class PixelMinecart(gym.ObservationWrapper):

    def __init__(self, env):
        # don't actually display pygame on screen
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        super(PixelMinecart, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(480, 480, 3), dtype=np.uint8)

    def observation(self, obs):
        obs = self.render('rgb_array')
        return obs


AGENT_DICT = {
    "DCRAC": DCRACAgent,
    "DCRACS": DCRACSAgent,
    "DCRACSE": DCRACSEAgent,
    "DCRAC0": DCRAC0Agent, 
    "CN": CNAgent, 
    "CN0": CN0Agent, 
}

mkdir_p("output")
mkdir_p("output/logs")
mkdir_p("output/networks")
mkdir_p("output/pred")
mkdir_p("output/imgs")

parser = OptionParser()
parser.add_option("--seed", dest="seed", default=0)
parser.add_option("-a", "--agent", dest="agent", choices=["DCRAC", "DCRACS", "DCRACSE", "DCRAC0", "CN", "CN0"], default="DCRACS")
parser.add_option("-n", "--net-type", dest="net_type", choices=["R", "M", "F"], default="R", help="Agent architecture type: Recurrent, MemNN or FC")
parser.add_option("-r", "--replay", dest="replay", default="DER", choices=["STD", "DER"], help="Replay type, one of 'STD','DER'")
parser.add_option("-s", "--buffer-size", dest="buffer_size", default="100000", help="Replay buffer size", type=int)
parser.add_option("-m", "--memnn-size", dest="memnn_size", default="9", help="Memory network memory size", type=int)
parser.add_option("-d", "--dup", dest="dup", action="store_false", default=True, help="Extra training")
parser.add_option("-t", "--timesteps", dest="timesteps", default="10", help="Recurrent timesteps", type=int)
parser.add_option("-e", "--end_e", dest="end_e", default="0.05", help="Final epsilon value", type=float)
parser.add_option("-l", "--lr-c", dest="lr_c", default="0.02", help="Critic learning rate", type=float)
parser.add_option("-L", "--lr-a", dest="lr_a", default="0.25", help="Actor learning rate", type=float)
parser.add_option("--buffer-a", dest="buffer_a", default="2.", help="reply buffer error exponent", type=float)
parser.add_option("--buffer-e", dest="buffer_e", default="0.01", help="reply buffer error offset", type=float)
parser.add_option("-u", "--update-period", dest="updates", default="4", help="Update interval", type=int)
parser.add_option("-f", "--frame-skip", dest="frame_skip", default="4", help="Frame skip", type=int)
parser.add_option("-b", "--batch-size", dest="batch_size", default="64", help="Sample batch size", type=int)
parser.add_option("-g", "--discount", dest="discount", default="0.98", help="Discount factor", type=float)
parser.add_option("--anneal-steps", dest="steps", default="100000", help="steps",  type=int)
parser.add_option("-p", "--mode", dest="mode", choices=["regular", "sparse"], default="regular")
parser.add_option("-v", "--obj-func", dest="obj_func", choices=["a", "am", "td", "q", "y"], default="a")
parser.add_option("--no-action", dest="action_conc", action="store_false", default=True)
parser.add_option("--no-embd", dest="feature_embd", action="store_false", default=True)
parser.add_option("--gpu", dest="gpu_setting", choices=["1", "2", "3"], default="2", help="1 for CPU, 2 for GPU, 3 for CuDNN")
parser.add_option("--log-game", dest="log_game", action="store_true", default=False)
parser.add_option("--dst", dest="dst_view", choices=["3", "5"], default="5")

(options, args) = parser.parse_args()

hyper_info = "{}_{}-r{}{}-d{}-t{}-batsiz{}-{}steps-lr{}-lr2{}-{}-acteval_{}".format(
    options.agent, options.net_type, options.replay, str(options.buffer_size), options.dup, options.timesteps, 
    options.batch_size, options.steps, str(options.lr_c), str(options.lr_a), options.mode, options.obj_func)

# create evironment
# env = DeepSeaTreasure(view=(5,5), scale=9) if options.dst_view == '5' else DeepSeaTreasure(view=(3,3), scale=15)
random.seed(options.seed)
np.random.seed(options.seed)

json_file = "mine_config_det.json"
minecart = Minecart.from_json(json_file)
pixel_minecart = PixelMinecart(minecart)
obj_cnt = minecart.obj_cnt()

all_weights = list(np.loadtxt("regular_weights"))

# all_weights = get_weights_from_json('./train_weights_dst.json') if options.mode == "sparse" else get_weights_from_json('./train_weights_dst_r.json')
timestamp = time.strftime("%m%d_%H%M", time.localtime())
deep_agent = AGENT_DICT[options.agent]
agent = deep_agent(minecart, 
                   pixel_minecart,
                   gamma=options.discount,
                   weights=None,
                   timesteps=options.timesteps,
                   batch_size=options.batch_size,
                   replay_type=options.replay,
                   buffer_size=options.buffer_size,
                   buffer_a=options.buffer_a,
                   buffer_e=options.buffer_e,
                   memnn_size=options.memnn_size,
                   end_e=options.end_e,
                   net_type=options.net_type,
                   obj_func=options.obj_func,
                   lr=options.lr_c,
                   lr_2=options.lr_a,
                   frame_skip=options.frame_skip,
                   update_interval=options.updates,
                   dup=options.dup,
                   action_conc=options.action_conc,
                   feature_embd=options.feature_embd,
                   extra='{}_{}'.format(timestamp, hyper_info),
                   gpu_setting=options.gpu_setting)

steps_per_weight = 5000 if options.mode == "sparse" else 1

# log_file_name = 'output/logs/{}_dst{}_rewards_{}.log'.format(timestamp, options.dst_view, hyper_info)
log_file_name = 'output/logs/rewards_AP_1-regular-lambda3-minecart'
with open(log_file_name, 'w', 1) as log_file:
    agent.train(log_file, options.steps, all_weights, steps_per_weight, options.steps*10, log_game_step=options.log_game)

# print stats
length, step_rewards, step_regrets, step_nb_episode = rebuild_log(total=options.steps*10, log_file=log_file_name)
print_stats(length, step_rewards, step_regrets, step_nb_episode, write_to_file=True, timestamp=timestamp)


