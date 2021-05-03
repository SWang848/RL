import os
import numpy as np
import pandas as pd

OPT={
    "r1": [0.85192097, 0.923623597, 0.65784273, 0.47126046, 0.55392683, 0.57676679, 0.45001093, 0.16992796, 0.18423008, 0],
    "r2": [0.16992796, 0.18423008, 0.45001093, 0.47126046, 0.55392683, 0.57676679, 0.65784273, 0.85192097, 0.923623597, 0],
    "r3": [-0.81650404, -0.94917702, -1.11527075, -0.70880496, -0.79616165, -1.11842114, -1.11527075, -0.81650404, -0.94917702, -0.24923698]
}

OPT_1={
    "r1": [ 0.55612687,  0.85192097,  0.60293373,  0.92362357,  0.40115733,  0.58274497,
   0.63179215,  0.65784273,  0.47126046,  0.55392683,  0.57676679,  0.27441997,
   0.39863875,  0.43219049,  0.45001093,  0.11092755,  0.16992796,  0.12026385,
   0.18423008,  0.        ],
    "r2": [ 0.11092755,  0.16992796,  0.12026385,  0.18423008,  0.27441997,  0.39863875,
   0.43219049,  0.45001093,  0.47126046,  0.55392683,  0.57676679,  0.40115733,
   0.58274497,  0.63179215,  0.65784273,  0.55612687,  0.85192097,  0.60293373,
   0.92362357,  0.        ],
    "r3": [-0.65061296, -0.81650404, -0.77743027, -0.94917702, -0.67291286, -0.83521908,
  -0.96483263, -1.11527075, -0.70880496, -0.79616165, -1.11842114, -0.67291286,
  -0.83521908, -0.96483263, -1.11527075, -0.65061296, -0.81650404, -0.77743027,
  -0.94917702, -0.24923698]
  }

OPT_R = np.array([OPT['r1'], OPT['r2'], OPT['r3']]).T

def parse_array(text):
    array = text.lstrip(' [').rstrip(' ]').split()
    array = [eval(a) for a in array]
    return array

log_file = r'.\output\logs\rewards_memory_net-True temp_att-False nl-False lstm-False a-cond m-DER s-None  e-0.05 d-CN x- sparse p-full fs-4 d-0.98 up-4 lr-0.02 e-1 p-100000 m-2.0-0.01'
regret_list = []
steps_list = []
weight_list = []
opt_reward_list = []
scal_reward_list = []
act_reward = []
opt_reward = []
error = 0
total_eps = 0
total_reward = 0
total_regret = 0

with open(log_file, 'r') as fin:
    for line in fin.readlines():
        line = line.rstrip('\n')
        log = line.split(';')
        if log[0] == 'episode':
            steps_list.append(log[1])
            weight = parse_array(log[-2])
            weight_list.append(weight)
            scal_reward = eval(log[5])
            act_reward.append(parse_array(log[7])[:2])
            scal_reward_list.append(scal_reward)
            opt_reward = max(np.dot(OPT_R, weight))
            # opt_treasure.append(treasure[np.argmax(np.dot(OPT_R, weight))])
            opt_reward_list.append(opt_reward)

            total_eps += 1

            total_reward += scal_reward
            if opt_reward - scal_reward < 0:
                error += (opt_reward - scal_reward)
            total_regret += (opt_reward - scal_reward)
            regret_list.append(total_regret)


df = pd.DataFrame({'step':steps_list, 'regret':regret_list, 'weight':weight_list, 'scal_reward':scal_reward_list, 'act_reward':act_reward, 'opt_reward':opt_reward_list})
df.to_csv(log_file+'.csv')