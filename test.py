import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import time
import os

RESULT_FILE_PATH = 'output/results.csv'
def cal_opt_penality(r, discount):
    if r==1:
        return r*discount
    reward = discount**r
    reward += cal_opt_penality(r-1, discount)
    return reward


# print([cal_opt_penality(i, 0.98) for i in [2, 3, 4, 7, 8, 10, 14, 15, 18, 20]])
# print([i*0.98**j for i, j in zip([18,26,31,44,48.2,56,72,76.3,90,100], [2, 3, 4, 7, 8, 10, 14, 15, 18, 20])])

OPT = {
    "r1": [17.099999999999998, 23.465, 26.578624999999995, 32.34404318749999, 33.65985767171874, 35.29396694457811, 36.96062999612435, 37.2096009065149, 37.63083016972594, 37.73536025353073], 
    "r2": [-1.95, -2.8525, -3.709875, -6.033254078, -6.731591374, -8.025261215, -10.24650042, -10.7341754, -12.05571363, -12.83028155], 
    "thre": [0.12418300653594767, 0.21590909090909108, 0.28723418349345714, 0.3467153284367459, 0.44186046511367333, 0.5713207427630154, 0.662020906329149, 0.7582986457734945, 0.881093935656347]
    }

# OPT = {
#     "r1": [17.2872, 24.470992, 28.593412959999995, 38.19752346285568, 41.006777688442064, 45.756077185702615, 54.26221978619292, 56.3528225318443, 62.56217977713295, 66.76079717550942], 
#     "r2": [-1.9404, -2.881592, -3.80396016, -6.461848870910719, -7.312611893492505, -8.963432462510202, -12.071544867729804, -12.810113970375207, -14.93836878800538, -16.287209384000366], 
#     "thre": [0.12418300653594767, 0.21590909090909108, 0.28723418349345714, 0.3467153284367459, 0.44186046511367333, 0.5713207427630154, 0.662020906329149, 0.7582986457734945, 0.881093935656347]
#     }

OPT_R = np.array([OPT['r1'],OPT['r2']]).T

def parse_array(text):
    array = text.lstrip(' [').rstrip(' ]').split()
    array = [eval(a) for a in array]
    return array

def rebuild_log(total, log_file, step_length=100):
    length = int(total/step_length)
    step_rewards = [0.] * (length)
    step_regrets = [0.] * (length)
    step_nb_episode = [0] * (length)

    i = 0
    error = 0

    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            if log[0] == 'episode':
                game_steps_till_now = eval(log[1])
                weight = parse_array(log[-2])
                reward = parse_array(log[7])
                scal_reward = eval(log[5])

                opt_reward = max(np.dot(OPT_R, weight))

                while game_steps_till_now > (i+1)*step_length:
                    i+=1
                if i >= length: break
                
                if reward[0] > 0:
                    step_nb_episode[i] += 1

                    step_rewards[i] += scal_reward
                    if opt_reward - scal_reward < 0:
                        error += (opt_reward - scal_reward)
                    step_regrets[i] += (opt_reward - scal_reward)

    return length, step_rewards, step_regrets, step_nb_episode

def accumulate(length, step_rewards, step_regrets, step_nb_episode):
    acc_rewards = [0.] * (length)
    acc_regrets = [0.] * (length)

    acc_rewards[0] = step_rewards[0]
    acc_regrets[0] = step_regrets[0]
    for i in range(1,length):
        # acc_rewards[i] = acc_rewards[i-1] + step_rewards[i]
        # acc_regrets[i] = acc_regrets[i-1] + step_regrets[i]

        if step_nb_episode[i] != 0:
            acc_regrets[i] = acc_regrets[i-1] + (step_regrets[i]/step_nb_episode[i])
            acc_rewards[i] = acc_rewards[i-1] + (step_rewards[i]/step_nb_episode[i])
        else:
            acc_regrets[i] = acc_regrets[i-1]
            acc_rewards[i] = acc_rewards[i-1]

    return acc_rewards, acc_regrets

def print_stats(length, step_rewards, step_regrets, step_nb_episode, write_to_file=False, timestamp=None):
    total_eps = sum(step_nb_episode)
    avg_rwd = sum(step_rewards)/total_eps
    avg_reg = sum(step_regrets)/total_eps

    last_quarter = int(0.25 * length)
    last_qtr_eps = sum(step_nb_episode[-last_quarter:])
    last_qtr_rwd = sum(step_rewards[-last_quarter:])/last_qtr_eps
    last_qtr_reg = sum(step_regrets[-last_quarter:])/last_qtr_eps

    print('avg_reward (total/last_qtr):', avg_rwd, last_qtr_rwd)
    print('avg_regret (total/last_qtr):', avg_reg, last_qtr_reg)

    if write_to_file:
        if timestamp is None:
            timestamp = time.strftime("%m%d_%H%M", time.localtime()) 
        if not os.path.exists(RESULT_FILE_PATH): 
            with open(RESULT_FILE_PATH, 'w') as result_file:
                result_file.write('time,total avg reward,last qtr avg reward,total avg regret,last qtr avg regret\n')
        with open(RESULT_FILE_PATH, 'a+') as result_file:
            print('{},{},{},{},{}'.format(timestamp, avg_rwd, last_qtr_rwd, avg_reg, last_qtr_reg), file=result_file)


log_file = r'C:\Users\Shang\reinforcement learning\DynMORL-master\output\logs\rewards_3-newEnv-lstm-False clipN-1.0 clipV-0.5 attention-True a-cond m-DER s-None  e-0.01 d-CN x- sparse p-full fs-1 d-0.95 up-4 lr-0.02 e-1 p-10000 m-2.0-0.01'
# log_file = r'C:\Users\Shang\reinforcement learning\DynMORL-master\output\logs\rewards_new_a-scal m-STD s-None  e-0.05 d-False x- sparse p-full fs-4 d-0.98 up-4 lr-0.02 e-1 p-10000 m-1-0.02'
# log_file = r'C:\Users\Shang\reinforcement learning\DynMORL-master\output\logs\rewards_a-scal m-STD s-None  e-0.05 d-False x- sparse p-full fs-4 d-0.98 up-4 lr-0.02 e-1 p-10000 m-1-0.02'
# log_file = r'C:\Users\Shang\reinforcement learning\DynMORL-master\output\logs\rewards_a-scal m-STD s-None  e-0.05 d-False x- sparse p-full fs-4 d-0.98 up-4 lr-0.02 e-1 p-100000 m-1-0.02'
# length, step_rewards, step_regrets, step_nb_episode = rebuild_log(total=10000*10, log_file=log_file)
# print(step_regrets)
# print_stats(length, step_rewards, step_regrets, step_nb_episode, write_to_file=True, timestamp=time.strftime("%m%d_%H%M", time.localtime()))

regret_list = []
steps_list = []
scal_reward_list = []
opt_reward_list = []
act_treasure = []
opt_treasure = []
error = 0
total_eps = 0
total_reward = 0
total_regret = 0
treasure = [18, 26, 31, 44, 48.2, 56, 72, 76.3, 90, 100]
with open(log_file, 'r') as fin:
    for line in fin.readlines():
        line = line.rstrip('\n')
        log = line.split(';')
        if log[0] == 'episode':
            steps_list.append(log[1])
            # print(log[1])
            weight = parse_array(log[-2])
            # print(weight)
            # reward = parse_array(log[7])
            scal_reward = eval(log[5])
            act_treasure.append(parse_array(log[7])[0])
            scal_reward_list.append(scal_reward)
            # disc_reward = parse_array(log[6])
            # print(disc_reward)
            # scal_reward = np.dot(disc_reward, weight)
            # print(scal_reward)
            opt_reward = max(np.dot(OPT_R, weight))
            opt_treasure.append(treasure[np.argmax(np.dot(OPT_R, weight))])
            opt_reward_list.append(opt_reward)

            # print(opt_reward)
            total_eps += 1

            total_reward += scal_reward
            if opt_reward - scal_reward < 0:
                # print(log[1])
                # print(weight)
                # print(opt_reward)
                error += (opt_reward - scal_reward)
            total_regret += (opt_reward - scal_reward)
            regret_list.append(total_regret)
                
# avg_reward = total_reward/total_eps
# avg_regret = total_regret/total_eps

df = pd.DataFrame({'step':steps_list, 'regret':regret_list, 'scal_reward':scal_reward_list, 'act_treasure':act_treasure, 'opt_reward':opt_reward_list, 'opt_treasure':opt_treasure})
df.to_csv(log_file+'.csv')

# # print(avg_reward)
# # print(avg_regret)

# plt.plot(steps_list,regret_list,color="red",linewidth=2)
# # plt.xlim((0, 1e5))
# plt.xlabel("Step(s)")
# plt.ylabel("Cumulative regret")
# plt.title("DST-DDDQN_DER")
# # plt.xticks(np.arange(0, 1e5, 5000))
# # ax=plt.gca()
# # ax.xaxis.set_major_locator(MultipleLocator(100))
# # plt.xlim(0, 1e5)
# plt.show()


 

# def str_to_list(str_):
#     return [float(x) for x in str_.strip('[').strip(']').strip(' ').split()]

# def find_optimal(dict_):
#     opt_dict = {}
#     for i in dict_.keys():
#         list_ = []
#         for j in dict_[i]:
#             list_.append(np.dot(str_to_list(i), str_to_list(j)))
#         optimal_value = np.max(list_)
#         opt_dict[i] = optimal_value
#     return opt_dict


# disc_reward_list = []
# weight_list = []
# dict_ = {}
# steps_list = []

# with open(r'output\logs\rewards_a-scal m-STD s-None  e-0.05 d-False x- sparse p-full fs-4 d-0.98 up-4 lr-0.02 e-1 p-100000 m-1-0.02', 'r') as f:
#     for line in f.readlines():
#         line = line.rstrip('\n')
#         log = line.split(';')
#         if log[0] == 'episode':
#             steps_list.append(log[1])
#             disc_reward = log[6]
#             disc_reward_list.append(log[6])
#             weight = log[-2]
#             weight_list.append(log[-2])
#             if weight not in dict_.keys():
#                 dict_[weight] = [log[6]]
#             else:
#                 dict_[weight].append(log[6])

# # print(dict_)
# opt_dict = find_optimal(dict_)
# # print(opt_dict)
# regret_list = [opt_dict[weight_list[i]] - np.dot(str_to_list(disc_reward_list[i]), str_to_list(weight_list[i])) \
#                     for i in range(len(disc_reward_list))]
# for i in range(len(regret_list)):
#     if i == 0:
#         pass
#     else:
#         regret_list[i] += regret_list[i-1]
# # print(regret_list)
# # print(steps_list)

# # plt.figure() 
# fig = plt.plot(steps_list,regret_list,color="red",linewidth=2)
# plt.xlabel("Step(s)")
# plt.ylabel("Cumulative regret")
# plt.title("DST-DDDQN_DER")
# ax=plt.gca()
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
# plt.show()