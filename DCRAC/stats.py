import os
import time
import numpy as np

RESULT_FILE_PATH = 'output/results.csv'

OPT = {
    "r1": [17.099999999999998, 23.465, 26.578624999999995, 32.34404318749999, 33.65985767171874, 35.29396694457811, 36.96062999612435, 37.2096009065149, 37.63083016972594, 37.73536025353073], 
    "r2": [-1.95, -2.8525, -3.709875, -6.033254078, -6.731591374, -8.025261215, -10.24650042, -10.7341754, -12.05571363, -12.83028155], 
    "thre": [0.12418300653594767, 0.21590909090909108, 0.28723418349345714, 0.3467153284367459, 0.44186046511367333, 0.5713207427630154, 0.662020906329149, 0.7582986457734945, 0.881093935656347]
    }

OPT_R = np.array([OPT['r1'],OPT['r2']]).T

def parse_array(text):
    array = text.lstrip(' [').rstrip(' ]').split()
    array = [eval(a) for a in array]
    return array

def compute_log(log_file, result_file_path=None, timestamp=None):
    error=0
    total_eps = 0
    total_reward = 0
    total_regret = 0

    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            if log[0] == 'episode':
                weight = parse_array(log[-1])
                # reward = parse_array(log[7])
                scal_reward = eval(log[6])

                opt_reward = max(np.dot(OPT_R, weight))
                
                total_eps += 1

                total_reward += scal_reward
                if opt_reward - scal_reward < 0:
                    error += (opt_reward - scal_reward)
                total_regret += (opt_reward - scal_reward)
                
    avg_reward = total_reward/total_eps
    avg_regret = total_regret/total_eps

    print('avg_reward:', avg_reward, 'avg_regret:',  avg_regret)

    if result_file_path is not None:
        if timestamp is None:
            timestamp = time.strftime("%m%d_%H%M", time.localtime()) 
        if not os.path.exists(result_file_path): 
            with open(result_file_path, 'w') as result_file:
                result_file.write('time,avg reward,avg regret\n')
        with open(result_file_path, 'a+') as result_file:
            print('{},{},{}'.format(timestamp, avg_reward, avg_regret), file=result_file)


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
                game_steps_till_now = eval(log[2])
                weight = parse_array(log[-1])
                reward = parse_array(log[7])
                scal_reward = eval(log[6])

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
