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

def episodes_evaluate(file_path):
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
    error_list = []
    treasure = [18, 26, 31, 44, 48.2, 56, 72, 76.3, 90, 100]
    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            if log[0] == 'episode':
                steps_list.append(log[1])
                # print(log[1])
                weight = parse_array(log[-2])
                error = log[-4]
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
                # if opt_reward - scal_reward < 0:
                #     # print(log[1])
                #     # print(weight)
                #     # print(opt_reward)
                #     error += (opt_reward - scal_reward)
                total_regret += (opt_reward - scal_reward)
                regret_list.append(total_regret)
                error_list.append(error)

    df = pd.DataFrame({'step':steps_list, 'regret':regret_list, 'scal_reward':scal_reward_list, 'act_treasure':act_treasure, 'opt_reward':opt_reward_list, 'opt_treasure':opt_treasure, 'error':error_list})
    df.to_csv(file_path+'.csv')

def logs_evaluate(file_path):
    log_file = file_path 
    steps_list = []

    error_list = []

    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            if log[0] == 'logs':
                steps_list.append(log[1])
                # print(log[1])
                weight = parse_array(log[-2])
                error = log[-5]

                error_list.append(error)

    df = pd.DataFrame({'step':steps_list, 'error':error_list})
    df.to_csv(log_file+'.csv')


def cal_adhesion(file_path):
    steps_list = list()
    adhesion_list = list()

    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            batch_size = int(log[1])
            steps_list.append(log[0])
            adhesion = 0
            for i in eval(log[2]):
                adhesion += np.linalg.norm(np.array(i[0])-np.array(parse_array(log[-1])))*i[1]
            adhesion_list.append(adhesion/batch_size)

            # if log[0] == stop:
            #     break
            
    
    plt.plot(steps_list[::8], adhesion_list[::8])

    plt.title('adhesion degree')
    plt.xlabel('steps')
    plt.ylabel('adhesion degree')

    x_major_locator = MultipleLocator(160)
    ax = plt.gca()

    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=30)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 3200)

    # plt.savefig(log_file+'.jpg')
    plt.show()
    plt.close()

def cal_adhesion_2(file_path, record_range, sample_interval=1):
    steps_list = list()
    adhesion_list = list()

    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            
            if int(log[0]) >= record_range[0]:
                batch_size = int(log[1])
                steps_list.append(log[0])
                adhesion = 0
                for i in eval(log[2]):
                    adhesion += np.linalg.norm(np.array(i[0])-np.array(parse_array(log[-1]))) * i[1]
                adhesion_list.append(adhesion/batch_size)
                
            if int(log[0]) >= record_range[1]:
                break

    plt.plot(steps_list[::sample_interval], adhesion_list[::sample_interval])
    plt.title('adhesion degree')
    plt.xlabel('steps')
    plt.ylabel('adhesion degree')

    x_major_locator = MultipleLocator((record_range[1]-record_range[0])/100)
    ax = plt.gca()

    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=30)
    ax.xaxis.set_major_locator(x_major_locator)

    for i in range(0, 10, 1):
        plt.hlines(i/10, 0, len(steps_list), colors = "black", linestyles = "dashed")
    # plt.tick_params(axis='x', labelsize=6)
    # plt.xticks(rotation=30)
    # plt.xlim(record_range[0], record_range[1])

    plt.show()
    plt.close()
            

def draw_episodes(file_path):
    log_file = file_path 
    data = pd.read_csv(log_file+'.csv')

    step_list = data['step'].to_list()
    new_steps_list = []
    regret_list = []
    error_list = []
    act_treasure = []
    opt_treasure = []

    # sample the date per 100 steps
    flag = [i for i in range(step_list[0],step_list[-1],100)]

    new_steps_list.append(step_list[0])
    regret_list.append(data['regret'][0])
    error_list.append(data['error'][0])
    act_treasure.append(data['act_treasure'][0])
    opt_treasure.append(data['opt_treasure'][0])
    j = 1

    for i in range(1, len(step_list), 1):
        
        if j == len(flag):
            break
        # prevent the situation that the agent runs over 100 step
        while step_list[i] - flag[j] >= 99:
            j = j + 1
        
        # to find the cloest the step to the drawing point
        if step_list[i] >= (flag[j] - 99) and step_list[i] <= (flag[j] + 99):
            if abs(step_list[i+1] - flag[j]) < abs(step_list[i] - flag[j]):
                continue
            else:
                new_steps_list.append(step_list[i])
                regret_list.append(data['regret'][i])
                error_list.append(data['error'][i])
                act_treasure.append(data['act_treasure'][i])
                opt_treasure.append(data['opt_treasure'][i])
                j = j + 1

    # regret
    ax1 = plt.subplot(1, 3, 1)
    plt.sca(ax1)
    plt.plot(new_steps_list, regret_list)

    plt.title('Total Regret')
    plt.xlabel('steps')
    plt.ylabel('regret')

    x_major_locator = MultipleLocator(10000)
    ax = plt.gca()

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['bottom'].set_position(('data',0))  

    plt.tick_params(axis='x', labelsize=6)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 100000)
    for i in range(0, 100001, 5000):
        plt.vlines(i, 0, 20000, colors = "black", linestyles = "dashed")

    # treasure
    ax2 = plt.subplot(1, 3, 2)
    plt.sca(ax2)
    plt.plot(new_steps_list, act_treasure, color='blue', label='actural treasure')
    plt.plot(new_steps_list, opt_treasure, color='orange', label='optimal treasure')

    plt.title('Treasure')
    plt.xlabel('steps')
    plt.ylabel('Treasure')

    x_major_locator = MultipleLocator(10000)
    ax = plt.gca()

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['bottom'].set_position(('data',0))  

    plt.tick_params(axis='x', labelsize=6)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 100000)
    for i in range(0, 100001, 5000):
        plt.vlines(i, 0, 100, colors = "black", linestyles = "dashed")

    plt.legend(loc="upper right")
    
    # error
    ax3 = plt.subplot(1, 3, 3)
    plt.sca(ax3)
    plt.plot(new_steps_list, error_list)
    plt.title('error')
    plt.xlabel('steps')
    plt.ylabel('error')

    x_major_locator = MultipleLocator(10000)
    ax = plt.gca()

    ax.spines['bottom'].set_position(('data',0))  

    plt.tick_params(axis='x', labelsize=6)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 100000)
    
    # plt.savefig(log_file+'.jpg')
    plt.show()


logs_file_path = os.path.join(os.getcwd(), '***logs/rewards_P_1-regular')
transitions_file_path = os.path.join(os.getcwd(), '***logs/rewards_AP_2-regular-transitions_logs')
# episodes_evaluate(logs_file_path)
# draw_episodes(logs_file_path)
cal_adhesion(transitions_file_path)
# cal_adhesion_2(transitions_file_path, [1004, 6036], 1)