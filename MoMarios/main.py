# import gym
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from nes_py.wrappers import JoypadSpace

from optparse import OptionParser
import numpy as np
from MoMarios_qlearning_demo import DeepAgent
from MoMarios import MoMarios

def generate_weights(count=1, n=3, m=1):
    all_weights = []
    target = np.random.dirichlet(np.ones(n), 1)[0]
    prev_t = target
    for _ in range(count // m):
        target = np.random.dirichlet(np.ones(n), 1)[0]
        if m == 1:
            all_weights.append(target)
        else:
            for i in range(m):
                i_w = target * (i + 1) / float(m) + prev_t * \
                    (m - i - 1) / float(m)
                all_weights.append(i_w)
        prev_t = target + 0.

    return all_weights

parser = OptionParser()
parser.add_option(
    "-l",
    "--algorithm",
    dest="alg",
    choices=["scal", "mo", "mn", "cond", "uvfa", "random", "naive"],
    default="cond",
    help="Architecture type, one of 'scal','mo','meta','cond'")
parser.add_option(
    "-m",
    "--memory",
    dest="mem",
    default="DER",
    choices=["STD", "DER", "SEL", "EXP"],
    help="Memory type, one of 'std','crowd','exp','sel'")
parser.add_option(
    "-d",
    "--dupe",
    dest="dupe",
    default="CN",
    choices=["none", "CN", "CN-UVFA", "CN-ACTIVE"],
    help="Extra training")
parser.add_option(
    "--end_e",
    dest="end_e",
    default="0.01",
    help="Final epsilon value",
    type=float)
parser.add_option(
    "--start_e",
    dest="start_e",
    default="0.1",
    help="start epsilon value",
    type=float)
parser.add_option(
    "-r", "--lr", dest="lr", default="0.02", help="learning rate", type=float)
parser.add_option(
    "--clipnorm", dest="clipnorm", default="1", help="clipnorm", type=float)
parser.add_option(
    "--mem-a", dest="mem_a", default="2.", help="memory error exponent", type=float)
parser.add_option(
    "--mem-e", dest="mem_e", default="0.01", help="error offset", type=float)
parser.add_option(
    "--clipvalue",
    dest="clipvalue",
    default="0.5",
    help="clipvalue",
    type=float)
parser.add_option(
    "--momentum", dest="momentum", default="0.9", help="momentum", type=float)
parser.add_option(
    "-u",
    "--update_period",
    dest="updates",
    default="4",
    help="Update interval",
    type=int)
parser.add_option(
    "--target-update",
    dest="target_update_interval",
    default="150",
    help="Target update interval",
    type=int)
parser.add_option(
    "-f",
    "--frame-skip",
    dest="frame_skip",
    default="1",
    help="Frame skip",
    type=int)
parser.add_option(
    "--sample-size",
    dest="sample_size",
    default="16",
    help="Sample batch size",
    type=int)
parser.add_option(
    "-g",
    "--discount",
    dest="discount",
    default="0.95",
    help="Discount factor",
    type=float)
parser.add_option("--scale", dest="scale", default=1,
                help="Scaling", type=float)
parser.add_option("--anneal-steps",
                dest="steps", default=10000, help="steps",  type=int)
parser.add_option("-x", "--extra", dest="extra", default="")
parser.add_option("-p", "--reuse", dest="reuse",
                choices=["full", "sectionned", "proportional"], default="full")
parser.add_option(
    "-c", "--mode", dest="mode", choices=["regular", "sparse"], default="sparse")
parser.add_option(
    "-s", "--seed", dest="seed", default=None, help="Random Seed", type=int)
parser.add_option(
    "--lstm", dest="lstm", default=False)
parser.add_option(
    "--non_local", dest="attention", default=True)

(options, args) = parser.parse_args()


extra = "Mario-lstm-{} clipN-{} clipV-{} attention-{} a-{} m-{} s-{}  e-{} d-{} x-{} {} p-{} fs-{} d-{} up-{} lr-{} e-{} p-{} m-{}-{}".format(
options.lstm, options.clipnorm, options.clipvalue, options.attention,
options.alg, options.mem, options.seed,
options.end_e, options.dupe, options.extra, options.mode, options.reuse,
options.frame_skip,
np.round(options.discount, 4), options.updates,
np.round(options.lr, 4),
np.round(options.scale, 2), np.round(options.steps, 2), np.round(options.mem_a, 2), np.round(options.mem_e, 2))

# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = MoMarios(life_done=True,
               single_stage=True,
               n_obj=2)

obj_cnt = 2
steps = 10000
all_weights = generate_weights(count=steps, n=obj_cnt, m=1)

agent = DeepAgent(
    range(5),               # actions
    obj_cnt,                # objective_cnt
    options.steps,          # memory_size
    sample_size=options.sample_size,
    weights=None,
    discount=options.discount,
    learning_rate=options.lr,
    target_update_interval=options.target_update_interval,
    alg=options.alg,
    frame_skip=options.frame_skip,
    start_e=options.start_e,
    end_e=options.end_e,
    memory_type=options.mem,
    update_interval=options.updates,
    reuse=options.reuse,
    mem_a=options.mem_a,
    mem_e=options.mem_e,
    extra=extra,
    clipnorm=options.clipnorm,
    clipvalue=options.clipvalue,
    momentum=options.momentum,
    scale=options.scale,
    dupe=None if options.dupe == "none" else options.dupe,
    lstm=options.lstm,
    non_local=options.attention)

steps_per_weight = 5000 if options.mode == "sparse" else 1
log_file = open('MoMarios/output/logs/rewards_{}'.format(extra), 'w')
agent.train(environment=env, 
            log_file=log_file,
            learning_steps=options.steps,
            weights=all_weights, 
            per_weight_steps=steps_per_weight, 
            total_steps=options.steps*10)