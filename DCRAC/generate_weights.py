import sys
import json
from utils import generate_weights

if len(sys.argv) > 1 and sys.argv[1] == 'd':
    from deep_sea_treasure import DeepSeaTreasure
    env = DeepSeaTreasure()
    filename = 'train_weights_dst.json'
else:
    from minecart import Minecart
    json_file = "mine_config.json"
    env = Minecart.from_json(json_file)
    filename = 'train_weights_minecart.json'

all_weights = generate_weights(500, n=env.obj_cnt, m=1)
# all_weights = generate_weights(100000, n=env.obj_cnt, m=10)
all_weights = [w.tolist() for w in all_weights]

with open(filename, 'w') as f:
    json.dump(all_weights, f)