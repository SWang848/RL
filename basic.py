
import vizdoom as vzd

from random import choice
from time import sleep

#configuration and initialization 
game = vzd.DoomGame()
game.set_doom_scenario_path('vizdoomgym/vizdoomgym/envs/scenarios/basic.wad')
game.set_doom_map('map01')

#rendering options
#Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
game.set_render_hud(False)
game.set_render_minimal_hud(False)  # If hud is enabled
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)  # Bullet holes and blood on the walls
game.set_render_particles(False)
game.set_render_effects_sprites(False)  # Smoke and blood
game.set_render_messages(False)  # In-game messages
game.set_render_corpses(False)
game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

#enables depth buffer
game.set_depth_buffer_enabled(True)

#enables labeling of in game objects labeling.
game.set_labels_buffer_enabled(True)

#enables buffer with top down map of the current episode/level.
game.set_automap_buffer_enabled(True)

#enabels information about all object present in the current episode/levle
game.set_objects_info_enabled(True)

#enables information about all sectors (map layout).
game.set_sectors_info_enabled(True)


#determine which buttons can be used by the agent
game.add_available_button(vzd.Button.MOVE_LEFT)
game.add_available_button(vzd.Button.MOVE_RIGHT)
game.add_available_button(vzd.Button.ATTACK)

#add variables into state
game.add_available_game_variable(vzd.GameVariable.AMMO2)

#other setting
game.set_episode_timeout(200)
game.set_episode_start_time(10)
game.set_window_visible(True)

#agents get a -1 reward for each move no matter what happened
game.set_living_reward(-1)

#sets vizdoom mode, default player
game.set_mode(vzd.Mode.PLAYER)

#initialte the game
game.init()

actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
episodes = 10

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

for i in range(episodes):
    print("Episode #" + str(i + 1))

    game.new_episode()

    while not game.is_episode_finished():

        #get the state
        state = game.get_state()

        #which consists of:
        n = state.number
        vars = state.game_variables
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        #make random action and get reward
        skiprate = 4
        r = game.make_action(choice(actions), skiprate)

        #Prints state's game variables and reward.
        print("State #" + str(n))
        print("Game variables:", vars)
        print("Reward:", r)
        print("=====================")

        if sleep_time > 0:
            sleep(sleep_time)

    print("Episode finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")