from random import choice
import cv2

import vizdoom as vzd

print("VizDoom example showing different buffers (screen, depth, labels)")

DEFAULT_CONFIG = 'ViZDoom/scenarios/health_gathering_supreme.cfg'

game = vzd.DoomGame()
game.load_config(DEFAULT_CONFIG)

game.set_screen_format(vzd.ScreenFormat.BGR24)
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

# Enables depth buffer.
game.set_depth_buffer_enabled(True)

# Enables labeling of in game objects labeling.
game.set_labels_buffer_enabled(True)

# Enables buffer with top down map of he current episode/level .
game.set_automap_buffer_enabled(True)
game.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)
game.set_automap_rotate(False)
game.set_automap_render_textures(False)

game.set_render_hud(True)
game.set_render_minimal_hud(False)

game.set_mode(vzd.Mode.PLAYER)
game.init()

actions = [[True, False, False], [False, True, False], [False, False, True]]

episodes = 10
sleep_time = 0.028

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Not needed for the first episode but the loop is nicer.
    game.new_episode()
    while not game.is_episode_finished():
        # Gets the state and possibly do something with it
        state = game.get_state()

        # Display all the buffers here!

        # Just screen buffer, given in selected format. This buffer is always available.
        # screen = state.screen_buffer
        # cv2.imshow('ViZDoom Screen Buffer', screen)

        # Depth buffer, always in 8-bit gray channel format.
        # This is most fun. It looks best if you inverse colors.
        depth = state.depth_buffer
        if depth is not None:
            cv2.imshow('ViZDoom Depth Buffer', depth)

        # Labels buffer, always in 8-bit gray channel format.
        # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
        # Labels data are available in state.labels, also see labels.py example.
        labels = state.labels_buffer
        if labels is not None:
            cv2.imshow('ViZDoom Labels Buffer', labels)

        # Map buffer, in the same format as screen buffer.
        # Shows top down map of the current episode/level.
        automap = state.automap_buffer
        if automap is not None:
            cv2.imshow('ViZDoom Map Buffer', automap)

        cv2.waitKey(int(sleep_time * 1000))

        game.make_action(choice(actions))

        print("State #" + str(state.number))
        print("=====================")

    print("Episode finished!")
    print("************************")

cv2.destroyAllWindows()