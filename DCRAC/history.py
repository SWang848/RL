import numpy as np
import time
import cv2
try:
    from PIL import Image
except:
    import Image

from utils import *


class History():
    """
        Manages frame history
    """

    def __init__(self, length, im_shape, nb_action):
        """
            Args:
                length: How many frames should be stored in the history
                im_shape: Target size to crop to, im_shape = (WIDTH,HEIGHT,CHANNEL)
        """
        assert len(im_shape) == 3

        self.im_shape = im_shape
        self.black_and_white = True if im_shape[2] == 1 else False
        self.length = length

        self.history_o = None
        self.history_a_prev = None
        self.reset()

    def reset(self):
        """Reset the history of observation and action
        """
        self.history_o = np.zeros((self.length, ) + self.im_shape, dtype=np.uint8)
        # self.history_a_prev = np.zeros((self.length, ), dtype=np.int8)
        
        # action '-1' means None
        self.history_a_prev = -np.ones((self.length, ), dtype=np.int8)

    def reset_with_raw_frame(self, raw_frame, action_prev=None, fill=False):
        """Fill the history with a raw frame
        """
        self.reset()
        # ↓ action '-2' means no action, will be translated to all zeros with shape (nb_objective) as one-hot
        action_prev = -2 if action_prev is None else action_prev
        if fill:
            self.add_raw_frame(raw_frame, action_prev)
            return self.fill_with_last_frame()
        else:
            return self.add_raw_frame(raw_frame, action_prev)

    def add_raw_frame(self, raw_frame, action_prev, save=False):
        """Adds a new frame to the history
        """
        self.history_o = np.roll(self.history_o, -1, axis=0)
        self.history_o[-1] = self.process_frame(raw_frame, save=save)
        self.history_a_prev = np.roll(self.history_a_prev, -1, axis=0)
        self.history_a_prev[-1] = action_prev

        return self.history_o, self.history_a_prev

    def fill_with_last_frame(self):
        """
            Fills the state with the latest experienced frame
        """
        for i in range(len(self.history_o)-1):
            self.history_o[i] = self.history_o[-1]
            # self.history_a_prev[i] = self.history_a_prev[-1]
        return self.history_o, self.history_a_prev

    def process_frame(self, raw_frame, save=False, filename=None):
        """Processes a frame by resizing and cropping as necessary and then
        converting to grayscale
        
        Arguments:
            raw_frame {np.array} -- Raw pixels
        
        Keyword Arguments:
            save {bool} -- Whether to save the converted frame to disk (default: {False})
            filename {str} -- Filename to save it to (default: {None})
        
        Returns:
            np.array -- The processed frame
        """

        if self.black_and_white:
            raw_frame = cv2.cvtColor(raw_frame,cv2.COLOR_RGB2GRAY)
       
        cropped = cv2.resize(raw_frame, dsize=self.im_shape[:2], interpolation=cv2.INTER_AREA)
        cropped = cropped.reshape(self.im_shape)
        
        if save:
            self.save_image(cropped)

        return cropped
      

    def save_image(self, frame, filename=None):
        if filename is None:
            filename = "./output/imgs/"+str(time.time())+".png"
        if self.black_and_white:
            frame = frame.reshape(self.im_shape[:2])
            img = Image.fromarray(frame, mode='L')
            img.save(filename)
        else:
            img = Image.fromarray(frame, mode='RGB')
            img.save(filename)