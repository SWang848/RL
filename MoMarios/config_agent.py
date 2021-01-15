IM_SIZE = 84

# Neural Network config
CONV_FILTERS = [32, 48]
CONV_KERNEL_SIZES = [6, 5]
CONV_STRIDES = [2, 2]
POST_CONV_DENSE_SIZE = 512
DUELING_DEPTH = 3
DUELING_LAYER_SIZE = 512

KAPPA = -1e-2
DENSE_INIT = 'glorot_uniform'

BLACK_AND_WHITE = False

DEBUG = 1

def LEAKY_RELU(): return LeakyReLU(0.01)

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

def masked_error(args):
    """
        Masked asolute error function

        Args:
            y_true: Target output
            y_pred: Actual output
            mask: Scales the loss, should be compatible with the shape of 
                    y_true, if an element of the mask is set to zero, the
                    corresponding loss is ignored
    """
    y_true, y_pred, mask = args
    loss = K.abs(y_true - y_pred)
    loss *= mask
    return K.sum(loss, axis=-2)