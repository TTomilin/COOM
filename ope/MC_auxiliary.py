import chainer.functions as F
import cupy as cp
import numpy as np

ID = "MC_auxiliary"

initial_z_t = None


def transform_to_weights(args, parameters):
    if args.weights_type == 1:
        W_c = parameters[0:args.action_dim * (args.z_dim + args.hidden_dim)].reshape(args.action_dim,
                                                                                     args.z_dim + args.hidden_dim)
        b_c = parameters[args.action_dim * (args.z_dim + args.hidden_dim):]
    elif args.weights_type == 2:
        W_c = parameters
        b_c = None
    return W_c, b_c


def action(args, W_c, b_c, z_t, h_t, c_t, gpu):
    if args.weights_type == 1:
        input = F.concat((z_t, h_t), axis=0).data
        action = F.tanh(W_c.dot(input) + b_c).data
    elif args.weights_type == 2:
        input = F.concat((z_t, h_t, c_t), axis=0).data
        dot = W_c.dot(input)
        if gpu is not None:
            dot = cp.asarray(dot)
        else:
            dot = np.asarray(dot)
        output = F.tanh(dot).data
        if output == 1.:
            output = 0.999
        action_dim = args.action_dim + 1
        action_range = 2 / action_dim
        action = [0. for i in range(action_dim)]
        start = -1.
        for i in range(action_dim):
            if start <= output and output <= (start + action_range):
                action[i] = 1.
                break
            start += action_range
        mid = action_dim // 2  # reserve action[mid] for no action
        action = action[0:mid] + action[mid + 1:action_dim]
    if gpu is not None:
        action = cp.asarray(action).astype(cp.float32)
    else:
        action = np.asarray(action).astype(np.float32)
    return action
