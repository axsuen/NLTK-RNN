import numpy as np

from utils.tools import *
from nn.functional import sigmoid
from itertools import product
# Attension:
# - Never change the value of input, which will change the result of backward



class operator(object):
    """
    operator abstraction
    """

    def forward(self, input):
        """Forward operation, reture output"""
        raise NotImplementedError

    def backward(self, out_grad, input):
        """Backward operation, return gradient to input"""
        raise NotImplementedError


class relu(operator):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, input):
        output = np.maximum(0, input)
        return output

    def backward(self, out_grad, input):
        in_grad = (input >= 0) * out_grad
        return in_grad


class flatten(operator):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, input):
        batch = input.shape[0]
        output = input.copy().reshape(batch, -1)
        return output

    def backward(self, out_grad, input):
        in_grad = out_grad.copy().reshape(input.shape)
        return in_grad


class matmul(operator):
    def __init__(self):
        super(matmul, self).__init__()

    def forward(self, input, weights):
        """
        # Arguments
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            output: numpy array with shape(batch, out_features)
        """
        return np.matmul(input, weights)

    def backward(self, out_grad, input, weights):
        """
        # Arguments
            out_grad: gradient to the forward output of linear layer, with shape (batch, out_features)
            input: numpy array with shape (batch, in_features)
            weights: numpy array with shape (in_features, out_features)

        # Returns
            in_grad: gradient to the forward input with same shape as input
            w_grad: gradient to weights, with same shape as weights            
        """
        in_grad = np.matmul(out_grad, weights.T)
        w_grad = np.matmul(input.T, out_grad)
        return in_grad, w_grad




class vanilla_rnn(operator):
    def __init__(self):
        """
        # Arguments
            in_features: int, the number of inputs features
            units: int, the number of hidden units
            initializer: Initializer class, to initialize weights
        """
        super(vanilla_rnn, self).__init__()

    def forward(self, input, kernel, recurrent_kernel, bias):
        """
        # Arguments
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)]

        # Returns
            outputs: numpy array with shape (batch, units)
        """
        x, prev_h = input
        output = np.tanh(x.dot(kernel) + prev_h.dot(recurrent_kernel) + bias)
        return output

    def backward(self, out_grad, input, kernel, recurrent_kernel, bias):
        """
        # Arguments
            in_grads: numpy array with shape (batch, units), gradients to outputs
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)], same with forward inputs

        # Returns
            out_grads: [gradients to input numpy array with shape (batch, in_features), 
                        gradients to state numpy array with shape (batch, units)]
        """
        x, prev_h = input
        tanh_grad = np.nan_to_num(
            out_grad*(1-np.square(self.forward(input, kernel, recurrent_kernel, bias))))

        in_grad = [np.matmul(tanh_grad, kernel.T), np.matmul(
            tanh_grad, recurrent_kernel.T)]
        kernel_grad = np.matmul(np.nan_to_num(x.T), tanh_grad)
        r_kernel_grad = np.matmul(np.nan_to_num(prev_h.T), tanh_grad)
        b_grad = np.sum(tanh_grad, axis=0)

        return in_grad, kernel_grad, r_kernel_grad, b_grad


class gru(operator):
    def __init__(self):
        """
        # Arguments
            in_features: int, the number of inputs features
            units: int, the number of hidden units
            initializer: Initializer class, to initialize weights
        """
        super(gru, self).__init__()

    def forward(self, input, kernel, recurrent_kernel):
        """
        # Arguments
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)]

        # Returns
            outputs: numpy array with shape (batch, units)
        """
        x, prev_h = input
        _, all_units = kernel.shape
        units = all_units // 3
        kernel_z, kernel_r, kernel_h = kernel[:, :units], kernel[:, units:2*units],  kernel[:, 2*units:all_units]
        recurrent_kernel_z = recurrent_kernel[:, :units]
        recurrent_kernel_r = recurrent_kernel[:, units:2*units]
        recurrent_kernel_h = recurrent_kernel[:, 2*units:all_units]

        #####################################################################################
        # code here
        # reset gate
        x_r = sigmoid(np.matmul(prev_h, recurrent_kernel_r) + np.matmul(x, kernel_r))
        # update gate
        x_z = sigmoid(np.matmul(prev_h, recurrent_kernel_z) + np.matmul(x, kernel_z))
        # new gate
        x_h = np.tanh(np.matmul(x_r * prev_h, recurrent_kernel_h) + np.matmul(x, kernel_h))

        #####################################################################################

        output = (1 - x_z) * x_h + x_z * prev_h
        
        return output

    def backward(self, out_grad, input, kernel, recurrent_kernel):
        """
        # Arguments
            in_grads: numpy array with shape (batch, units), gradients to outputs
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)], same with forward inputs

        # Returns
            out_grads: [gradients to input numpy array with shape (batch, in_features), 
                        gradients to state numpy array with shape (batch, units)]
        """
        x, prev_h = input
        _, all_units = kernel.shape
        units = all_units // 3
        kernel_z, kernel_r, kernel_h = kernel[:, :units], kernel[:, units:2*units],  kernel[:, 2*units:all_units]
        recurrent_kernel_z = recurrent_kernel[:, :units]
        recurrent_kernel_r = recurrent_kernel[:, units:2*units]
        recurrent_kernel_h = recurrent_kernel[:, 2*units:all_units]

        #####################################################################################        
        # code here
        
        # reset gate
        x_r = sigmoid(np.matmul(prev_h, recurrent_kernel_r) + np.matmul(x, kernel_r))
        # update gate
        x_z = sigmoid(np.matmul(prev_h, recurrent_kernel_z) + np.matmul(x, kernel_z))
        # new gate
        x_h = np.tanh(np.matmul(x_r * prev_h, recurrent_kernel_h) + np.matmul(x, kernel_h))

        d0 = out_grad
        d1 = x_z * d0 
        d2 = prev_h * d0
        d3 = x_h * d0 
        d4 = -1 * d3
        d5 = d2 + d4
        d6 = (1- x_z) * d0
        d7 = d5 * (x_z * (1 - x_z))
        d8 = d6 * (1 - x_h**2)
        d9 = np.matmul(d8, kernel_h.T)
        d10 = np.matmul(d8, recurrent_kernel_h.T)
        d11 = np.matmul(d7, kernel_z.T)
        d12 = np.matmul(d7, recurrent_kernel_z.T)
        d13 = d10 * x_r
        d14 = d10 * prev_h
        d15 = d14 * (x_r * (1 - x_r))
        d16 = np.matmul(d15, kernel_r.T)
        d17 = np.matmul(d15, recurrent_kernel_r.T)

        x_grad = d9 + d11 + d16
        prev_h_grad = d12 + d13 + d1 + d17

        kernel_r_grad = np.matmul(x.T, d15)
        kernel_z_grad = np.matmul(x.T, d7)
        kernel_h_grad = np.matmul(x.T, d8)

        recurrent_kernel_r_grad = np.matmul(prev_h.T, d15)
        recurrent_kernel_z_grad = np.matmul(prev_h.T, d7)
        recurrent_kernel_h_grad = np.matmul((x_r*prev_h).T, d8)
        #####################################################################################

        in_grad = [x_grad, prev_h_grad]
        kernel_grad = np.concatenate([kernel_z_grad, kernel_r_grad, kernel_h_grad], axis=-1)
        r_kernel_grad = np.concatenate([recurrent_kernel_z_grad, recurrent_kernel_r_grad, recurrent_kernel_h_grad], axis=-1)

        return in_grad, kernel_grad, r_kernel_grad


class softmax_cross_entropy(operator):
    def __init__(self):
        super(softmax_cross_entropy, self).__init__()

    def forward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            output: scalar, average loss
            probs: the probability of each category
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)

        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)
        output = -1 * np.sum(log_probs[np.arange(batch), labels]) / batch
        return output, probs

    def backward(self, input, labels):
        """
        # Arguments
            input: numpy array with shape (batch, num_class)
            labels: numpy array with shape (batch,)
            eps: float, precision to avoid overflow

        # Returns
            in_grad: gradient to forward input of softmax cross entropy, with shape (batch, num_class)
        """
        # precision to avoid overflow
        eps = 1e-12

        batch = len(labels)
        input_shift = input - np.max(input, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_shift), axis=1, keepdims=True)
        log_probs = input_shift - np.log(Z+eps)
        probs = np.exp(log_probs)

        in_grad = probs.copy()
        in_grad[np.arange(batch), labels] -= 1
        in_grad /= batch
        return in_grad

