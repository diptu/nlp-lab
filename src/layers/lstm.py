import numpy as np


class LSTMCell:
    def __init__(self, in_features, hidden_dim):
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Xavier initialization
        limit = np.sqrt(6 / (in_features + hidden_dim + 4 * hidden_dim))
        self.W = np.random.uniform(
            -limit, limit, (in_features + hidden_dim, 4 * hidden_dim)
        )
        self.b = np.zeros((1, 4 * hidden_dim))
        # Forget gate bias = 1.0
        self.b[:, hidden_dim : 2 * hidden_dim] = 1.0

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache = None

    def zero_grad(self):
        self.dW.fill(0)
        self.db.fill(0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, x, h_prev, c_prev):
        combined = np.hstack([x, h_prev])
        gates = np.dot(combined, self.W) + self.b
        d = self.hidden_dim
        i = self.sigmoid(gates[:, :d])
        f = self.sigmoid(gates[:, d : 2 * d])
        g = np.tanh(gates[:, 2 * d : 3 * d])
        o = self.sigmoid(gates[:, 3 * d :])
        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)
        self.cache = (combined, i, f, g, o, c_prev, c_next)
        return h_next, c_next

    def backward(self, grad_h_next, grad_c_next):
        """
        grad_h_next: (batch_size, hidden_dim)
        grad_c_next: (batch_size, hidden_dim)
        """
        combined, i, f, g, o, c_prev, c_next = self.cache
        d = self.hidden_dim

        # 1. Backprop through output gate
        # h_next = o * tanh(c_next)
        grad_o = grad_h_next * np.tanh(c_next)
        grad_o_raw = grad_o * o * (1 - o)

        # 2. Backprop through c_next (from tanh and input/forget gates)
        grad_c_next += grad_h_next * o * (1 - np.tanh(c_next) ** 2)

        # 3. Backprop through f, i, g
        grad_f = grad_c_next * c_prev
        grad_f_raw = grad_f * f * (1 - f)

        grad_i = grad_c_next * g
        grad_i_raw = grad_i * i * (1 - i)

        grad_g = grad_c_next * i
        grad_g_raw = grad_g * (1 - g**2)

        # Concatenate gate gradients
        grad_gates = np.hstack([grad_i_raw, grad_f_raw, grad_g_raw, grad_o_raw])

        # 4. Accumulate weight and bias gradients (CRITICAL: Use +=)
        self.dW += np.dot(combined.T, grad_gates)
        self.db += np.sum(grad_gates, axis=0, keepdims=True)

        # 5. Gradient w.r.t input and previous states
        grad_combined = np.dot(grad_gates, self.W.T)
        grad_x = grad_combined[:, : self.in_features]
        grad_h_prev = grad_combined[:, self.in_features :]
        grad_c_prev = grad_c_next * f

        return grad_x, grad_h_prev, grad_c_prev
