import numpy as np


class Adam:
    def __init__(
        self,
        params_dict: dict,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # Initialize moments: m for mean (1st), v for variance (2nd)
        self.m = {k: np.zeros_like(v) for k, v in params_dict.items()}
        self.v = {k: np.zeros_like(v) for k, v in params_dict.items()}

    def update(self, params: dict, grads: dict):
        """
        Updates parameters in-place using Adam rule.
        params: dict of {name: weight_matrix}
        grads: dict of {name: weight_gradient}
        """
        self.t += 1

        for k in params:
            # 1. Update 1st moment (momentum)
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]

            # 2. Update 2nd moment (RMSprop-like scaling)
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)

            # 3. Compute bias-corrected moments
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)

            # 4. Update parameter
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
