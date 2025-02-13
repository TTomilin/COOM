from collections import deque


class PIDLagrangian:
    """PID version of Lagrangian.

    Similar to the :class:`Lagrange` module, this module implements the PID version of the
    lagrangian method.

    .. note::
        The PID-Lagrange is more general than the Lagrange, and can be used in any policy gradient
        algorithm. As PID_Lagrange use the PID controller to control the lagrangian multiplier, it
        is more stable than the naive Lagrange.

    Args:
        pid_kp (float): The proportional gain of the PID controller.
        pid_ki (float): The integral gain of the PID controller.
        pid_kd (float): The derivative gain of the PID controller.
        pid_d_delay (int): The delay of the derivative term.
        pid_delta_p_ema_alpha (float): The exponential moving average alpha of the delta_p.
        pid_delta_d_ema_alpha (float): The exponential moving average alpha of the delta_d.
        sum_norm (bool): Whether to use the sum norm.
        diff_norm (bool): Whether to use the diff norm.
        penalty_max (int): The maximum penalty.
        lagrangian_multiplier_init (float): The initial value of the lagrangian multiplier.
        cost_limit (float): The cost limit.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Adam Stooke, Joshua Achiam, Pieter Abbeel.
        - URL: `PID Lagrange <https://arxiv.org/abs/2007.03964>`_
    """

    def __init__(
            self,
            pid_kp: float,
            pid_ki: float,
            pid_kd: float,
            pid_d_delay: int,
            pid_delta_p_ema_alpha: float,
            pid_delta_d_ema_alpha: float,
            sum_norm: bool,
            diff_norm: bool,
            penalty_max: int,
            lagrangian_multiplier_init: float,
            cost_limit: float,
    ) -> None:
        """Initialize an instance of :class:`PIDLagrangian`."""
        self._pid_kp: float = pid_kp
        self._pid_ki: float = pid_ki
        self._pid_kd: float = pid_kd
        self._pid_d_delay = pid_d_delay
        self._pid_delta_p_ema_alpha: float = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = pid_delta_d_ema_alpha
        self._penalty_max: int = penalty_max
        self._sum_norm: bool = sum_norm
        self._diff_norm: bool = diff_norm
        self._pid_i: float = lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_limit: float = cost_limit
        self._cost_penalty: float = 0.0

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        return self._cost_penalty

    def pid_update(self, ep_cost_avg: float) -> float:
        r"""Update the PID controller.

        PID controller update the lagrangian multiplier following the next equation:

        .. math::

            \lambda_{t+1} = \lambda_t + (K_p e_p + K_i \int e_p dt + K_d \frac{d e_p}{d t}) \eta

        where :math:`e_p` is the error between the current episode cost and the cost limit,
        :math:`K_p`, :math:`K_i`, :math:`K_d` are the PID parameters, and :math:`\eta` is the
        learning rate.

        Args:
            ep_cost_avg (float): The average cost of the current episode.
        """
        # Calculate the error between the average cost of the episode and the cost limit.
        delta = float(ep_cost_avg - self._cost_limit)

        # Update the integral part of the PID controller by adding the error multiplied by the integral gain.
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)

        # If differential normalization is active, clamp the integral term between 0 and 1.
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))

        # Calculate the exponentially weighted moving average of the proportional error.
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta

        # Calculate the exponentially weighted moving average of the derivative of cost.
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)

        # Calculate the derivative term of the PID controller, taking into account the historical data.
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])

        # Compute the overall PID output by combining proportional, integral, and derivative terms.
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d

        # Determine the new Lagrange multiplier, ensuring it is non-negative.
        self._cost_penalty = max(0.0, pid_o)

        # If differential normalization is used, ensure the Lagrange multiplier is between 0 and 1.
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)

        # If neither differential nor sum normalization is active, cap the Lagrange multiplier at a maximum value.
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)

        # Update the historical cost data queue with the latest derived cost.
        self._cost_ds.append(self._cost_d)

        return delta
