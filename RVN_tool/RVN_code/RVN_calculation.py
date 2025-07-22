import copy
import torch
from numpy import ndarray
import arguments


class RVN_cal:

    def __init__(self, input_model):
        # target_model = copy.deepcopy(input_model.state_dict())
        self.dynamics = copy.deepcopy(input_model.dynamics)
        self.kappa = float(input_model.kappa)
        self.controller = copy.deepcopy(input_model.controller)
        self.lyapunov = copy.deepcopy(input_model.lyapunov)
        self.state = torch.ones(size=[1, input_model.controller.layers[0].in_features])
        self.state = self.state.float()
        self.action = None
        self.network_output = torch.ones(size=self.state.size())
        self.rho = None
        self.norm = 2

    def once_calculate_action(self, input_state):
        """
        once controller process, dynamic function: state_t to action_t
        :param input_state: state_t
        :return: action_t+1
        """

        action = self.controller(input_state)
        return action

    def total_calculate_state(self):
        """
            calculate the final action through time: T
        :return:
        """
        # TODO: define the domain of the input state_0
        step = float(self.dynamics.dt)
        time_total = arguments.Config["rvn_setting"]["steps"] * step

        time_points = [i * step for i in range(int(time_total / step) + 1)]


        for t in time_points:
            # u = f(x) - f(x*) + u*
            if self.norm == 2:
                network_output = torch.nn.functional.pairwise_distance(self.state, self.lyapunov.goal_state, p=2)

            elif self.norm == 1:
                network_output = torch.nn.functional.pairwise_distance(self.state, self.lyapunov.goal_state, p=1)

            else:
                network_output = torch.nn.functional.pairwise_distance(self.state, self.lyapunov.goal_state,
                                                                       p=float('inf'))
            if network_output is not None:
                if network_output.item() <= 0.0001:
                    break
            # self.network_output = self.once_calculate_action(self.state) - self.once_calculate_action(self.lyapunov.goal_state)
            self.action = self.once_calculate_action(self.state)
            # state_update for path tracking tasks
            self.state = self.dynamics.forward(self.state, self.action)

            # self.linearized_dynamics_pathtracking()

        return self.state


    def calculate_lyapunov(self,x):
        """
        lyapunov function used in forward feedback path-tracking tasks
        :return:
        """
        lyapunov_value = self.lyapunov(x)
        return lyapunov_value

    # def V_psd(self):
    #     """
    #     Compute
    #     |(εI+RᵀR)(x-x*)|₁
    #     or
    #     (x-x*)ᵀ(εI+RᵀR)(x-x*)
    #     or
    #     |R(x-x*)|₁
    #     """
    #     x = copy.deepcopy(self.state.T)
    #     if self.lyapunov.R_rows > 0:
    #         eps_plus_RtR = self.lyapunov.eps * torch.eye(self.lyapunov.x_dim, device=x.device) + (
    #             self.lyapunov.R.transpose(0, 1) @ self.lyapunov.R
    #         )
    #         if self.lyapunov.V_psd_form == "L1":
    #             Rx = (x - self.lyapunov.goal_state).float() @ eps_plus_RtR
    #             # Use relu(x) + relu(-x) instead of torch.abs(x) since the verification code does relu splitting.
    #             l1_term = (
    #                 torch.nn.functional.relu(Rx) + torch.nn.functional.relu(-Rx)
    #             ).sum(dim=-1, keepdim=True)
    #             return l1_term
    #         elif self.lyapunov.V_psd_form == "quadratic":
    #             return torch.sum(
    #                 (x - self.lyapunov.goal_state) * ((x - self.lyapunov.goal_state) @ eps_plus_RtR),
    #                 dim=-1,
    #                 keepdim=True,
    #             )
    #         elif self.lyapunov.V_psd_form == "L1_R_free":
    #             Rx = (x - self.lyapunov.goal_state) @ self.lyapunov.R.transpose(0, 1)
    #             # Use relu(x) + relu(-x) instead of torch.abs(x) since the verification code does relu splitting.
    #             l1_term = (
    #                 torch.nn.functional.relu(Rx) + torch.nn.functional.relu(-Rx)
    #             ).sum(dim=-1, keepdim=True)
    #             return l1_term
    #         else:
    #             raise NotImplementedError
    #     else:
    #         return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

    def torch2state(self, torch_value):
        if isinstance(torch_value, torch.Tensor):
            self.state = copy.deepcopy(torch_value).float()
        elif isinstance(torch_value, list):
            self.state = torch.tensor(torch_value).float()
        elif isinstance(torch_value, ndarray):
            self.state = torch.from_numpy(torch_value).float()
        else:
            print("The value is neither list or torch or array!")
        self.state = self.state.reshape(shape=[1, self.controller.layers[0].in_features]).detach()

    def reset_state(self):
        self.state = torch.ones(size=[1, self.controller.layers[0].in_features])
        self.action = None
        self.network_output = torch.ones(size=[1, self.controller.layers[0].in_features])
