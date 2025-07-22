import copy
import torch
from numpy import ndarray
import arguments

class RVN_cal_obs():

    def __init__(self, input_model):
        self.controller = copy.deepcopy(input_model.controller)
        self.dynamics = copy.deepcopy(input_model.dynamics)
        self.lyapunov = copy.deepcopy(input_model.lyapunov)
        self.observer = copy.deepcopy(input_model.observer)
        self.kappa = float(input_model.kappa)
        self.fuse_dV = input_model.fuse_dV

        self.last_lyapunov_x = None

        self.xe = torch.ones(size=[1, input_model.controller.layers[0].in_features])
        self.xe = self.xe.float()
        self.xe = self.xe.detach()

        self.rho = None
        self.norm = 2

        self.new_xe = None

        self.nx = input_model.nx

        # additional terms
        self.action = None

        self.detach_system()

    def detach_system(self):
        # print(f'Before detach, the requires_grad is {self.controller.u_lo.requires_grad}')
        for key, value in self.controller.__dict__.items():
            # 检查属性是否是tensor
            if isinstance(value, torch.Tensor):
                # 对tensor进行detach操作

                setattr(self.controller, key, value.detach())

        for key, value in self.dynamics.__dict__.items():
            # 检查属性是否是tensor
            if isinstance(value, torch.Tensor):
                # 对tensor进行detach操作

                setattr(self.dynamics, key, value.detach())

        for key, value in self.lyapunov.__dict__.items():
            # 检查属性是否是tensor
            if isinstance(value, torch.Tensor):
                # 对tensor进行detach操作

                setattr(self.lyapunov, key, value.detach())

        for key, value in self.observer.__dict__.items():
            # 检查属性是否是tensor
            if isinstance(value, torch.Tensor):
                # 对tensor进行detach操作

                setattr(self.observer, key, value.detach())

        # print(f'After detach, the requires_grad is {self.controller.u_lo.requires_grad}')

    def total_calculate_xe(self):
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
                network_output = torch.nn.functional.pairwise_distance(self.xe, self.lyapunov.goal_state, p=2)

            elif self.norm == 1:
                network_output = torch.nn.functional.pairwise_distance(self.xe, self.lyapunov.goal_state, p=1)

            else:
                network_output = torch.nn.functional.pairwise_distance(self.xe, self.lyapunov.goal_state,
                                                                       p=float('inf'))
            if network_output is not None:
                if network_output.item() <= 0.0001:
                    break
            # self.network_output = self.once_calculate_action(self.state) - self.once_calculate_action(self.lyapunov.goal_state)
            self.forward()

        return self.xe

    def forward(self):
        xe = copy.deepcopy(self.xe)
        xe = xe.reshape(1,xe.size()[-1])
        x = xe[:,:self.nx]
        e = xe[:, self.nx:]
        z = x - e
        y = self.observer.h(x)
        ey = y - self.observer.h(z)
        u = self.controller.forward(torch.cat((z, ey), dim=1))
        # u = self.controller(z)
        new_x = self.dynamics.forward(x, u)
        new_z = self.observer.forward(z, u, y)
        lyapunov_x = self.lyapunov(xe)
        # Save the results for reference.
        self.last_lyapunov_x = lyapunov_x.detach()
        self.xe = torch.cat((new_x, new_x - new_z), dim=1)
        self.xe = self.xe.detach()
        # self.xe = copy.deepcopy(self.new_xe)
        # self.new_xe = self.new_xe.detach()


    def torch2state(self, torch_value):
        if isinstance(torch_value, torch.Tensor):
            self.xe = copy.deepcopy(torch_value).float()
        elif isinstance(torch_value, list):
            self.xe = torch.tensor(torch_value).float()
        elif isinstance(torch_value, ndarray):
            self.xe = torch.from_numpy(torch_value).float()
        else:
            print("The value is neither list or torch or array!")
        self.xe = self.xe.reshape(shape=self.xe.shape).detach()