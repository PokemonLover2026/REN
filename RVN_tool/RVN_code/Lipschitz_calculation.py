import copy
from numpy import ndarray
import numpy as np
import torch
from multiprocessing import Pool
from functools import partial

from sympy.physics.paulialgebra import epsilon

import arguments
import random

class LipschitzCalculation:
    def __init__(self, model, specifications,pure_sample_count, N_b, N_s, step_num):
        self.model = copy.deepcopy(model)
        self.specifications = copy.deepcopy(specifications)
        self.pure_sample_count = pure_sample_count
        self.N_b = N_b
        self.N_s = N_s
        self.R = 5
        self.perturbation_input = 0
        self.x_pure_sample = 0
        self.attack_switch = False
        self.step_num = step_num

        # tuple(state_t from perturbed prue state_0: tensor, Lyapunov value of state_t: tensor)
        self.pure_sample = []
        self.pure_lyapunov = [None] * self.pure_sample_count

        self.perturbed_sample = []
        self.perturbed_lyapunov = []
        self.eps_up = []
        self.L_q = [[0] * self.N_b for _ in range(self.pure_sample_count)]
        # self.L_lyapunov = [[0] * self.N_b for _ in range(self.pure_sample_count)]
        # the self.majo and self.mino will be added with a list like self.majo.append(List), the count of list is pure_sample_count
        # self.majo = []
        # self.mino = []
        self.majo = [[None] * 1 for _ in range(self.pure_sample_count)]
        self.mino = [[None] * 1 for _ in range(self.pure_sample_count)]

    def sample_pure_and_compute_Lq(self):
        self.reset_perturbed_samples()
        self.reset_pure_samples()
        # sample the prue input_sample x_0
        for i in range(self.pure_sample_count):
            random.seed(22+i)
            self.x_pure_sample = (np.array([np.random.uniform(low, high) for low, high in self.specifications.Box]))
            self.x_pure_sample = torch.from_numpy(self.x_pure_sample).float()
            self.x_pure_sample = self.x_pure_sample.reshape(1, self.x_pure_sample.size()[0])
            self.x_pure_sample = self.x_pure_sample.detach()

            while torch.all(self.model.lyapunov(self.x_pure_sample) > self.specifications.rho):
                self.x_pure_sample = (np.array([np.random.uniform(low, high) for low, high in self.specifications.Box]))
                self.x_pure_sample = torch.from_numpy(self.x_pure_sample).float()
                self.x_pure_sample = self.x_pure_sample.reshape(1, self.x_pure_sample.size()[0])
                self.x_pure_sample = self.x_pure_sample.detach()
            self.pure_sample.append(self.x_pure_sample)
        # pool:
        # self.calculate_prue_lyapunov()
        self.eps_up = self.find_eps()
        # for nb in range(self.N_b):
        for count in range(self.pure_sample_count):
            x_pure_example = np.array(self.pure_sample[count])
            pure_info = self.calculation_lyapunov(x_pure_example)
            pure_state_t, pure_lyapunov_t = pure_info
            pure_lyapunov_t = pure_lyapunov_t.detach()
            self.pure_lyapunov[count] = pure_lyapunov_t
            arguments.Config.update_eps(self.eps_up[count])
            for nb in range(self.N_b):
                self.reset_perturbed_samples()
                # sample N_s perturbed samples around pure sample x_0
                print(f'Now is the {nb}-th perturbed samples.')

                results = self.multiprocess_perturbed_sample_start(nb, count, pure_info)


                if max(results) is None :
                    self.L_q[count][nb] = 0.0
                else:
                    self.L_q[count][nb] = max(results)

            self.L_q[count]= list(filter(lambda x: x != 0, self.L_q[count]))

        self.L_KM()

    def L_KM(self):
        # 创建KMeans对象
        from sklearn.cluster import KMeans
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_point_list = []

        # 对数据进行拟合
        for i in range(self.pure_sample_count):
            L = copy.deepcopy(self.L_q)
            data = np.array(L[i])
            standard_deviation = np.std(data,ddof=1)
            data = data.reshape(-1, 1)
            kmeans.fit(data)

            # 获取聚类中心
            cluster_center = kmeans.cluster_centers_

            if (cluster_center.max() - cluster_center.min()) < 0.001 or n_clusters == 1:
                self.majo.append(data)

            else:
            # 预测每个数据点的簇标签
                labels = kmeans.predict(data)
                label_counts = np.bincount(labels)
                most_frequent_label = np.argmax(label_counts)
                least_frequent_label = np.argmin(label_counts)
                # 打印聚类中心
                print("Cluster center:\n", cluster_center)
                data_majo = data[labels == most_frequent_label]
                self.majo.append(data_majo)
                # mino_array = np.setdiff1d(data, data[labels == most_frequent_label])
                # self.mino.append([i] for i in mino_array)
                data_mino = data[labels != most_frequent_label]
                self.mino.append(data_mino)

    def perturbation_domain(self):
        n = len(self.specifications.Box)  # 获取维度
        x = np.array(self.x_pure_sample)
        random_vector = np.random.randn(*x.shape)
        norm_random_vector = random_vector / np.linalg.norm(random_vector, ord=self.specifications.norm)
        scaled_random_vector = norm_random_vector * self.specifications.epsilon
        x_n = x + scaled_random_vector
        # if self.specifications.norm == 2:
        #     perturbation = np.random.randn(n)
        #     perturbation /= np.linalg.norm(perturbation)
        #     perturbation *= self.specifications.epsilon
        # else:
        #     while True:
        #         perturbation = np.random.uniform(-self.specifications.epsilon, self.specifications.epsilon, n)
        #         if np.linalg.norm(perturbation, ord=self.specifications.norm) <= self.specifications.epsilon:
        #             break
        # x_n = self.x_pure_sample + perturbation
        return x_n

    def cal_lyapunov_start(self,nb):

        # 这下面的model可跟RVN_model不是一个东西了，请注意
        self.calculate_perturbed_lyapunov()

        # calculate the abs(V(state_0_pure) - V(state_0_perturbed)) / norm_p(state_0_pure - state_0_perturbed)
        self.calculate_L_q(nb)

    def calculate_perturbed_lyapunov(self):
        self.reset_perturbed_lyapunov()
        for sample in self.perturbed_sample:
            # one_sample_L = []
            for state_0 in sample:
                state_t, lyapunov_value = self.calculation_lyapunov(state_0)
                self.perturbed_lyapunov.append((state_t,lyapunov_value))
                self.model.reset_state()
                # self.perturbed_lyapunov.append(one_sample_L)

    def calculate_pure_lyapunov(self):
        self.reset_pure_lyapunov()
        for state_idx in range(len(self.prue_sample)):
            self.model.torch2state(self.prue_sample[state_idx])
            state_t, lyapunov_value = self.calculation_lyapunov(self.model.state)
            self.pure_lyapunov.append((state_t.detach(),lyapunov_value.detach()))
            self.model.reset_state()

    def calculation_lyapunov(self, state_0):
        self.model.torch2state(state_0)
        xe_t = self.model.total_calculate_state()
        lyapunov_value = self.model.calculate_lyapunov(xe_t)
        return xe_t, lyapunov_value

    def reset_pure_samples(self):
        self.prue_sample = []

    def reset_pure_lyapunov(self):
        self.pure_lyapunov = [None] * self.pure_sample_count

    def reset_perturbed_samples(self):
        self.perturbed_sample = []

    def reset_perturbed_lyapunov(self):
        self.perturbed_lyapunov = []

    def reset_L_q(self):
        self.L_q = [[None] * self.pure_sample_count for _ in range(self.N_b)]

    def calculate_L_q(self,nb):

        rho = self.specifications.rho
        pure_state_t, pure_lyapunov = self.pure_lyapunov[0]
        # pure_state_t, pure_lyapunov = prue_info
        for ns in range(self.N_s):

            # perturbed_batch = self.perturbed_lyapunov[nb]
            perturbed_batch = self.perturbed_lyapunov[ns]


            perturbed_state_t, perturbed_lyapunov =  perturbed_batch
            # perturbed_state_t = perturbed_info

            if self.specifications.norm == 2:
                L_q = abs(pure_lyapunov - perturbed_lyapunov) / torch.norm(pure_state_t - perturbed_state_t, p=2)
                # L_q = abs(pure_lyapunov - perturbed_lyapunov)

            elif self.specifications.norm == 1:
                L_q = abs(pure_lyapunov - perturbed_lyapunov) / torch.norm(pure_state_t - perturbed_state_t, p=float('inf'))
                # L_q = abs(pure_lyapunov - rho) / torch.norm(pure_state_t - perturbed_state_t, p=float('inf'))
            else:
                L_q = abs(pure_lyapunov - perturbed_lyapunov) / torch.norm(pure_state_t - perturbed_state_t, p=1)
                # L_q = abs(pure_lyapunov - rho) / torch.norm(pure_state_t - perturbed_state_t, p=1)

            if L_q > 0.5:
                return 0

            # Compare to find min(L_q)
            if self.L_q[nb] == None or self.L_q[nb] > L_q:
                self.L_q[nb] = float(L_q.data)

    def multiprocess_perturbed_sample_start(self, nb, count, pure_info):
        pure_state_t, pure_lyapunov_t = pure_info
        pure_lyapunov_t = pure_lyapunov_t.detach()
        norm = float(self.specifications.norm)
        epsilon = arguments.Config['rvn_setting']['epsilon']
        # prue_info = self.prue_lyapunov
        # pure_state_t = self.pure_sample[nb]
        # pure_lyapunov = pure_lyapunov.detach()
        N_s = int(self.N_s)
        step = float(self.model.dynamics.dt)
        goal_state = self.model.lyapunov.goal_state.detach()
        forward = copy.deepcopy(self.model.dynamics.forward)
        controller = copy.deepcopy(self.model.controller)
        size_of_state = int(self.model.controller.layers[0].in_features)
        lyapunov = copy.deepcopy(self.model.lyapunov)
        rho = float(self.specifications.rho)
        x_pure_example = self.pure_sample[count]
        # random_perturbations = np.random.uniform(-epsilon, epsilon, (N_s, 1))
        # perturbed_samples = x_pure_example + random_perturbations.reshape(-1,1)
        Box = copy.deepcopy(self.specifications.Box)

        samples = [np.random.uniform(low, high, size=N_s) for low, high in Box]
        samples_array = np.array(samples).T
        perturbed_samples = torch.tensor(samples_array, device='cpu')

        x_pure_example = torch2state(x_pure_example, size_of_state)
        step_num = self.step_num

        with Pool(processes=15, maxtasksperchild=20) as pool:
            res = pool.map(partial(multiprocess_perturbed_sample,norm,x_pure_example, pure_lyapunov_t,
                             goal_state, step, step_num, forward, controller, lyapunov, Box, size_of_state, rho),
                     perturbed_samples)

        pool.close()
        pool.join()
        return res

    def find_eps(self):
        max_corner = np.asarray([max(a) for a in self.specifications.Box])
        min_corner = np.asarray([min(a) for a in self.specifications.Box])
        eps_up_list = []
        for point in self.pure_sample:
            point = point.detach().numpy()
            dist_to_min = point - min_corner
            dist_to_max = max_corner - point
            min_distances = np.minimum(dist_to_min, dist_to_max)
            eps_up = np.min(min_distances)
            eps_up_list.append(eps_up)

        return eps_up_list

def multiprocess_perturbed_sample(norm, pure_state_0, pure_lyapunov,
                                  goal_state, step, step_num, forward, controller, lyapunov, Box, size_of_state, rho,
                                  perturbed_samples):
    # rng = random.Random()

    perturbed_state = torch2state(perturbed_samples, size_of_state)


    for i in range(len(Box)):
        dimension = Box[i]
        if perturbed_state[0, i] < dimension[0]:
            perturbed_state[0, i] = dimension[0]
        elif perturbed_state[0, i] > dimension[1]:
            perturbed_state[0, i] = dimension[1]


    perturbed_state_t = total_calculate_state(perturbed_state, goal_state, step, step_num, forward, controller, norm)
    perturbed_state_t = perturbed_state_t.detach()
    perturbed_lyapunov = calculate_lyapunov(perturbed_state_t,lyapunov)
    perturbed_lyapunov = perturbed_lyapunov.detach()
    if torch.all(perturbed_lyapunov > rho):
        return 0.0
    # if torch.all(perturbed_lyapunov >= 0.0001):
    #     return 0.0
    # if torch.all(perturbed_lyapunov <= 0.0001) and arguments.Config['rvn_setting']['steps'] <= 500:
    #     return 0.0
    #
    # if torch.all(perturbed_lyapunov > 0.0001) and arguments.Config['rvn_setting']['steps'] > 500:
    #     return 0.0
    if norm == 2:
        L_q = torch.nn.functional.pairwise_distance(perturbed_lyapunov,pure_lyapunov, p=2) / torch.nn.functional.pairwise_distance(perturbed_state,pure_state_0, p=2)
    elif norm == 1:
        L_q = torch.nn.functional.pairwise_distance(perturbed_lyapunov,pure_lyapunov, p=float('inf')) / torch.nn.functional.pairwise_distance(perturbed_state, pure_state_0, p=float('inf'))
    else:
        L_q = torch.nn.functional.pairwise_distance(perturbed_lyapunov,pure_lyapunov, p=1) / torch.nn.functional.pairwise_distance(perturbed_state, pure_state_0, p=1)

    # distance = torch.nn.functional.pairwise_distance(perturbed_state,  controller.x_equilibrium)
    # if L_q >0.25:
    #     return 0.0
    L = copy.deepcopy(L_q.item())

    return L


def total_calculate_state(state, goal_state, step, step_num, forward, controller, norm):
    network_output = None
    time_total = step_num * step
    time_points = [i * step for i in range(int(time_total / step) + 1)]

    for t in time_points:
        # u = f(x) - f(x*) + u*

        if network_output is not None and network_output.item() <= 0.0001:
            break
        # self.network_output = self.once_calculate_action(self.state) - self.once_calculate_action(self.lyapunov.goal_state)
        action = once_calculate_action(state,controller)

        # state_update for path tracking tasks
        state = forward(state, action)

        if norm == 2:
            network_output = torch.nn.functional.pairwise_distance(state, goal_state, p=2)

        elif norm == 1:
            network_output = torch.nn.functional.pairwise_distance(state, goal_state, p=1)

        else:
            network_output = torch.nn.functional.pairwise_distance(state, goal_state, p=float('inf'))

        # self.linearized_dynamics_pathtracking()

    return state

def once_calculate_action(input_state, controller):
    action = controller(input_state)
    return action

def torch2state(torch_value, size_of_state):
    if isinstance(torch_value, torch.Tensor):
        state = torch_value.float()
    elif isinstance(torch_value, list):
        state = torch.tensor(torch_value).float()
    elif isinstance(torch_value, ndarray):
        state = torch.from_numpy(torch_value).float()
    else:
        print("The value is neither list or torch or array!")
    state = state.reshape(shape=[1, size_of_state]).detach()
    return state

def calculate_lyapunov(state, lyapunov):
    lyapunov_value = lyapunov(state)
    return lyapunov_value


