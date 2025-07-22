import copy
from numpy import ndarray
import numpy as np
import torch
from multiprocessing import Pool
from functools import partial
from sklearn.cluster import KMeans
import arguments
import random


class LipschitzCalculation:
    def __init__(self, model, specifications):
        self.model = copy.deepcopy(model)
        self.specifications = copy.deepcopy(specifications)
        self.pure_sample_count = arguments.Config["rvn_setting"]["pure_sample_count"]
        self.N_b = arguments.Config["rvn_setting"]["N_b"]
        self.N_s = arguments.Config["rvn_setting"]["N_s"]
        self.R = 5
        self.perturbation_input = 0
        self.x_pure_sample = 0
        self.attack_switch = False

        # tuple(state_t from perturbed prue state_0: tensor, Lyapunov value of state_t: tensor)
        self.pure_sample = []
        self.pure_lyapunov = [None] * self.pure_sample_count

        self.perturbed_sample = []
        self.perturbed_lyapunov = []
        self.eps_up = []
        self.L_q = [[0] * self.N_b for _ in range(self.pure_sample_count)]
        # the self.majo and self.mino will be added with a list like self.majo.append(List), the count of list is pure_sample_count
        self.majo = []
        self.mino = []

    def sample_pure_and_compute_Lq(self):
        self.reset_perturbed_samples()
        self.reset_pure_samples()
        # sample the prue input_sample x_0
        for i in range(self.pure_sample_count):
            random.seed(22+i)
            self.x_pure_sample = (np.array([np.random.uniform(low, high) for low, high in self.specifications.Box]))
            self.x_pure_sample = torch.from_numpy(self.x_pure_sample).float()
            self.x_pure_sample = self.x_pure_sample.reshape(1,self.x_pure_sample.size()[0])
            self.x_pure_sample = self.x_pure_sample.detach()
            while torch.all(self.model.lyapunov(self.x_pure_sample) > self.specifications.rho):
                # np.random.seed(2+j)
                self.x_pure_sample = (np.array([np.random.uniform(low, high) for low, high in self.specifications.Box]))
                self.x_pure_sample = torch.from_numpy(self.x_pure_sample).float()
                self.x_pure_sample = self.x_pure_sample.reshape(1, self.x_pure_sample.size()[0])
                self.x_pure_sample = self.x_pure_sample.detach()
                # j += 1
            # self.x_pure_sample[-self.model.nx:] = 0
            self.pure_sample.append(self.x_pure_sample)
        # pool:
        # self.calculate_prue_lyapunov()

        # for nb in range(self.N_b):
        self.eps_up = self.find_eps()
        for count in range(self.pure_sample_count):
            x_pure_example = np.array(self.pure_sample[count])
            pure_info = self.calculation_lyapunov(x_pure_example)
            pure_state_t, pure_lyapunov_t = pure_info
            self.pure_lyapunov[count] = pure_lyapunov_t
            arguments.Config.update_eps(self.eps_up[count])
            for nb in range(self.N_b):
                self.reset_perturbed_samples()
                # sample N_s perturbed samples around pure sample x_0
                print(f'Now is the {nb}-th perturbed samples.')
                results = self.multiprocess_perturbed_res(nb, count, pure_info)

                # L_q = [t.cpu().detach() for t in results]
                # try:
                #     L_q_array = [t.numpy()[0] for t in L_q]
                # except:
                #     L_q_array = [t.numpy() for t in L_q]
                # L_q = np.stack(L_q_array)
                # self.L_q[count][nb] = self.L_KM(results)
                if results == []:
                    self.L_q[count][nb] = 0
                else:
                    self.L_q[count][nb] = max(results)

            self.L_q[count]= list(filter(lambda x: x != 0, self.L_q[count]))

        self.L_KM()
            # self.L_KM()

    def L_KM(self):
        # 创建KMeans对象
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
                # cluster_points = data[labels == most_frequent_label]
                # cluster_point = np.amax(self.data)
                # cluster_point = cluster_center[most_frequent_label]
                # cluster_point_list.append(cluster_point)

                # L = copy.deepcopy(self.majo[i])
                # data = np.array(L)
                #
                # data = data.reshape(-1, 1)
                # kmeans.fit(data)
                #
                # # 获取聚类中心
                # cluster_center = kmeans.cluster_centers_
                #
                #
                # # 预测每个数据点的簇标签
                # labels = kmeans.predict(data)
                # label_counts = np.bincount(labels)
                # most_frequent_label = np.argmax(label_counts)
                # least_frequent_label = np.argmin(label_counts)
                # # 打印聚类中心
                # print("Cluster center:\n", cluster_center)
                #
                # self.majo[i] = data[labels == most_frequent_label]

                # cluster_points = data[labels == most_frequent_label]
                # cluster_point = np.amax(self.data)
                # cluster_point = cluster_center[most_frequent_label]
                # cluster_point_list.append(cluster_point)



            # self.L_q[count] = self.L_KM(self.L_q[count])

    def calculation_lyapunov(self, xe_0):
        self.model.torch2state(xe_0)
        xe_t = self.model.total_calculate_xe()
        lyapunov_value = self.model.last_lyapunov_x
        xe_t = xe_t.detach()
        lyapunov_value = lyapunov_value.detach()
        return xe_t, lyapunov_value

    def multiprocess_perturbed_res(self, nb, count, pure_info):
        pure_state_t, pure_lyapunov_t = pure_info
        pure_lyapunov_t = pure_lyapunov_t.detach()
        norm = float(self.specifications.norm)
        # epsilon = arguments.Config['rvn_setting']['epsilon']
        N_s = int(self.N_s)
        step = float(self.model.dynamics.dt)
        goal_state = self.model.lyapunov.goal_state.detach()
        dy_forward = copy.deepcopy(self.model.dynamics.forward)
        controller = copy.deepcopy(self.model.controller)
        size_of_state = pure_state_t.size()[-1]
        lyapunov = copy.deepcopy(self.model.lyapunov)
        dynamics = copy.deepcopy(self.model.dynamics)
        # obs_h = copy.deepcopy(self.model.observer.h)
        obs_fc_net = copy.deepcopy(self.model.observer.fc_net)
        rho = float(self.specifications.rho)
        nx = int(self.model.nx)
        x_pure_example = self.pure_sample[count]
        # random_perturbations = np.random.uniform(-epsilon, epsilon, (N_s, 1))
        # perturbed_samples = x_pure_example + random_perturbations.reshape(-1,1)
        Box = copy.deepcopy(self.specifications.Box)

        samples = [np.random.uniform(low, high, size=N_s) for low, high in Box]
        # samples = [np.random.beta(low, high, size=N_s) for low, high in Box]
        samples_array = np.array(samples).T
        perturbed_samples = torch.tensor(samples_array, device='cpu')

        x_pure_example = torch2state(x_pure_example, x_pure_example.size()[-1])

        with Pool(processes=15, maxtasksperchild=10) as pool:
            res = pool.map(partial(multiprocess_perturbed_sample,norm,x_pure_example, pure_lyapunov_t,
                             goal_state, step, dy_forward, obs_fc_net, controller, dynamics, lyapunov, Box, size_of_state, rho, nx),
                     perturbed_samples)


        pool.close()
        pool.join()
        res = list(filter(lambda x: x != 0, res))
        # res is L_q list
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

    def reset_pure_samples(self):
        self.pure_sample = []

    def reset_perturbed_samples(self):
        self.perturbed_sample = []

    def reset_pure_lyapunov(self):
        self.pure_lyapunov = [None] * self.pure_sample_count

    def reset_perturbed_lyapunov(self):
        self.perturbed_lyapunov = []

    def reset_L_q(self):
        self.L_q = [[None] * self.pure_sample_count for _ in range(self.N_b)]

def multiprocess_perturbed_sample(norm, pure_state_0, pure_lyapunov,
                                  goal_state, step, dy_forward, obs_fc_net, controller, dynamics, lyapunov, Box, size_of_state, rho,
                                  nx, perturbed_samples):
    perturbed_state = torch2state(perturbed_samples, size_of_state)

    for i in range(len(Box)):
        dimension = Box[i]
        if perturbed_state[0, i] < dimension[0]:
            perturbed_state[0, i] = dimension[0]
        elif perturbed_state[0, i] > dimension[1]:
            perturbed_state[0, i] = dimension[1]

    perturbed_lyapunov = total_calculate_state(perturbed_state, goal_state, step, dy_forward, obs_fc_net, controller, dynamics, lyapunov, norm, nx)

    perturbed_lyapunov = perturbed_lyapunov.detach()
    if torch.all(perturbed_lyapunov > rho):
        return 0.0

    if norm == 2:
        L_q = torch.nn.functional.pairwise_distance(perturbed_lyapunov,pure_lyapunov, p=2) / torch.nn.functional.pairwise_distance(perturbed_state[:,:nx],pure_state_0[:,:nx], p=2)
    elif norm == 1:
        L_q = torch.nn.functional.pairwise_distance(perturbed_lyapunov,pure_lyapunov, p=float('inf')) / torch.nn.functional.pairwise_distance(perturbed_state[:,:nx], pure_state_0[:,:nx], p=float('inf'))
    else:
        L_q = torch.nn.functional.pairwise_distance(perturbed_lyapunov,pure_lyapunov, p=1) / torch.nn.functional.pairwise_distance(perturbed_state, pure_state_0, p=1)
    # if torch.all(L_q > 0.1):
    #     return 0.0
    L = copy.deepcopy(L_q.item())
    return L

def total_calculate_state(state, goal_state, step, dy_forward, obs_fc_net, controller, dynamics, lyapunov, norm, nx):
    network_output = None
    time_total = arguments.Config["rvn_setting"]["steps"] * step
    time_points = [i * step for i in range(int(time_total / step) + 1)]
    xe = state
    for t in time_points:
        # u = f(x) - f(x*) + u*

        if network_output is not None and network_output.item() <= 0.0001:
            break
        # self.network_output = self.once_calculate_action(self.state) - self.once_calculate_action(self.lyapunov.goal_state)

        xe = xe.reshape(1, xe.size()[-1])
        x = xe[:, :nx]
        e = xe[:, nx:]
        z = x - e
        y = obs_h(x,dynamics)
        ey = y - obs_h(z,dynamics)
        u = controller.forward(torch.cat((z, ey), dim=1))
        # u = self.controller(z)
        new_x = dy_forward(x, u)
        new_z = obs_forward(z, u, y,dynamics, obs_fc_net)
        lyapunov_x = lyapunov(xe)
        # Save the results for reference.
        last_lyapunov_x = lyapunov_x.detach()
        new_xe = torch.cat((new_x, new_x - new_z), dim=1)

        xe = new_xe
        new_xe = new_xe.detach()
        # action = once_calculate_action(state,controller)
        #
        # # state_update for path tracking tasks
        # state = dy_forward(state, action)

        if norm == 2:
            network_output = torch.nn.functional.pairwise_distance(new_xe, goal_state, p=2)

        elif norm == 1:
            network_output = torch.nn.functional.pairwise_distance(new_xe, goal_state, p=1)

        else:
            network_output = torch.nn.functional.pairwise_distance(new_xe, goal_state, p=float('inf'))

        # self.linearized_dynamics_pathtracking()

    return last_lyapunov_x


def obs_h(x, dynamics):
    h = dynamics.continuous_time_system.h(x)
    return h

def obs_forward(z,u,y, dynamics, obs_fc_net):
    zero_obs_error = torch.zeros((1,y.size(-1)), device=z.device)
    batch_size = z.shape[0]
    K = torch.ones((batch_size, 1), device=z.device)
    z_nominal = dynamics(z, u)
    obs_error = y - obs_h(z, dynamics)
    Le = obs_fc_net(torch.cat((z, obs_error), 1))
    L0 = obs_fc_net(torch.cat((z, (K*zero_obs_error).to(z.device)), 1))
    unclipped_z = z_nominal + Le - L0
    return unclipped_z

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
