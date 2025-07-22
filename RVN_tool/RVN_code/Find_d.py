import copy

import numpy as np
from scipy.optimize import minimize
import torch
def find_min_distance(x_list,model,rho):
    # 定义二次型曲面参数，例如椭球面: x^2/4 + y^2/9 + z^2 = 1
    L_list = []
    # nx = model.nx
    for i in range(len(x_list)):
        if hasattr(model, 'observer') is False:
            x = x_list[i].detach().numpy()
            A = model.lyapunov.R.data.detach().numpy()
            A = A.T @ A # 二次项系数矩阵
            b = np.ones(x.shape) * model.kappa              # 一次项系数向量
            c = 0                        # 常数项
            # 初始猜测点（建议选择靠近预期解的点，如沿y方向的投影）
            x0 = x.flatten()  # 对于椭球面，初始点可能在表面附近
            # 目标点

            y = x  # 例如，点 (3, 0, 0)
            lx = model.lyapunov(torch.from_numpy(x).float())
            if rho - lx.detach().numpy()[0] < 0 :
                L_list.append(0.0)
                continue
            # 目标函数：最小化距离平方
            def objective(x):
                return np.sum((x - y)**2)

            # 约束条件：二次型函数等于0
            constraint = {'type': 'eq', 'fun': lambda x: rho - x @ A @ x.T - b @ x.T }

            # 使用SLSQP方法进行优化
            result = minimize(objective, x0, method='SLSQP', constraints=[constraint])
            min_distance = np.sqrt(result.fun)
            if result.success:
                pass
            else:
                print("优化失败，请尝试调整初始点或检查参数设置。")
                min_distance = 0.0
            L_list.append(min_distance)
        else:
            x = x_list[i].detach().numpy()
            A = model.lyapunov.R.data.detach().numpy()
            A = A @ A.T  # 二次项系数矩阵
            b = np.ones(x.shape) * model.kappa  # 一次项系数向量
            c = 0  # 常数项
            # 初始猜测点（建议选择靠近预期解的点，如沿y方向的投影）
            x0 = x.flatten()  # 对于椭球面，初始点可能在表面附近
            # 目标点

            y = x  # 例如，点 (3, 0, 0)
            lx = model.lyapunov(torch.from_numpy(x).float())
            if rho - lx.detach().numpy()[0] < 0:
                L_list.append(0.0)
                continue

            # 目标函数：最小化距离平方
            def objective(x):
                return np.sum((x - y) ** 2)

            # 约束条件：二次型函数等于0
            constraint = {'type': 'eq', 'fun': lambda x: rho - x @ A @ x.T - b @ x.T}

            # 使用SLSQP方法进行优化
            result = minimize(objective, x0, method='SLSQP', constraints=[constraint])
            min_distance = np.sqrt(result.fun)
            # jac = result.jac.reshape(x.shape)
            # min_distance = np.sqrt(np.sum((x[:,:2] - jac[:,:2]) ** 2))
            if result.success:
                pass
            else:
                print("优化失败，请尝试调整初始点或检查参数设置。")
                min_distance = 0.0
            L_list.append(min_distance)

    return L_list

def find_eps_up(x_list,Box):

    max_corner = np.asarray([max(a) for a in Box])
    min_corner = np.asarray([min(a) for a in Box])
    eps_up_list = []
    for point in x_list:
        point = point.detach().numpy()
        dist_to_min = point - min_corner
        dist_to_max = max_corner - point
        min_distances = np.minimum(dist_to_min, dist_to_max)
        eps_up = np.min(min_distances)
        eps_up_list.append(eps_up)

    return eps_up_list
