import copy
import socket
import random
import os
import sys
import time
import gc
import torch
import numpy as np
from collections import defaultdict
import arguments
from attack.attack_with_REN import REN_attack
from loading import load_model_and_vnnlib, parse_run_mode, adhoc_tuning
from read_vnnlib import read_vnnlib
from utils import Logger, print_model
from specifications import (trim_batch, batch_vnnlib, sort_targets,
                            add_rhs_offset, RVN_specifications)

class RVN:
    def __init__(self, args=None, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, list):
                args.append(f'--{k}')
                args.extend(list(map(str, v)))
            elif isinstance(v, bool):
                if v:
                    args.append(f'--{k}')
                else:
                    args.append(f'--no_{k}')
            else:
                args.append(f'--{k}={v}')
        arguments.Config.parse_config(args)

    def main(self, interm_bounds=None):
        print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
        torch.manual_seed(arguments.Config['general']['seed'])
        random.seed(arguments.Config['general']['seed'])
        np.random.seed(arguments.Config['general']['seed'])
        torch.set_printoptions(precision=8)
        device = arguments.Config['general']['device']
        if device != 'cpu':
            torch.cuda.manual_seed_all(arguments.Config['general']['seed'])
            # Always disable TF32 (precision is too low for verification).
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if arguments.Config['general']['deterministic']:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True)
        if arguments.Config['general']['double_fp']:
            torch.set_default_dtype(torch.float64)
        # if arguments.Config['general']['precompile_jit']:
        #     precompile_jit_kernels()

        bab_args = arguments.Config['bab']
        timeout_threshold = bab_args['timeout']
        select_instance = arguments.Config['data']['select_instance']
        (run_mode, save_path, file_root, example_idx_list, model_ori,
        vnnlib_all, shape) = parse_run_mode()
        self.logger = Logger(run_mode, save_path, timeout_threshold)

        if arguments.Config['general']['return_optimized_model']:
            assert len(example_idx_list) == 1, (
                'To return the optimized model, only one instance can be processed'
            )

        model_ori.eval()
        model_ori = model_ori.to(device)
        rvn_specification = RVN_specifications()

        # load the whole specifications by vnnlib or vnn and get specifications for RVN(rvn_specification)
        for new_idx, csv_item in enumerate(example_idx_list):
            arguments.Globals['example_idx'] = new_idx
            vnnlib_id = new_idx + arguments.Config['data']['start']
            # Select some instances to verify
            if select_instance and not vnnlib_id in select_instance:
                continue
            if run_mode != 'customized_data':
                if len(csv_item) == 3:
                    # model, vnnlib, timeout
                    model_ori, shape, vnnlib, onnx_path = load_model_and_vnnlib(
                        file_root, csv_item)
                    arguments.Config['model']['onnx_path'] = os.path.join(file_root, csv_item[0])
                    arguments.Config['specification']['vnnlib_path'] = os.path.join(
                        file_root, csv_item[1])
                else:
                    # Each line contains only 1 item, which is the vnnlib spec.
                    vnnlib = read_vnnlib(os.path.join(file_root, csv_item[0]))
                    assert arguments.Config['model']['input_shape'] is not None, (
                        'vnnlib does not have shape information, '
                        'please specify by --input_shape')
                    shape = arguments.Config['model']['input_shape']
            else:
                vnnlib = vnnlib_all[new_idx]

            # get the specifications for RVN from vnnlib
            rvn_specification.load_from_vnnlib(vnnlib)

        # for norm, epsilon in zip():
        rvn_specification.update_norm_epsilon(arguments.Config['specification']['norm'], arguments.Config['rvn_setting']['epsilon'])
        # from RVN_code.Find_d import find_min_distance
        # d = find_min_distance(x_d, model_ori.lyapunov.R.data.detach().numpy(), model_ori.lyapunov(x))
        # construct the model for RVN
        # eps_up = torch.norm(
        #     torch.tensor([x[-1] for x in rvn_specification.Box], dtype=torch.float32).reshape((1, -1)),
        #     p=arguments.Config['specification']['norm']).item()
        # arguments.Config.update_eps(eps_up/2)
        if hasattr(model_ori, 'observer') is False:
            from RVN_code.RVN_calculation import RVN_cal
            from RVN_code.Lipschitz_calculation import LipschitzCalculation
            from RVN_code.KS_test import get_lipschitz_estimate
            from RVN_code.compute_delta import Delta_op
            rvn_model = RVN_cal(model_ori)
            start_time = time.time()
            L_calculation = LipschitzCalculation(rvn_model, rvn_specification,arguments.Config["rvn_setting"]["pure_sample_count"],
                                                     arguments.Config["rvn_setting"]["N_b"],arguments.Config["rvn_setting"]["N_s"],
                                                     arguments.Config["rvn_setting"]["steps"])

            # sample through rvn_specification and calculate the Lipschitz constant
            # calculate the Lyapunov value (from lift of states samples: 500 x 1024) respectively in prue and perturbed samples;
            # compute the delta between prue and perturbed Lyapunov, and / their delta(sample)
            # with the 'controller' and 'dynamics' in 'rvn_model'
            L_calculation.sample_pure_and_compute_Lq()
            # detect the eps_up and L robustness evaluation
            from RVN_code.Find_d import find_min_distance, find_eps_up
            L_list = find_min_distance(L_calculation.pure_sample, model_ori,rvn_specification.rho)
            eps_up_list = find_eps_up(L_calculation.pure_sample, rvn_specification.Box)
            # d_last = float('inf')
            delta_count_list = []
            for i in range(L_calculation.pure_sample_count):
                eps_up = eps_up_list[i]
                arguments.Config.update_eps(eps_up_list[i])
                # ks_dict_majo = get_lipschitz_estimate(L_calculation.majo[i], 'Fig_Majority')
                ks_dict_majo = get_lipschitz_estimate(L_calculation.L_q[i], 'Fig_Majority')
                loc_majo = -float(ks_dict_majo['loc'])
                pval_majo = float(ks_dict_majo['pVal'])
                d_l_majo = Delta_op(L_calculation, L_calculation.L_q[i], None, rvn_specification, loc_majo, pval_majo, i)
                d_op_majo = d_l_majo.delta[0]
                # d_l = Delta_op(L_calculation, L_calculation.L_q[i], None, rvn_specification, loc, pval, i)
                # d_op = d_l.delta[0]

                # the following is calculation delta for mino groups
                if L_calculation.mino[i][0] is not None:
                    ks_dict_mino = get_lipschitz_estimate(L_calculation.mino[i], 'Fig_Minority')
                    loc = -float(ks_dict_mino['loc'])
                    pval = float(ks_dict_mino['pVal'])
                    d_l_mino = Delta_op(L_calculation, None, L_calculation.mino[i], rvn_specification, loc,
                                         pval, i)
                    d_op_mino = d_l_mino.delta[0]
                else:
                    d_op_mino = 0

                d_op = max(d_op_mino, d_op_majo)
                # d_op = min(d_op,d_last)
                # d_last = d_op
                delta_count_list.append(min(d_op, eps_up))
                print(f'The min delta is {d_op} for the {i}-th sample.')


            end_time = time.time()
            framework_list = [max(a,b) for a,b in zip(delta_count_list, L_list)]
            mean_d = sum(framework_list) / arguments.Config["rvn_setting"]["pure_sample_count"]
            zero_L_record = 0
            delta_over_L = 0
            for delta, L in zip(delta_count_list, L_list):
                if L == 0:
                    zero_L_record += 1
                else:
                    if delta > L:
                        delta_over_L += 1

            L_calculation.reset_L_q()
            cloud_point(delta_count_list, L_list)

            # from attack.attack_with_REN import REN_attack
            # REN_attack(L_calculation.pure_sample, framework_list, vnnlib, model_ori, rvn_specification.Box,eps_up)
            # attack_time = time.time()
            # print(f'The total time of attack[REN] is {attack_time - start_time}.')
            print(f'The total time of REN is {end_time - start_time}.')
            print(f"The mean delta in REN framework is {mean_d}")
            print(f"There are {zero_L_record} samples cannot be dealt with by L, {delta_over_L} REN over L.")
            return d_op

        else:
            from RVN_code.RVN_calculate_observer import RVN_cal_obs
            from RVN_code.Lipschitz_calculation_observer import LipschitzCalculation
            from RVN_code.KS_test import get_lipschitz_estimate
            from RVN_code.compute_delta import Delta_op

            rvn_model = RVN_cal_obs(model_ori)
            start_time = time.time()
            L_calculation_obs = LipschitzCalculation(rvn_model, rvn_specification)
            # , arguments.Config["rvn_setting"]["pure_sample_count"],
            # arguments.Config["rvn_setting"]["N_b"], arguments.Config["rvn_setting"]["N_s"],
            # arguments.Config["rvn_setting"]["steps"]
            L_calculation_obs.sample_pure_and_compute_Lq()

            from RVN_code.Find_d import find_min_distance, find_eps_up
            L_list = find_min_distance(L_calculation_obs.pure_sample, model_ori, rvn_specification.rho)
            eps_up_list = find_eps_up(L_calculation_obs.pure_sample, rvn_specification.Box)
            # ks_dict = get_lipschitz_estimate(L_calculation_obs.L_q[0])
            # d_last = 100
            delta_count_list = []
            for i in range(L_calculation_obs.pure_sample_count):
                eps_up = eps_up_list[i]
                arguments.Config.update_eps(eps_up_list[i])

                ks_dict_majo = get_lipschitz_estimate(L_calculation_obs.majo[i], 'Fig_Majority')
                loc_majo = -float(ks_dict_majo['loc'])
                pval_majo = float(ks_dict_majo['pVal'])
                # d_op_majo = Delta_op(L_calculation, L_calculation.majo[i], None, rvn_specification, loc, pval, i)
                d_l_majo = Delta_op(L_calculation_obs, L_calculation_obs.L_q[i], None, rvn_specification, loc_majo, pval_majo, i)
                d_op_majo = d_l_majo.delta[0]

                # print(L_calculation_obs.mino[i])
                # if not(L_calculation_obs.mino[i] == 0).all():
                # if not all((x == [0] or x == 0) for x in L_calculation_obs.mino[i]):
                if L_calculation_obs.mino[i][0] is not None:
                    # d_op_majo = Delta_op(L_calculation, L_calculation.majo[i], None, rvn_specification, loc, pval, i)
                    ks_dict_mino = get_lipschitz_estimate(L_calculation_obs.mino[i], 'Fig_Minority')
                    loc_mino = -float(ks_dict_mino['loc'])
                    pval_mino= float(ks_dict_mino['pVal'])
                    d_l_mino = Delta_op(L_calculation_obs, L_calculation_obs.mino[i], None, rvn_specification, loc_mino,
                                        pval_mino, i)
                    d_op_mino = d_l_mino.delta[0]
                else:
                    d_op_mino = 0

                d_op = max(d_op_mino, d_op_majo)
                # d_op = min(d_op,d_last)
                # d_last = d_op
                delta_count_list.append(min(d_op,eps_up))
                print(f'The min delta is {d_op} for the {i}-th sample.')
            end_time = time.time()
            framework_list = [max(a, b) for a, b in zip(delta_count_list, L_list)]
            mean_d = sum(framework_list) / arguments.Config["rvn_setting"]["pure_sample_count"]
            zero_L_record = 0
            delta_over_L = 0
            for delta, L in zip(delta_count_list, L_list):
                if L == 0:
                    zero_L_record += 1
                else:
                    if delta > L:
                        delta_over_L += 1

            L_calculation_obs.reset_L_q()

            cloud_point(delta_count_list, L_list)
            # from attack.attack_with_REN import REN_attack
            # REN_attack(L_calculation_obs.pure_sample, framework_list, vnnlib, model_ori, rvn_specification.Box,eps_up)
            # attack_time = time.time()
            # print(f'The total time of attack[REN] is {attack_time - start_time}.')
            print(f'The total time of REN is {end_time - start_time}.')
            print(f"The mean delta in REN framework is {mean_d}")
            print(f"There are {zero_L_record} samples cannot be dealt with by L, {delta_over_L} REN over L.")
            return d_op




def cloud_point(list1, list2):
    import matplotlib.pyplot as plt
    n = len(list1)

    # 创建idx列表
    idx = list(range(n))

    # 绘制点云图
    plt.figure()
    # 分别绘制list1和list2中的点
    plt.scatter(idx, list1, color='blue', label='REN')
    plt.scatter(idx, list2, color='red', label='L')

    # 添加图例
    plt.legend()

    # 设置标题和坐标轴标签
    plt.title('CloudPoint REN and L')
    plt.xlabel('idx')
    plt.ylabel('min distortion')

    # 保存图表
    plt.savefig('pointcloudplot.png')


if __name__ == '__main__':
    # ['abcrown.py', '--config', '/home/txz/RVN_tool/verification/pendulum_output_feedback_lyapunov_in_levelset.yaml'], sys.argv = ['abcrown.py', '--config', '/home/txz/Lyapunov_Stable_NN_Controllers-main/verification/path_tracking_state_feedback_lyapunov_in_levelset.yaml']
    path_list = [
                 ['abcrown.py', '--config', '/home/txz/RVN_tool/verification/pendulum_state_feedback_lyapunov_in_levelset.yaml']]

    d_op_filelist = []
    for arg in path_list:
        sys.argv = arg
        rvn = RVN(args=sys.argv[1:])
        d_op = rvn.main()
        d_op_filelist.append(d_op)
    # d_op_filelist is the list for each path in path_list
    print(d_op_filelist)

