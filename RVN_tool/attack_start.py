import copy
import socket
import random
import os
import sys
import time
import torch
import numpy as np
import arguments
from loading import load_model_and_vnnlib, parse_run_mode, adhoc_tuning
from read_vnnlib import read_vnnlib
from utils import Logger, print_model
from specifications import (trim_batch, batch_vnnlib, sort_targets,
                            add_rhs_offset, RVN_specifications)
from attack.attack_pgd import attack
from multiprocessing import Pool
from functools import partial

class RVN_attack:
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
        # torch.manual_seed(arguments.Config['general']['seed'])
        # random.seed(arguments.Config['general']['seed'])
        # np.random.seed(arguments.Config['general']['seed'])
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
            rvn_specification.load_from_vnnlib(vnnlib)
        
        pure_sample_acount = arguments.Config['rvn_setting']['pure_sample_count']
        vnn_shape = shape

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
            vnnlib = vnnlib_all[new_idx]  # vnnlib_all is a list of all standard vnnlib
        # rvn_specification.load_from_vnnlib(vnnlib)

        attack_delta_record = []
        start_time = time.time()

        for i in range(pure_sample_acount):
            np.random.seed(22+i)
            verified_status = "unsafe-pgd"
            verified_success = False

            x = (np.array([np.random.uniform(low, high) for low, high in rvn_specification.Box]))
            x = torch.from_numpy(x).float()
            x = x.reshape(1, x.size()[0])
            while torch.all(model_ori.lyapunov(x) > rvn_specification.rho):
                x = (np.array([np.random.uniform(low, high) for low, high in rvn_specification.Box]))
                x = torch.from_numpy(x).float()
                x = x.reshape(1, x.size()[0])
                x = x.detach()
            from RVN_code.Find_d import find_min_distance, find_eps_up
            eps_up_list = find_eps_up([x], rvn_specification.Box)
            arguments.Config.update_eps(eps_up_list[0])
            eps_up = eps_up_list[0]
            eps = arguments.Config["rvn_setting"]["epsilon"]
            if eps != 0:
                x_new_range = [(low, up) for low, up in zip((x - eps).numpy(), (x + eps).numpy())]
                vnnlib = [tuple([x_new_range, vnnlib[0][1]])]
            else:
                vnnlib = [tuple([rvn_specification.Box, vnnlib[0][1]])]
            x = x.detach()

            verified_status, verified_success, attack_images, attack_margins, all_adv_candidates, pgd_input = attack(model_ori, x,
               vnnlib, verified_status, verified_success, crown_filtered_constraints=None, initialization='uniform')
            # if verified_success:
            #     attack_delta = torch.norm(x - pgd_input, p=arguments.Config["specification"]["norm"])
            # else:
            #     attack_delta = eps_up
            attack_delta = torch.norm(x - pgd_input, p=arguments.Config["specification"]["norm"])
            attack_delta_record.append(attack_delta)

        mean_value = sum(attack_delta_record)/arguments.Config["rvn_setting"]["pure_sample_count"]
        print(f"The mean attack delta is {mean_value}.")
        mean_end_time = (time.time() - start_time)/arguments.Config["rvn_setting"]["pure_sample_count"]
        print(f"The mean run time is {mean_end_time}.")

        return

    

if __name__ == '__main__':
    # ['abcrown.py', '--config', '/home/txz/RVN_tool/verification/pendulum_output_feedback_lyapunov_in_levelset.yaml'], sys.argv = ['abcrown.py', '--config', '/home/txz/Lyapunov_Stable_NN_Controllers-main/verification/path_tracking_state_feedback_lyapunov_in_levelset.yaml']
    path_list = [
                 ['abcrown.py', '--config', '/home/txz/RVN_tool/verification/pendulum_state_feedback_lyapunov_in_levelset.yaml']]

    for arg in path_list:
        sys.argv = arg
        rvn = RVN_attack(args=sys.argv[1:])
        rvn.main()

