import time
import torch
from attack.attack_pgd import attack
import numpy as np
import arguments

def REN_attack(pure_sample_list, delta_list, vnnlib, model_ori,Box,eps_up):
    attack_delta_record = []
    start_time = time.time()
    for i in range(len(pure_sample_list)):
        x = pure_sample_list[i]
        eps = delta_list[i]
        verified_status = "unsafe-pgd"
        verified_success = False

        x_new_range = [(low, up) for low, up in zip((x - eps).numpy(), (x + eps).numpy())]
        for i in range(len(Box)):
            dimension = Box[i]
            x_new_range[0][0][i] = max(x_new_range[0][0][i], dimension[0])
            x_new_range[0][1][i] = min(x_new_range[0][1][i], dimension[1])
            #
            # if x_new_range[0][i] < dimension[0]:
            #     x_new_range[0][i] = dimension[0]
            # elif x_new_range[0][i] > dimension[1]:
            #     x_new_range[0][i] = dimension[1]

        vnnlib = [tuple([x_new_range, vnnlib[0][1]])]



        x = x.detach()

        verified_status, verified_success, attack_images, attack_margins, all_adv_candidates, pgd_input = attack(
            model_ori, x,
            vnnlib, verified_status, verified_success, crown_filtered_constraints=None, initialization='uniform')
        if verified_success:
            attack_delta = torch.norm(x - pgd_input, p=arguments.Config["specification"]["norm"])
        else:
            attack_delta = eps_up
        attack_delta_record.append(attack_delta)

    mean_value = sum(attack_delta_record) / arguments.Config["rvn_setting"]["pure_sample_count"]
    print(f"The mean attack delta is {mean_value}.")
    mean_end_time = (time.time() - start_time) / arguments.Config["rvn_setting"]["pure_sample_count"]
    print(f"The mean run time is {mean_end_time}.")

    return