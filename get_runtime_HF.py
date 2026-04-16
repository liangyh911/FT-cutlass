import numpy as np
import os

def get_preparation_time(fp):
    prep = np.loadtxt(f"{fp}/preparation.txt")
    gemm = np.loadtxt(f"{fp}/gemm.txt")
    bgemm = np.loadtxt(f"{fp}/bgemm.txt")
    
    update_path = f"{fp}/update.txt"
    # update 可能不存在或为空
    if os.path.exists(update_path) and os.path.getsize(update_path) > 0:
        update = np.loadtxt(update_path)

        # 保证形状一致
        if np.size(update) == np.size(bgemm):
            bgemm = np.maximum(bgemm, update)
    
    group_size = 9
    num_groups = len(prep) // group_size
    
    attn_preparation = 0
    mlp_preparation = 0
    
    
    for i in range(num_groups):
        p = prep[i*9:(i+1)*9]
        g = gemm[i*7:(i+1)*7]
        b = bgemm[i*2:(i+1)*2]
    
        nonkernel_1 = p[:6].sum() - (g[:4].sum() + b.sum())
        nonkernel_2 = p[6:].sum() - g[4:].sum()
    
        attn_preparation += nonkernel_1
        mlp_preparation += nonkernel_2
        # print(f"group {i}:")
        # print(f"  non-kernel part1 = {nonkernel_1:.3f} ms")
        # print(f"  non-kernel part2 = {nonkernel_2:.3f} ms")
    
    attn_pre = attn_preparation/num_groups
    mlp_pre = mlp_preparation/num_groups

    total = attn_preparation + mlp_preparation

    return attn_pre, mlp_pre, total

def get_attn_mlp_time(file):
    fp = open(file, 'r')
    Lines = fp.readlines()
    
    cnt = 0
    res = 0
    for idx, line in enumerate(Lines):
        tmp = float(line)
        res += tmp
        cnt += 1
    return res / cnt

def get_GEMM_kernel_time(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]

    time = {"WQ":0, "WK":0, "WV":0, "WO":0, "UP":0, "GA":0, "DO":0}
    cnt = len(data)/7
    for idx, element in enumerate(data):
        if idx % 7 == 0:
            time["WQ"] += element
        elif idx % 7 == 1:
            time["WK"] += element
        elif idx % 7 == 2:
            time["WV"] += element
        elif idx % 7 == 3:
            time["WO"] += element
        elif idx % 7 == 4:
            time["UP"] += element
        elif idx % 7 == 5:
            time["GA"] += element
        elif idx % 7 == 6:
            time["DO"] += element
        
    for k in time:
        time[k] = time[k] / cnt

    return time

def get_BGEMM_kernel_time(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]

    time = {"QK_GEMM":0, "AV_GEMM":0}

    cnt = len(data)/2
    for idx, element in enumerate(data):
        if idx % 2 == 0:
            time["QK_GEMM"] += element
        elif idx % 2 == 1:
            time["AV_GEMM"] += element
        
    for k in time:
        time[k] = time[k] / cnt

    return time

def get_update_kernel_time(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]

    time = {"QK_Update":0, "AV_Update":0}

    cnt = len(data)/2
    for idx, element in enumerate(data):
        if idx % 2 == 0:
            time["QK_Update"] += element
        elif idx % 2 == 1:
            time["AV_Update"] += element
        
    for k in time:
        time[k] = time[k] / cnt

    return time

def main():
    Attn = f"./control/0/time/attn.txt"
    attn = get_attn_mlp_time(Attn)*1000

    MLP = f"./control/0/time/mlp.txt"
    mlp = get_attn_mlp_time(MLP)*1000

    Training = f"./control/0/time/training.txt"
    training = get_attn_mlp_time(Training)*1000

    attn_pre, mlp_pre, training_pre = get_preparation_time(f"./control/0/time")

    # print(f"Core_Checker: Attn: {attn, (attn-attn_pre)}; MLP: {mlp, (mlp-mlp_pre)}, Training: {training, training-training_pre}")

    with open("./control/training.txt") as file:
        file.write(training - training_pre)
    with open("./control/attn.txt") as file:
        file.write(attn-attn_pre)
    with open("./control/mlp.txt") as file:
        file.write(mlp-mlp_pre) 


if __name__ == "__main__":
    main()