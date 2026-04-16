import numpy as np
import os

def get_preparation_time(fp):
    prep = np.loadtxt(f"{fp}/preparation.txt")
    gemm = np.loadtxt(f"{fp}//gemm.txt")
    bgemm = np.loadtxt(f"{fp}//bgemm.txt")
    
    update_path = f"{fp}/update.txt"
    # update 可能不存在或为空
    if os.path.exists(update_path) and os.path.getsize(update_path) > 0:
        update = np.loadtxt(update_path)

        # 保证形状一致
        if np.size(update) == np.size(bgemm):
            bgemm = np.maximum(bgemm, update)
    
    group_size = 6
    num_groups = len(prep) // group_size
    
    attn_preparation = 0
    mlp_preparation = 0
    
    
    for i in range(num_groups):
        p = prep[i*6:(i+1)*6]
        g = gemm[i*4:(i+1)*4]
        b = bgemm[i*2:(i+1)*2]
    
        nonkernel_1 = p[:4].sum() - (g[:2].sum() + b.sum())
        nonkernel_2 = p[4:].sum() - g[2:].sum()
    
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

    time = {"WA":0, "WO":0, "UP":0, "DO":0}
    cnt = len(data)/7
    for idx, element in enumerate(data):
        if idx % 7 == 0:
            time["WA"] += element
        elif idx % 7 == 1:
            time["WO"] += element
        elif idx % 7 == 2:
            time["UP"] += element
        elif idx % 7 == 3:
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

def model_performance():
    num_gpus = 4
    
    # 用于存储所有 GPU 的数据
    all_attn = []
    all_mlp = []
    all_attn_pre = []
    all_mlp_pre = []
    all_training_pre = []
    
    # 1. 遍历收集 4 个 GPU 的数据
    for gpu_id in range(num_gpus):
        path_prefix = f"./control/{gpu_id}/time"
        
        # 获取 Attn 和 MLP 的原始时间 (转为 ms)
        attn_raw = get_attn_mlp_time(f"{path_prefix}/attn.txt") * 1000
        mlp_raw = get_attn_mlp_time(f"{path_prefix}/mlp.txt") * 1000
        
        # 获取该 GPU 的准备时间
        attn_p, mlp_p, train_p = get_preparation_time(path_prefix)
        
        all_attn.append(attn_raw)
        all_mlp.append(mlp_raw)
        all_attn_pre.append(attn_p)
        all_mlp_pre.append(mlp_p)
        all_training_pre.append(train_p)
    
    # 2. 处理全局训练时间 (假设 training.txt 在根目录下)
    # 注意：PP=2 时，真正的“准备完成”时刻是最后一个 Stage 准备好的时刻
    training_file = f"./control/training.txt"
    training_raw = get_attn_mlp_time(training_file) * 1000
    
    # 核心逻辑：取 4 个 GPU 中最大的准备时间作为流水线真正开始“稳态训练”的起点
    # max_training_pre = max(all_training_pre)
    pp_rank_1_train_pre = max(all_training_pre[:2])
    pp_rank_2_train_pre = max(all_training_pre[2:])
    
    
    actual_training_time = training_raw - (pp_rank_1_train_pre + pp_rank_2_train_pre)
    
    # 3. 打印结果
    # print(f"=== {method} Performance (TP=2, PP=2) ===")
    max_attn = 0
    max_mlp = 0
    for i in range(num_gpus):
        # 计算每个 GPU 自己的实际计算耗时
        actual_attn = all_attn[i] - all_attn_pre[i]
        actual_mlp = all_mlp[i] - all_mlp_pre[i]

        max_attn =  max(max_attn, actual_attn)
        max_mlp = max(max_mlp, actual_mlp)
        
        # print(f"GPU {i} | Attn: {all_attn[i]:.4f}ms (Actual: {actual_attn:.4f}ms) | "
        #       f"MLP: {all_mlp[i]:.4f}ms (Actual: {actual_mlp:.4f}ms)")
    
    with open("./control/training.txt") as file:
        file.write(actual_training_time)
    with open("./control/attn.txt") as file:
        file.write(max_attn)
    with open("./control/mlp.txt") as file:
        file.write(max_mlp) 

    # print("-" * 50)
    # print(f"Total Training Time (Raw): {training_raw:.4f}ms")
    # print(f"Pipeline Pipeline Max Prep Time: {(pp_rank_1_train_pre + pp_rank_2_train_pre):.4f}ms")
    # print(f"Actual Training Time (Steady State): {actual_training_time:.4f}ms")

if __name__ == "__main__":
    main()