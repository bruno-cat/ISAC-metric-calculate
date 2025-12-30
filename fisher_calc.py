import numpy as np
import matplotlib.pyplot as plt


def calculate_fim_and_crlb(bs_list, target_pos, zeta_sq=1e4, beta=2):
    """
    计算 Fisher Information Matrix (FIM) 和 Position Error Bound (PEB)
    对应论文公式 (5), (6), (7)
    """
    N = len(bs_list)
    F_N = np.zeros((2, 2))  # 初始化 2x2 的 FIM 矩阵

    # 预先计算每个基站相对于目标的 距离 和 角度
    # 这里的 d_vec 是从 目标 指向 基站 的向量
    d_infos = []
    for bs in bs_list:
        vec = bs - target_pos
        dist = np.linalg.norm(vec)
        # arctan2 计算角度 theta (弧度制)
        angle = np.arctan2(vec[1], vec[0]) #y坐标与x坐标的比值
        d_infos.append({'dist': dist, 'angle': angle})

    # --- 核心双重循环 (对应公式 5 的双重求和) ---
    for i in range(N):
        for j in range(N):
            # 获取基站 i 和 j 的信息
            info_i = d_infos[i]
            info_j = d_infos[j]

            # 1. 计算路径损耗权重 (对应 ||di||^-beta * ||dj||^-beta)
            # 注意：距离越远，权重越小
            path_loss_weight = (info_i['dist'] ** -beta) * (info_j['dist'] ** -beta)

            # 2. 计算几何方向因子 (对应公式 6 和 7)
            # a_ij = cos(theta_i) + cos(theta_j)
            # b_ij = sin(theta_i) + sin(theta_j)
            a_ij = np.cos(info_i['angle']) + np.cos(info_j['angle'])
            b_ij = np.sin(info_i['angle']) + np.sin(info_j['angle'])

            # 3. 构建几何矩阵 (对应公式 5 中括号内的矩阵)
            # [ a^2   ab ]
            # [ ab    b^2]
            geo_matrix = np.array([
                [a_ij ** 2, a_ij * b_ij],
                [a_ij * b_ij, b_ij ** 2]
            ])

            # 累加到总 FIM 中
            F_N += path_loss_weight * geo_matrix

    # 乘以常数系数 |zeta|^2
    F_N = zeta_sq * F_N

    # --- 计算 CRLB (FIM 的逆矩阵) ---
    try:
        CRLB = np.linalg.inv(F_N)
        # PEB (位置误差界) 通常定义为 trace(CRLB) 的平方根，单位是米
        PEB = np.sqrt(np.trace(CRLB))
    except np.linalg.LinAlgError:
        print("矩阵奇异，无法求逆 (基站共线或数量不足)")
        CRLB = None
        PEB = float('inf')

    return F_N, CRLB, PEB

# ================= 运行测试 =================

# 1. 定义坐标 (单位: 米)
# 假设基站呈三角形分布，目标在中间偏右
bs_locations = np.array([
    [0, 0],  # BS 1
    [50, 50],  # BS 2
    [50, -50], # BS 3
    [100, 0],
    [0, 50]
])
target_location = np.array([30, 10])

# 2. 设定参数
# zeta_sq 包含了发射功率、天线增益等，这里假设一个较大的值保证矩阵数值稳定
# beta = 2 (自由空间衰减)
zeta_squared = 1e3
path_loss_exp = 2.0

# 3. 执行计算
FIM, CRLB_matrix, peb_val = calculate_fim_and_crlb(bs_locations, target_location, zeta_squared, path_loss_exp)

# 4. 打印结果
print(f"--- 场景设置 ---")
print(f"基站数量: {len(bs_locations)}")
print(f"目标位置: {target_location}")
print("-" * 30)

print(f"计算得到的 FIM (Fisher Information Matrix):\n{FIM}")
print("-" * 30)

if CRLB_matrix is not None:
    print(f"CRLB 矩阵 (FIM的逆):\n{CRLB_matrix}")
    print("-" * 30)
    print(f"理论定位误差下界 (PEB): {peb_val:.4f} 米")
    print(f"(这意味着在这个配置下，最好的估计算法误差也不会低于这个值)")