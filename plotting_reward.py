import torch
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────
# Plot Style
# ─────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 12,
    "lines.linewidth": 1.2,
    "figure.dpi": 150
})

# ============================
# 파일 경로
# ============================
ddpg_path = "./results_compare/ddpg/RL/ddpg_RL.pt"
td3_path  = "./results_compare/ddpg/RL/td3_RL.pt"   # <--- 경로 맞게 수정 필요


# ============================
# EMA smoothing
# ============================
def smooth_curve(x, alpha=0.15):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


# ============================
# Load 결과
# ============================
ddpg_data = torch.load(ddpg_path, weights_only=False)
td3_data  = torch.load(td3_path,  weights_only=False)

ddpg_reward = np.array(ddpg_data['reward_ep'])
td3_reward  = np.array(td3_data['reward_ep'])

ddpg_smooth = smooth_curve(ddpg_reward, alpha=0.12)
td3_smooth  = smooth_curve(td3_reward,  alpha=0.12)


# ============================
# Plot 비교
# ============================
fig, ax = plt.subplots(figsize=(6.0, 4.3))

mark_every = 5

# DDPG
ax.plot(ddpg_reward, color='blue', alpha=0.3, label='DDPG Raw')
ax.plot(ddpg_smooth, '-', color='blue', label='DDPG Smoothed')
ax.plot(
    np.arange(len(ddpg_smooth))[::mark_every],
    ddpg_smooth[::mark_every],
    's', color='blue', ms=6
)

# TD3
ax.plot(td3_reward, color='red', alpha=0.3, label='TD3 Raw')
ax.plot(td3_smooth, '-', color='red', label='TD3 Smoothed')
ax.plot(
    np.arange(len(td3_smooth))[::mark_every],
    td3_smooth[::mark_every],
    'o', color='red', ms=6
)

# Plot style
ax.set_xlabel("Episode")
ax.set_ylabel("Average Reward")
ax.grid(True, ls='--', lw=0.7, alpha=0.8)
ax.tick_params(direction='in', top=True, right=True)
ax.legend(loc='best', frameon=True, edgecolor='black')

max_len = max(len(ddpg_reward), len(td3_reward))
ax.set_xlim(0, max_len - 1)

fig.tight_layout()
plt.show()
