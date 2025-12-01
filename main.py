import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from channel import (
    load_and_process_data,
    generate_w,
    cal_loss,
    linear_to_db
)

#from ddpg import DDPGAgent as Agent
from TD3 import TD3Agent as Agent

# ============================================================
# 환경 설정
# ============================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = 'cpu'
torch.set_default_dtype(torch.float32)

K, M, N = 1, 8, 32
lr = 1e-3
sigma2 = 1e-2
TaudB = 10
Tau = 10 ** (TaudB / 10)
batch_size = 200
Episode = 30
gamma = 0.99
tau_soft = 0.01
PdB = 10
P = 10 ** (PdB / 10)

alpha_corr = 1.0
beta_w = 0.8
lambda_ = 0.5

SNR_com_th_dB = TaudB
eps_corr = 1e-8

# ==========================================
# Train/Test 데이터
# ==========================================
Train_Files = [
    f"C:/Users/CNL-A2/Desktop/DL-Beamforming-RIS-ISAC-main/channel/data/Train_Channel_{i:03d}.mat"
    for i in range(1, 30)
]

Test_File = "C:/Users/CNL-A2/Desktop/DL-Beamforming-RIS-ISAC-main/channel/data/Test_Channel.mat"


# ============================================================
# 데이터 불러오기
# ============================================================
train_sets = []
for f in Train_Files:
    R_sep_scaled, origin, originH, num = load_and_process_data(f, N, M)
    train_sets.append((R_sep_scaled.to(device), origin.to(device), originH.to(device), num))

R_test_sep_scaled, origin_dataset_test, origin_datasetH_test, test_num = \
    load_and_process_data(Test_File, N, M, is_test=True)

R_test_sep_scaled = R_test_sep_scaled.to(device).float()
origin_dataset_test = origin_dataset_test.to(device)
origin_datasetH_test = origin_datasetH_test.to(device)

# ============================================================
# 💡 STATE DIM 수정: prev_SNRt + prev_SNRc → 2개 추가
# ============================================================
state_dim = 4 * N * N + 2


# ============================================================
# 학습
# ============================================================
if __name__ == "__main__":

    agent = Agent(
        N=N, device=device, lr=lr, gamma=gamma, tau=tau_soft,
        use_ou_noise=True, ou_rho=0.9, ou_sigma=0.2
    )

    episode_reward_log = []
    episode_snrt_log = []
    episode_snrc_log = []
    train_losses = []

    for e in range(Episode):

        agent.reset_noise()
        agent.actor.train()

        total_reward = 0.0
        step_count = 0

        SNRt_ep_list = []
        SNRc_ep_list = []

        # 여러 Trajectory 반복
        for (R_sep_scaled, origin_dataset, origin_datasetH, total_num) in train_sets:

            # 초기 prev SNR
            prev_SNRt = 0.0
            prev_SNRc = 0.0

            for t in range(5, total_num):

                # --------------------------------------------------
                # 1) STATE 구성
                # --------------------------------------------------
                R_t5 = R_sep_scaled[t - 5]

                state_vec = torch.cat([
                    R_t5.reshape(-1),
                    torch.tensor([prev_SNRt, prev_SNRc],
                                 dtype=torch.float32, device=device)
                ], dim=0)

                state_batch = state_vec.unsqueeze(0)


                # --------------------------------------------------
                # 2) ACTION 선택
                # --------------------------------------------------
                theta_pred, thetaH_pred = agent.select_action(state_batch, noise=True)


                # --------------------------------------------------
                # 3) Channel 계산
                # --------------------------------------------------
                y_chan = origin_dataset[t].unsqueeze(0)
                yH_chan = origin_datasetH[t].unsqueeze(0)

                _, hc_batch, ht_batch, Ht_batch = cal_loss(
                    y_chan, yH_chan, theta_pred, thetaH_pred, M, N
                )

                ht_i = ht_batch[0]
                hc_i = hc_batch[0]
                Ht_i = Ht_batch[0]


                # --------------------------------------------------
                # 4) w 생성
                # --------------------------------------------------
                w = generate_w(ht_i, hc_i, P, Tau, sigma2)


                # --------------------------------------------------
                # 5) SNR 계산
                # --------------------------------------------------
                SNRt_lin = torch.linalg.norm(Ht_i @ w)**2 / sigma2
                SNRc_lin = torch.abs(w.conj().T @ hc_i)**2 / sigma2

                SNRt_dB = linear_to_db(SNRt_lin)
                SNRc_dB = linear_to_db(SNRc_lin)

                SNRt_ep_list.append(SNRt_dB.item())
                SNRc_ep_list.append(SNRc_dB.item())


                # --------------------------------------------------
                # 6) reward 계산
                # --------------------------------------------------
                norm_ht2 = torch.linalg.norm(ht_i)**2
                norm_hc2 = torch.linalg.norm(hc_i)**2
                corr_num = torch.abs(torch.conj(ht_i).T @ hc_i)**2
                rho_t = corr_num / (norm_ht2 * norm_hc2 + eps_corr)

                penalty = torch.nn.functional.relu(SNR_com_th_dB - SNRc_dB)

                reward_t = (
                    alpha_corr * rho_t +
                    beta_w * SNRt_dB +
                    (1.0 - beta_w) * SNRc_dB
                )


                # --------------------------------------------------
                # 7) NEXT-STATE 구성
                #    ⭐ 요청사항 반영: curr_SNRt, curr_SNRc 포함 ⭐
                # --------------------------------------------------
                R_next = R_sep_scaled[t - 4]

                next_state_vec = torch.cat([
                    R_next.reshape(-1),
                    torch.tensor([SNRt_dB.item(), SNRc_dB.item()],
                                 dtype=torch.float32, device=device)
                ], dim=0)


                # --------------------------------------------------
                # 8) Replay 저장
                # --------------------------------------------------
                agent.replay_buffer.add((
                    state_vec.detach().cpu(),
                    theta_pred[0].detach().cpu(),
                    float(reward_t.item()),
                    next_state_vec.detach().cpu(),
                    0.0
                ))


                # --------------------------------------------------
                # 9) 학습 업데이트
                # --------------------------------------------------
                agent.train(batch_size)

                total_reward += reward_t.item()
                step_count += 1

                # 다음 state에서 사용할 prev_SNR 업데이트
                prev_SNRt = float(SNRt_dB.item())
                prev_SNRc = float(SNRc_dB.item())

        avg_reward = total_reward / max(step_count, 1)
        avg_snrt = np.mean(SNRt_ep_list)
        avg_snrc = np.mean(SNRc_ep_list)

        episode_reward_log.append(avg_reward)
        episode_snrt_log.append(avg_snrt)
        episode_snrc_log.append(avg_snrc)
        train_losses.append(-avg_reward)

        print(f"Ep {e+1:03d} | Reward={avg_reward:.4f} | "
              f"SNRt={avg_snrt:.2f} dB | SNRc={avg_snrc:.2f} dB")


    # ============================================================
    # ⭐ Test evaluation ⭐
    # ============================================================
    print("\n=== Test evaluation 시작 ===")
    agent.actor.eval()

    SNRt_test_list = []
    SNRc_test_list = []
    reward_test_list = []

    prev_SNRt_eval = 0.0
    prev_SNRc_eval = 0.0

    for t in range(5, test_num):

        R_t5 = R_test_sep_scaled[t - 5]

        # state 구성
        state_vec = torch.cat([
            R_t5.reshape(-1),
            torch.tensor([prev_SNRt_eval, prev_SNRc_eval],
                         dtype=torch.float32, device=device)
        ], dim=0)

        state_batch = state_vec.unsqueeze(0)

        # 행동
        theta_eval, thetaH_eval = agent.select_action(state_batch, noise=False)

        # channel
        y_chan = origin_dataset_test[t].unsqueeze(0)
        yH_chan = origin_datasetH_test[t].unsqueeze(0)

        _, hc_batch, ht_batch, Ht_batch = cal_loss(
            y_chan, yH_chan, theta_eval, thetaH_eval, M, N
        )

        ht_i = ht_batch[0]
        hc_i = hc_batch[0]
        Ht_i = Ht_batch[0]

        # w 생성
        w_te = generate_w(ht_i, hc_i, P, Tau, sigma2)

        SNRt_lin = torch.linalg.norm(Ht_i @ w_te)**2 / sigma2
        SNRc_lin = torch.abs(w_te.conj().T @ hc_i)**2 / sigma2

        SNRt_test_list.append(SNRt_lin)
        SNRc_test_list.append(SNRc_lin)

        SNRt_dB = linear_to_db(SNRt_lin)
        SNRc_dB = linear_to_db(SNRc_lin)

        # reward
        norm_ht2 = torch.linalg.norm(ht_i)**2
        norm_hc2 = torch.linalg.norm(hc_i)**2
        corr_num = torch.abs(torch.conj(ht_i).T @ hc_i)**2
        rho_t = corr_num / (norm_ht2 * norm_hc2 + eps_corr)

        penalty = torch.nn.functional.relu(SNR_com_th_dB - SNRc_dB)

        reward_eval = alpha_corr * rho_t + beta_w * SNRt_dB + (1 - beta_w) * SNRc_dB
        reward_test_list.append(reward_eval.item())

        # next prev update
        prev_SNRt_eval = float(SNRt_dB.item())
        prev_SNRc_eval = float(SNRc_dB.item())

    SNRt_test = torch.stack(SNRt_test_list)
    SNRc_test = torch.stack(SNRc_test_list)
    reward_test_mean = np.mean(reward_test_list)

    print("\n=== Test 결과 ===")
    print(f"SNRt_test = {linear_to_db(SNRt_test.mean()):.2f} dB")
    print(f"SNRc_test = {linear_to_db(SNRc_test.mean()):.2f} dB")
    print(f"Reward_test = {reward_test_mean:.4f}")


    # 저장
    result_dir = "data"
    os.makedirs(result_dir, exist_ok=True)

    torch.save({
        'losses': np.array(train_losses),
        'reward_ep': np.array(episode_reward_log),
        'snrt_ep': np.array(episode_snrt_log),
        'snrc_ep': np.array(episode_snrc_log),
        'snrt_test': linear_to_db(SNRt_test.mean()).cpu().numpy(),
        'snrc_test': linear_to_db(SNRc_test.mean()).cpu().numpy(),
        'reward_test': reward_test_mean,
    }, os.path.join(result_dir, f"td3_train.pt"))

    print(f"\n== Test 결과가 '{result_dir}'에 저장되었습니다. ==")
