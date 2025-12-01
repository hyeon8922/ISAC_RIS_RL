# -*- coding: utf-8 -*-
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio

# OpenMP 중복 로드 임시 완화(근본 해결은 PATH 정리 권장)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = 'cpu'
torch.set_default_dtype(torch.float32)

# ───────────────────────────────
# 하이퍼파라미터(고정 값)
# ───────────────────────────────
K, M, N = 1, 8, 32
lr = 0.001
sigma2 = 1e-2
batch_size = 200      # (지금은 사용 X, 필요하면 나중에 time-batch로 확장 가능)
epochs = 100
ppp = 200             # (지금은 사용 X)
TaudB = 10.0
Tau = 10 ** (TaudB / 10)
PdB = 10
P = 10 ** (PdB / 10)

ChannelFile = r"C:\Users\CNL-A2\Desktop\DL-Beamforming-RIS-ISAC-main\manhattan\quadriga_data\Train_Time_Channel_N32_Jakes.mat"
ChannelFile_test = r"C:\Users\CNL-A2\Desktop\DL-Beamforming-RIS-ISAC-main\manhattan\quadriga_data\Test_Time_Channel_N32_jakes.mat"

# 저장 폴더
log_root = "./results_compare/ddpg/RL"
os.makedirs(log_root, exist_ok=True)

# ───────────────────────────────
# [SAVE] 마지막 epoch 시계열 저장 유틸
# ───────────────────────────────
def save_time_series(csv_path, snrt_db_vec, snrc_db_vec):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "snrt_db", "snrc_db"])
        T = len(snrt_db_vec)
        for t_idx in range(T):
            w.writerow([t_idx + 1, float(snrt_db_vec[t_idx]), float(snrc_db_vec[t_idx])])

# ───────────────────────────────
# 채널 데이터 로드 (Train)
# ───────────────────────────────
chan_mat = scio.loadmat(ChannelFile)
G0, hrc0, hrt0 = chan_mat['G_all'], chan_mat['hrc_all'], chan_mat['hrt_all']
GH0, hrcH0, hrtH0 = chan_mat['GH_all'], chan_mat['hrcH_all'], chan_mat['hrtH_all']

total_num = G0.shape[2]
G    = torch.zeros([total_num, N, M], dtype=torch.complex128)
hrc  = torch.zeros([total_num, N, 1], dtype=torch.complex128)
hrt  = torch.zeros([total_num, N, 1], dtype=torch.complex128)
GH   = torch.zeros([total_num, M, N], dtype=torch.complex128)
hrcH = torch.zeros([total_num, 1, N], dtype=torch.complex128)
hrtH = torch.zeros([total_num, 1, N], dtype=torch.complex128)

for i in range(total_num):
    G[i]    = torch.from_numpy(G0[:, :, i])
    hrc[i]  = torch.from_numpy(hrc0[:, i].reshape(N, 1))
    hrt[i]  = torch.from_numpy(hrt0[:, i].reshape(N, 1))
    GH[i]   = torch.from_numpy(GH0[:, :, i])
    hrcH[i] = torch.from_numpy(hrcH0[:, i].reshape(1, N))
    hrtH[i] = torch.from_numpy(hrtH0[:, i].reshape(1, N))

# ───────────────────────────────
# 채널 데이터 로드 (Test)
# ───────────────────────────────
chan_mat_test = scio.loadmat(ChannelFile_test)
G0_test, hrc0_test, hrt0_test = chan_mat_test['G_all'], chan_mat_test['hrc_all'], chan_mat_test['hrt_all']
GH0_test, hrcH0_test, hrtH0_test = chan_mat_test['GH_all'], chan_mat_test['hrcH_all'], chan_mat_test['hrtH_all']

test_num = G0_test.shape[2]
G_test    = torch.zeros([test_num, N, M], dtype=torch.complex128)
hrc_test  = torch.zeros([test_num, N, 1], dtype=torch.complex128)
hrt_test  = torch.zeros([test_num, N, 1], dtype=torch.complex128)
GH_test   = torch.zeros([test_num, M, N], dtype=torch.complex128)
hrcH_test = torch.zeros([test_num, 1, N], dtype=torch.complex128)
hrtH_test = torch.zeros([test_num, 1, N], dtype=torch.complex128)

for i in range(test_num):
    G_test[i]    = torch.from_numpy(G0_test[:, :, i])
    hrc_test[i]  = torch.from_numpy(hrc0_test[:, i].reshape(N, 1))
    hrt_test[i]  = torch.from_numpy(hrt0_test[:, i].reshape(N, 1))
    GH_test[i]   = G_test[i].conj().transpose(0, 1)
    hrcH_test[i] = hrc_test[i].conj().reshape(1, N)
    hrtH_test[i] = hrt_test[i].conj().reshape(1, N)

# ───────────────────────────────
# 공통 행렬 전처리(Train/Test)
# ───────────────────────────────
psi_com    = torch.matmul(torch.diag_embed(hrcH.squeeze(), dim1=1), G)
psiH_com   = torch.matmul(GH, torch.diag_embed(hrc.squeeze(), dim1=1))
R_com      = torch.matmul(psi_com, psiH_com).reshape(total_num, 1, N, N)
R_com_sep  = torch.cat([torch.real(R_com), torch.imag(R_com)], dim=1)

psi_rad    = torch.matmul(torch.diag_embed(hrtH.squeeze(), dim1=1), G)
psiH_rad   = torch.matmul(GH, torch.diag_embed(hrt.squeeze(), dim1=1))
R_rad      = torch.matmul(psi_rad, psiH_rad).reshape(total_num, 1, N, N)
R_rad_sep  = torch.cat([torch.real(R_rad), torch.imag(R_rad)], dim=1)

R_sep      = torch.cat([R_rad_sep, R_com_sep], dim=1)  # [T, 4, N, N]
R_sep_mean = torch.mean(R_sep, dim=2, keepdim=True)
R_sep_std  = torch.std(R_sep, dim=2, keepdim=True)
R_sep_scaled = (R_sep - R_sep_mean) / (R_sep_std + 1e-10)

hrc_vec = hrc.view(total_num, -1)
hrt_vec = hrt.view(total_num, -1)
G_vec   = G.view(total_num, -1)
origin_dataset = torch.cat((torch.real(G_vec), torch.imag(G_vec),
                            torch.real(hrt_vec), torch.imag(hrt_vec),
                            torch.real(hrc_vec), torch.imag(hrc_vec)), dim=-1)

hrcH_vec = hrcH.view(total_num, -1)
hrtH_vec = hrtH.view(total_num, -1)
GH_vec   = GH.view(total_num, -1)
origin_datasetH = torch.cat((torch.real(GH_vec), torch.imag(GH_vec),
                             torch.real(hrtH_vec), torch.imag(hrtH_vec),
                             torch.real(hrcH_vec), torch.imag(hrcH_vec)), dim=-1)

psi_com_test     = torch.matmul(torch.diag_embed(hrcH_test.squeeze(), dim1=1), G_test)
psiH_com_test    = torch.matmul(GH_test, torch.diag_embed(hrc_test.squeeze(), dim1=1))
R_com_test       = torch.matmul(psi_com_test, psiH_com_test).reshape(test_num, 1, N, N)
R_com_test_sep   = torch.cat([torch.real(R_com_test), torch.imag(R_com_test)], dim=1)

psi_rad_test     = torch.matmul(torch.diag_embed(hrtH_test.squeeze(), dim1=1), G_test)
psiH_rad_test    = torch.matmul(GH_test, torch.diag_embed(hrt_test.squeeze(), dim1=1))
R_rad_test       = torch.matmul(psi_rad_test, psiH_rad_test).reshape(test_num, 1, N, N)
R_rad_test_sep   = torch.cat([torch.real(R_rad_test), torch.imag(R_rad_test)], dim=1)

R_test_sep       = torch.cat([R_rad_test_sep, R_com_test_sep], dim=1)
R_test_sep_mean  = torch.mean(R_test_sep, dim=2, keepdim=True)
R_test_sep_std   = torch.std(R_test_sep, dim=2, keepdim=True)
R_test_sep_scaled = (R_test_sep - R_test_sep_mean) / (R_test_sep_std + 1e-10)

hrc_test_vec = hrc_test.reshape(test_num, -1)
hrt_test_vec = hrt_test.reshape(test_num, -1)
G_test_vec   = G_test.reshape(test_num, -1)
origin_dataset_test = torch.cat((torch.real(G_test_vec), torch.imag(G_test_vec),
                                 torch.real(hrt_test_vec), torch.imag(hrt_test_vec),
                                 torch.real(hrc_test_vec), torch.imag(hrc_test_vec)), dim=-1)

hrcH_test_vec = hrcH_test.reshape(test_num, -1)
hrtH_test_vec = hrtH_test.reshape(test_num, -1)
GH_test_vec   = GH_test.reshape(test_num, -1)
origin_datasetH_test = torch.cat((torch.real(GH_test_vec), torch.imag(GH_test_vec),
                                  torch.real(hrtH_test_vec), torch.imag(hrcH_test_vec)), dim=-1)

# ───────────────────────────────
# 보조 함수
# ───────────────────────────────
def linear_to_db(value):
    return 10 * torch.log10(value + 1e-10)

class mynet(nn.Module):
    """
    입력 state:
      - 채널: R_sep_scaled[t-5] ∈ ℝ^{4×N×N}
      - 스칼라: SNR_prev_dB ∈ ℝ (이전 sensing SNR in dB)
    구조:
      CNN으로 채널 feature 추출 → FC에서 SNR_prev_dB를 concat → θ 출력
    """
    def __init__(self, M, N):
        super().__init__()
        def conv_bn(cin, cout, s):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, s, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
            )
        def conv_dw(cin, cout, s):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, s, 1, groups=cin, bias=False),
                nn.BatchNorm2d(cin),
                nn.ReLU(inplace=True),
                nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
            )
        self.model = nn.Sequential(
            conv_bn(4, 32, 1),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 256, 2),
            conv_dw(256, 512, 2),
            conv_dw(512, 1024, 2),
            nn.AvgPool2d(2)
        )
        # CNN feature (1024) + SNR_prev_dB (1) → N
        self.fc = nn.Linear(1024 + 1, N)

    def forward(self, x, snr_prev_db):
        """
        x: [B, 4, N, N]  (R_sep_scaled[t-5])
        snr_prev_db: [B] or [B,1] (이전 sensing SNR in dB)
        """
        B = x.size(0)
        x = self.model(x).view(B, -1)   # [B, 1024]

        if snr_prev_db.dim() == 1:
            snr_prev_db = snr_prev_db.unsqueeze(1)  # [B,1]
        # concat: [B, 1025]
        x = torch.cat([x, snr_prev_db], dim=1)

        x = self.fc(x).to(torch.complex128)
        theta_real, theta_imag = torch.cos(x), torch.sin(x)
        return theta_real + 1j * theta_imag, theta_real - 1j * theta_imag

def generate_w(ht, hc, Pt, Tau, sigma):
    temp1 = torch.abs(hc.conj().T @ ht)**2
    u1 = hc / torch.linalg.norm(hc)
    u2_ = ht - (u1.conj().T @ ht) * u1
    u2 = u2_ / torch.linalg.norm(u2_)
    x1_ = u1.conj().T @ ht
    x1 = torch.sqrt(Tau*sigma / torch.linalg.norm(hc)**2) * x1_/torch.linalg.norm(x1_)
    x2_ = u2.conj().T @ ht
    temp2 = Pt - Tau*sigma / torch.linalg.norm(hc)**2
    temp2sqrt = (1j if temp2 < 0 else 1) * torch.sqrt(abs(temp2))
    x2 = temp2sqrt * x2_/torch.linalg.norm(x2_)
    if np.sqrt(Pt)*temp1 > Tau*sigma*torch.linalg.norm(ht)**2:
        w = np.sqrt(Pt)*ht/torch.linalg.norm(ht)
    else:
        w = x1*u1 + x2*u2
        if torch.linalg.norm(w)**2 > Pt:
            w = np.sqrt(Pt)*w/torch.linalg.norm(w)
    return w

def cal_loss(chan, chanH, theta, thetaH, M, N):
    G = chan[:, :2*N*M].reshape(-1, 2*N*M)
    G = G[:, :N*M] + 1j*G[:, N*M:]
    G = G.view(-1, N, M).to(torch.complex128)
    GH = chanH[:, :2*N*M].reshape(-1, 2*N*M)
    GH = GH[:, :N*M] + 1j*GH[:, N*M:]
    GH = GH.view(-1, M, N).to(torch.complex128)

    hrt = chan[:, 2*N*M:2*(N*M+N)].reshape(-1, 2*N)
    hrt = hrt[:, :N] + 1j*hrt[:, N:]
    hrt = hrt.view(-1, N, 1).to(torch.complex128)
    hrtH = chanH[:, 2*N*M:2*(N*M+N)].reshape(-1, 2*N)
    hrtH = hrtH[:, :N] + 1j*hrtH[:, N:]
    hrtH = hrtH.view(-1, 1, N).to(torch.complex128)

    hrc = chan[:, 2*(N*M+N):].reshape(-1, 2*N)
    hrc = hrc[:, :N] + 1j*hrc[:, N:]
    hrc = hrc.view(-1, N, 1).to(torch.complex128)
    hrcH = chanH[:, 2*(N*M+N):].reshape(-1, 2*N)
    hrcH = hrcH[:, :N] + 1j*hrcH[:, N:]
    hrcH = hrcH.view(-1, 1, N).to(torch.complex128)

    temp1 = GH @ torch.diag_embed(theta, dim1=1)
    temp2 = torch.diag_embed(thetaH, dim1=1) @ G
    hc = temp1 @ hrc
    ht = temp1 @ hrt
    htH = hrtH @ temp2
    Ht = ht @ htH

    rv = (Ht @ hc)[:, :, 0]
    reward = torch.linalg.norm(rv, dim=-1).to(torch.float32)
    a = 0.8 * torch.linalg.norm(Ht, dim=[1, 2])
    return -torch.mean(reward + a), hc, ht, Ht

# ───────────────────────────────
# 모델 및 옵티마이저
# ───────────────────────────────
model = mynet(M, N).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 마지막 epoch의 시계열을 담아 둘 변수
last_snrt_train_db_vec = None
last_snrc_train_db_vec = None
last_snrt_test_db_vec  = None
last_snrc_test_db_vec  = None

# ───────────────────────────────
# 학습 루프 (time-series + state = R_{t-5} + SNR_prev_dB)
# ───────────────────────────────
for e in range(epochs):
    model.train()
    running_loss = 0.0
    step_count = 0

    # 초기 이전 sensing SNR(dB)
    SNR_prev_dB = 0.0

    # t = 5 부터: state에 t-5 채널 사용
    for t in range(5, total_num):
        # state: (R_sep_scaled[t-5], SNR_prev_dB)
        R_t5 = R_sep_scaled[t - 5].unsqueeze(0).to(device).float()  # [1,4,N,N]
        snr_prev_tensor = torch.tensor([SNR_prev_dB],
                                       dtype=torch.float32,
                                       device=device).unsqueeze(0)  # [1,1]

        # forward
        theta_pred, thetaH_pred = model(R_t5, snr_prev_tensor)

        # label: 현재 시점 t의 채널
        y_chan  = origin_dataset[t].unsqueeze(0).to(device)
        yH_chan = origin_datasetH[t].unsqueeze(0).to(device)

        loss, hc_batch, ht_batch, Ht_batch = cal_loss(
            y_chan, yH_chan, theta_pred, thetaH_pred, M, N
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step_count += 1

        # SNR_prev_dB 업데이트 (RL과 동일한 방식)
        with torch.no_grad():
            ht_i = ht_batch[0]
            hc_i = hc_batch[0]
            Ht_i = Ht_batch[0]

            w = generate_w(ht_i, hc_i, P, Tau, sigma2)
            SNRt_lin = torch.linalg.norm(Ht_i @ w) ** 2 / sigma2
            SNRt_dB = linear_to_db(SNRt_lin)
            SNR_prev_dB = float(SNRt_dB.item())

    avg_train_loss = running_loss / max(step_count, 1)

    # ── Train SNR (시퀀스, t>=5 구간) ─────────────────────
    model.eval()
    SNRt_train_list = []
    SNRc_train_list = []

    SNR_prev_dB_eval = 0.0
    for t in range(5, total_num):
        R_t5 = R_sep_scaled[t - 5].unsqueeze(0).to(device).float()
        snr_prev_tensor = torch.tensor([SNR_prev_dB_eval],
                                       dtype=torch.float32,
                                       device=device).unsqueeze(0)

        theta_eval, thetaH_eval = model(R_t5, snr_prev_tensor)

        y_chan_train = origin_dataset[t].unsqueeze(0).to(device)
        yH_chan_train = origin_datasetH[t].unsqueeze(0).to(device)

        _, hc_tr2, ht_tr2, Ht_tr2 = cal_loss(
            y_chan_train, yH_chan_train, theta_eval, thetaH_eval, M, N
        )
        ht_i = ht_tr2[0]
        hc_i = hc_tr2[0]
        Ht_i = Ht_tr2[0]

        w_tr = generate_w(ht_i, hc_i, P, Tau, sigma2)
        SNRc_lin = torch.abs(w_tr.T.conj() @ hc_i) ** 2 / sigma2
        SNRt_lin = torch.linalg.norm(Ht_i @ w_tr) ** 2 / sigma2

        SNRc_train_list.append(SNRc_lin)
        SNRt_train_list.append(SNRt_lin)

        SNRt_dB = linear_to_db(SNRt_lin)
        SNR_prev_dB_eval = float(SNRt_dB.item())

    if len(SNRt_train_list) > 0:
        SNRt_train = torch.stack(SNRt_train_list)  # [T-5,1]
        SNRc_train = torch.stack(SNRc_train_list)
    else:
        SNRt_train = torch.tensor([1.0], device=device)
        SNRc_train = torch.tensor([1.0], device=device)

    # ── Test SNR (시퀀스, t>=5 구간) ──────────────────────
    SNRt_test_list = []
    SNRc_test_list = []

    SNR_prev_dB_eval = 0.0
    t0 = time.time()
    for t in range(5, test_num):
        R_t5 = R_test_sep_scaled[t - 5].unsqueeze(0).to(device).float()
        snr_prev_tensor = torch.tensor([SNR_prev_dB_eval],
                                       dtype=torch.float32,
                                       device=device).unsqueeze(0)

        theta_eval, thetaH_eval = model(R_t5, snr_prev_tensor)

        y_chan_test = origin_dataset_test[t].unsqueeze(0).to(device)
        yH_chan_test = origin_datasetH_test[t].unsqueeze(0).to(device)

        _, hc_te2, ht_te2, Ht_te2 = cal_loss(
            y_chan_test, yH_chan_test, theta_eval, thetaH_eval, M, N
        )
        ht_i = ht_te2[0]
        hc_i = hc_te2[0]
        Ht_i = Ht_te2[0]

        w_te = generate_w(ht_i, hc_i, P, Tau, sigma2)
        SNRc_lin = torch.abs(w_te.T.conj() @ hc_i) ** 2 / sigma2
        SNRt_lin = torch.linalg.norm(Ht_i @ w_te) ** 2 / sigma2

        SNRc_test_list.append(SNRc_lin)
        SNRt_test_list.append(SNRt_lin)

        SNRt_dB = linear_to_db(SNRt_lin)
        SNR_prev_dB_eval = float(SNRt_dB.item())
    _ = time.time() - t0

    if len(SNRt_test_list) > 0:
        SNRt_test = torch.stack(SNRt_test_list)
        SNRc_test = torch.stack(SNRc_test_list)
    else:
        SNRt_test = torch.tensor([1.0], device=device)
        SNRc_test = torch.tensor([1.0], device=device)

    # ── 이번 epoch의 dB 벡터를 '마지막 epoch용' 변수에 갱신
    last_snrt_train_db_vec = linear_to_db(SNRt_train.squeeze()).detach().cpu().numpy()
    last_snrc_train_db_vec = linear_to_db(SNRc_train.squeeze()).detach().cpu().numpy()
    last_snrt_test_db_vec  = linear_to_db(SNRt_test.squeeze()).detach().cpu().numpy()
    last_snrc_test_db_vec  = linear_to_db(SNRc_test.squeeze()).detach().cpu().numpy()

    print(f"[Epoch {e+1}/{epochs}] PdB {PdB} | "
          f"TrainLoss={avg_train_loss:.4f} | "
          f"Train SNRt mean={last_snrt_train_db_vec.mean():.2f} dB | "
          f"Train SNRc mean={last_snrc_train_db_vec.mean():.2f} dB | "
          f"Test SNRt mean={last_snrt_test_db_vec.mean():.2f} dB | "
          f"Test SNRc mean={last_snrc_test_db_vec.mean():.2f} dB")

# ── 마지막 epoch 결과만 저장 (시간 순, 길이 ≈ total_num-5, test_num-5)
train_csv_final = os.path.join(log_root, f"IBF-net_P{PdB}.csv")
test_csv_final  = os.path.join(log_root, f"IBF-net_P{PdB}.csv")

save_time_series(train_csv_final, last_snrt_train_db_vec, last_snrc_train_db_vec)
save_time_series(test_csv_final,  last_snrt_test_db_vec,  last_snrc_test_db_vec)

print(f"== 저장 완료 ==\n  Train: {train_csv_final} (rows={len(last_snrt_train_db_vec)})"
      f"\n  Test : {test_csv_final}  (rows={len(last_snrt_test_db_vec)})")
