mat_file = fullfile(pwd, 'data', 'Train_Channel_001.mat');

% ---- [설정] ----
PdB = 10;                 % 송신 전력 [dB]
sigma2_e = 1e-2;          % 잡음 분산
epsv = 1e-100;            % log/나눗셈 방지용

% ---- [데이터 로드] ----
S = load(mat_file);
G_all   = S.G_all;      % [N,M,T]
GH_all  = S.GH_all;     % [M,N,T]
hrc_all = S.hrc_all;    % [N,T]
hrt_all = S.hrt_all;    % [N,T]
[N,M,T] = size(G_all);
P = 10^(PdB/10);

% ---- [Tap-power sum 기반 파워 요약] ----
if isfield(S,'G_pow_all') && isfield(S,'hrc_pow_all') && isfield(S,'hrt_pow_all')
    G_pow_all   = S.G_pow_all;    % [N,M,T]
    hrc_pow_all = S.hrc_pow_all;  % [N,T]
    hrt_pow_all = S.hrt_pow_all;  % [N,T]
    % 벡터화 및 dB 변환
    G_pow_dB   = 10*log10(G_pow_all(:)   + epsv);
    hrc_pow_dB = 10*log10(hrc_pow_all(:) + epsv);
    hrt_pow_dB = 10*log10(hrt_pow_all(:) + epsv);

    fprintf('\n--- [Tap-power-sum (∑|h_k|^2) based] ---\n');
    pr_stats_db('|G|^2 (dB)',    G_pow_dB);
    pr_stats_db('|hrc|^2 (dB)',  hrc_pow_dB);
    pr_stats_db('|hrt|^2 (dB)',  hrt_pow_dB);
end

% ---- [Echo SNR(γ_e), 통신 SNR(SNR_c) 계산] ----
gamma_e_lin = zeros(T,1);   % sensing SNR (echo)
snrc_lin    = zeros(T,1);   % comm SNR

for t = 1:T
    hrc_t = hrc_all(:,t);         % [N x 1]
    hrt_t = hrt_all(:,t);         % [N x 1]
    GH_t  = GH_all(:,:,t);        % [M x N]

    % RIS 위상 (단순 conjugate 보정)
    theta_t = exp(1j * angle(conj(hrt_t)));      % [N x 1]
    hc_t = GH_t * (theta_t .* hrc_t);            % [M x 1]
    ht_t = GH_t * (theta_t .* hrt_t);            % [M x 1]

    % MRT 빔포머 (통신 기준)
    if norm(hc_t) < 1e-12
        w_t = zeros(M,1);
    else
        w_t = sqrt(P) * ht_t / norm(ht_t);
    end

    % Sensing: γ_e = || H_t w ||^2 / σ_e^2,  H_t = h_t h_t^H
    Ht_t = ht_t * (ht_t');
    gamma_e_lin(t) = norm(Ht_t * w_t)^2 / sigma2_e;

    % Communication: |w^H hc|^2 / σ_e^2
    snrc_lin(t) = abs(w_t' * hc_t)^2 / sigma2_e;
end

gamma_e_dB = 10*log10(gamma_e_lin + epsv);
snrc_dB    = 10*log10(snrc_lin    + epsv);

fprintf('\n=== [Echo SNR γ_e summary (dB)] ===\n');
pr_stats_db('gamma_e (dB)', gamma_e_dB);

fprintf('\n=== [Communication SNR summary (dB)] ===\n');
pr_stats_db('SNR_c (dB)',   snrc_dB);

% ---- [시간축 power fluctuation 시각화] ----
% 평균 파워 (tap-power-sum 기준, 스냅샷별 평균)
G_pow_t   = squeeze(mean(mean(abs(G_all).^2, 1),2));     % [T,1]
hrc_pow_t = squeeze(mean(abs(hrc_all).^2, 1));           % [T,1]
hrt_pow_t = squeeze(mean(abs(hrt_all).^2, 1));           % [T,1]

figure('Color','w','Name','Channel Power over Time');
plot(1:T, 10*log10(G_pow_t + epsv), 'LineWidth',1.4); hold on;
plot(1:T, 10*log10(hrc_pow_t + epsv), 'LineWidth',1.4);
plot(1:T, 10*log10(hrt_pow_t + epsv), 'LineWidth',1.4);
legend('|G|^2','|h_{rc}|^2','|h_{rt}|^2','Location','best');
xlabel('Snapshot t'); ylabel('Power (dB)');
title('Channel tap-power-sum fluctuation over time');
grid on;

% ---- [SNR(t) 시각화] ----
figure('Color','w','Name','gamma_e over time (dB)');
plot(1:T, gamma_e_dB, 'LineWidth',1.2); hold on;
plot(1:T, snrc_dB, '--', 'LineWidth',1.2);
grid on;
xlabel('Snapshot t'); ylabel('SNR (dB)');
title(sprintf('\\gamma_e (solid), SNR_c (dashed)  [P=%.1f dB, \\sigma_e^2=%.3g]', PdB, sigma2_e));
legend('\gamma_e (Echo SNR)','SNR_c (Comm SNR)');

% ---- [dt 정보] ----
if isfield(S,'dt_UE') || isfield(S,'dt_TG')
    dt_ue = NaN; dt_tg = NaN;
    if isfield(S,'dt_UE'), dt_ue = S.dt_UE; end
    if isfield(S,'dt_TG'), dt_tg = S.dt_TG; end
    fprintf('\n[info] dt_UE = %s s, dt_TG = %s s', num2str(dt_ue,'%.4g'), num2str(dt_tg,'%.4g'));
    if ~isnan(dt_ue), fprintf('  (fs_UE=%.3f Hz)', 1/dt_ue); end
    if ~isnan(dt_tg), fprintf('  (fs_TG=%.3f Hz)', 1/dt_tg); end
    fprintf('\n');
end

% ---- [Helper: dB 통계 출력] ----
function pr_stats_db(name, x_db)
    x_db = x_db(~isnan(x_db) & isfinite(x_db));
    if isempty(x_db), fprintf('%s: (no data)\n', name); return; end
    q = quantile(x_db,[0.1 0.5 0.9]);
    fprintf('%-28s  mean=%8.3f dB  std=%8.3f dB  p10=%8.3f dB  median=%8.3f dB  p90=%8.3f dB\n', ...
        name, mean(x_db), std(x_db), q(1), q(2), q(3));
end
