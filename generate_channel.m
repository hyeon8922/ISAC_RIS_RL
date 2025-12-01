function generate_jakes_ris_data(out_dir, vUE_kmh, vTG_kmh, file_suffix)
% =======================================================================
%  Minimal simplified RIS-ISAC Jakes channel generator (Fixed Version)
%  - Saves: G_all, GH_all, hrc_all, hrcH_all, hrt_all, hrtH_all
%  - Compatible with load_and_process_data() in Python code
% =======================================================================

%% Output folder
if nargin < 1 || isempty(out_dir)
    out_dir = fullfile(pwd, 'data/');
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

if nargin < 4
    file_suffix = "";
end

rng('shuffle');

%% System parameters
fc     = 3.5e9;
c      = physconst('LightSpeed');
lambda = c / fc;

Mx = 8;
Nx = 32;
M = Mx;
N = Nx;
d = lambda/2;

T  = 30;      % number of snapshots
Ts = 0.1;

% Speed
vUE = vUE_kmh / 3.6;
vTG = vTG_kmh / 3.6;

% Node positions
BS_pos  = [0; 0];
RIS_pos = [50; 0];

UE_0 = [0; -5];
TG_0 = [0; -10];

UE_dir = [1; 0];
TG_dir = [1; 0];

% Rician K factors
K_BR = 10^(10/10);
K_RU = 10^(3/10);
K_RT = 10^(3/10);

n_pw_sp_dB = -55;

%% Antenna geometry
BS_elem  = ula_pos_2d(M, d, BS_pos);
RIS_elem = ula_pos_2d(N, d, RIS_pos);

%% UE/TG trajectories
tvec = (0:T-1);
UE_pos = UE_0 + UE_dir * (vUE * Ts) .* tvec;
TG_pos = TG_0 + TG_dir * (vTG * Ts) .* tvec;

%% Allocate channels
G_all   = zeros(N, M, T);
GH_all  = zeros(M, N, T);    % <--- 추가됨

hrc_all  = zeros(N, T);
hrcH_all = zeros(N, T);      % <--- 추가됨

hrt_all  = zeros(N, T);
hrtH_all = zeros(N, T);      % <--- 추가됨

%% Precompute BS → RIS LoS
az_BR_tx = azimuth_from_to(BS_pos, RIS_pos);
az_BR_rx = azimuth_from_to(RIS_pos, BS_pos);

a_BS  = steering_ula(M, d, lambda, az_BR_tx);
a_RIS = steering_ula(N, d, lambda, az_BR_rx);

G_los = a_RIS * a_BS.';
G_los = G_los / sqrt(norm(G_los,'fro')^2/(N*M));

D_BR  = mean(vecnorm(RIS_elem - BS_elem(:,ceil(M/2)), 2, 1));
PL_BR = PL_LOS(D_BR, fc);
alpha_BR = 10^(-(PL_BR + n_pw_sp_dB)/20);

%% Jakes fading init
st_BR = jakes_init(N*M, 32);
st_RU = jakes_init(N, 32);
st_RT = jakes_init(N, 32);

%% Main loop
for t = 1:T
    n = t-1;

    %% BS → RIS
    g_BR = reshape(jakes_step(st_BR, Ts, 0, n), [N M]);
    G = sqrt(K_BR/(K_BR+1))*G_los + sqrt(1/(K_BR+1))*g_BR;
    G = alpha_BR * G;

    G_all(:,:,t)  = G;
    GH_all(:,:,t) = G';           % <--- 저장

    %% RIS → UE
    UE_t = UE_pos(:,t);
    az = azimuth_from_to(RIS_pos, UE_t);

    a = steering_ula(N, d, lambda, az);
    h_los = a / sqrt(norm(a)^2/N);

    D = mean(vecnorm(RIS_elem - UE_t, 2, 1));
    PL = PL_LOS(D, fc) + 13;
    alpha = 10^(-(PL + n_pw_sp_dB)/20);

    g = jakes_step(st_RU, Ts, vUE/lambda, n);
    h = sqrt(K_RU/(K_RU+1))*h_los + sqrt(1/(K_RU+1))*g;
    h = alpha * h;

    hrc_all(:,t)  = h;
    hrcH_all(:,t) = conj(h);      % <--- 저장

    %% RIS → TG
    TG_t = TG_pos(:,t);
    az = azimuth_from_to(RIS_pos, TG_t);

    a = steering_ula(N, d, lambda, az);
    h_los = a / sqrt(norm(a)^2/N);

    D = mean(vecnorm(RIS_elem - TG_t, 2, 1));
    PL = PL_LOS(D, fc) + 13;
    alpha = 10^(-(PL + n_pw_sp_dB)/20);

    g = jakes_step(st_RT, Ts, vTG/lambda, n);
    h = sqrt(K_RT/(K_RT+1))*h_los + sqrt(1/(K_RT+1))*g;
    h = alpha * h;

    hrt_all(:,t)  = h;
    hrtH_all(:,t) = conj(h);      % <--- 저장
end

%% Save files
train_name = sprintf("Train_Channel%s.mat", file_suffix);
test_name  = sprintf("Test_Channel%s.mat",  file_suffix);

save(fullfile(out_dir, train_name), ...
    'G_all','GH_all', ...
    'hrc_all','hrcH_all', ...
    'hrt_all','hrtH_all', ...
    'BS_pos','RIS_pos','UE_pos','TG_pos','fc','Ts','vUE_kmh','vTG_kmh');

save(fullfile(out_dir, test_name), ...
    'G_all','GH_all', ...
    'hrc_all','hrcH_all', ...
    'hrt_all','hrtH_all', ...
    'BS_pos','RIS_pos','UE_pos','TG_pos','fc','Ts','vUE_kmh','vTG_kmh');

fprintf("✔ Saved: Train/Test Channel %s\n", file_suffix);

end

%% ========== Helper functions ==========
function elems = ula_pos_2d(M, d, center)
x = ((0:M-1) - (M-1)/2) * d;
elems = [center(1) + x; center(2)*ones(1,M)];
end

function a = steering_ula(M, d, lambda, az)
m = (0:M-1).';
phase = -2*pi*(d/lambda)*m*cos(az);
a = exp(1j * phase);
end

function az = azimuth_from_to(p1, p2)
v = p2 - p1;
az = atan2(v(2), v(1));
end

function pl = PL_LOS(d, fc)
pl = 22*log10(max(d,1)) + 28 + 20*log10(fc/1e9);
end

function st = jakes_init(Ns, K)
st.K = K;
st.phi = 2*pi*rand(Ns, K);
st.alpha = 2*pi*rand(1, K);
end

function g = jakes_step(st, Ts, fd, n)
wt  = 2*pi*fd*Ts*n;
arg = wt*cos(st.alpha) + st.phi;
re = sum(cos(arg), 2);
im = sum(sin(arg), 2);
g  = (re + 1j*im) / sqrt(st.K/2);
g  = g / sqrt(mean(abs(g).^2) + eps);
end
