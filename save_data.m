%% =====================================================================
%  Create many RIS-ISAC channel datasets (100 sets)
% =====================================================================

N = 100;
out_dir = fullfile(pwd, "data");
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

v_min = 30;
v_max = 100;

for i = 1:N

    % ---- speed sampling ----
    vUE_kmh = v_min + (v_max - v_min) * rand();
    vTG_kmh = v_min + (v_max - v_min) * rand();

    suffix = sprintf("_%03d", i);

    generate_channel(out_dir, vUE_kmh, vTG_kmh, suffix);

    fprintf("[%d/%d] Created → Train_Channel%s.mat  (UE=%.1f km/h, TG=%.1f km/h)\n", ...
        i, N, suffix, vUE_kmh, vTG_kmh);
end

fprintf("✔ All %d datasets generated in %s\n", N, out_dir);
