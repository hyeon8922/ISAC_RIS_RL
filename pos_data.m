mat_file = fullfile(pwd,'data','Train_Channel_001.mat');
plot_positions_only(mat_file);

function plot_positions_only(mat_file, out_dir)
% ---------------------------------------------------------------
% RIS-ISAC 시뮬레이션 결과에서 위치(트랙)만 시각화/저장
% 시작점: 동그라미(o), 끝점: 화살표(>)
% ---------------------------------------------------------------

assert(isfile(mat_file),'파일을 찾을 수 없습니다: %s',mat_file);
S = load(mat_file);

% ---- 필드 읽기 ----
BS_pos  = pick_field(S, {'BS_pos'}, []);
RIS_pos = pick_field(S, {'RIS_pos'}, []);
UE_pos  = pick_field(S, {'UE_pos','UE_track'}, []);
TG_pos  = pick_field(S, {'TG_pos','TG_track'}, []);

% ---- 출력 폴더 ----
if nargin<2 || isempty(out_dir)
    [folder, base, ~] = fileparts(string(mat_file));
    out_dir = fullfile(folder, "figs_" + base);
end
if ~isfolder(out_dir), mkdir(out_dir); end

% ---- 2D 위치 플롯 ----
f = figure('Name','Positions 2D','Color','w','Position',[300 300 800 600]);
hold on; grid on; box on;

h_legend = []; 
legend_names = {};

% ==== BS 위치 ====
if ~isempty(BS_pos)
    h_bs = scatter(BS_pos(1), BS_pos(2), 80, 'ks', 'filled'); 
    text(BS_pos(1)+1, BS_pos(2), 'BS');
    h_legend(end+1) = h_bs;
    legend_names{end+1} = 'BS';
end

% ==== RIS 위치 ====
if ~isempty(RIS_pos)
    h_ris = scatter(RIS_pos(1), RIS_pos(2), 80, 'gd', 'filled'); 
    text(RIS_pos(1)+1, RIS_pos(2), 'RIS');
    h_legend(end+1) = h_ris;
    legend_names{end+1} = 'RIS';
end

% ==== UE 트랙 ====
if ~isempty(UE_pos)
    h_ue = plot(UE_pos(1,:), UE_pos(2,:), 'b-', 'LineWidth',1.5);
    scatter(UE_pos(1,1), UE_pos(2,1), 60, 'bo','filled');      % 시작점
    scatter(UE_pos(1,end), UE_pos(2,end), 80, 'b>','filled');  % 끝점
    h_legend(end+1) = h_ue;
    legend_names{end+1} = 'UE track';
end

% ==== TG 트랙 ====
if ~isempty(TG_pos)
    h_tg = plot(TG_pos(1,:), TG_pos(2,:), 'r-', 'LineWidth',1.5);
    scatter(TG_pos(1,1), TG_pos(2,1), 60, 'ro','filled');      % 시작점
    scatter(TG_pos(1,end), TG_pos(2,end), 80, 'r>','filled');  % 끝점
    h_legend(end+1) = h_tg;
    legend_names{end+1} = 'TG track';
end

xlabel('x [m]'); ylabel('y [m]');
axis equal;
legend(h_legend, legend_names, 'Location','northeast', 'FontSize', 14);
title('Node positions and trajectories (2D)');

% ---- 저장 ----
saveas(f, fullfile(out_dir,'positions2d.png'));
fprintf('[완료] 위치 그래프 저장 → %s\n', fullfile(out_dir,'positions2d.png'));
end

%% ---- Helper ----
function v = pick_field(S, keys, default)
for i=1:numel(keys)
    if isfield(S, keys{i}), v = S.(keys{i}); return; end
end
if nargin==3, v = default; else, v = []; end
end
