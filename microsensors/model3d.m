
clear; clc; close all;

% 参数设置 (依据论文实验值)
grid_size = 100;       % 网格尺寸 (100×100×100单元)
voxel_size = 0.1e-6;   % 单元尺寸=0.1μm
porosity = 0.75;       % 孔隙率75%
vox_ratio = 0.15;      % VOX颗粒在LIG骨架中占比
R0 = 1000;             % 初始电阻
alpha = -0.003;        % 温度系数
beta = -0.02929;       % 气体响应系数

%% 1. 构建三维异质结构
[lig_matrix, vox_matrix] = generate_heterojunction(grid_size, porosity, vox_ratio);

% 可视化截面结构
figure('Name', 'VOX/LIG异质结构三维截面');
slice(double(vox_matrix), [], [grid_size/2], []);
shading interp;
colormap([0.8 0.8 0.8; 0.2 0.6 0.9]); % LIG灰 VOX蓝
title('VOX/LIG三维结构截面图');
xlabel('X/μm'); ylabel('Y/μm'); zlabel('Z/μm');
set(gca, 'DataAspectRatio', [1 1 1]);

%% 2. 气体扩散仿真（二维平面）
D_no2 = 2e-9;      % NO2扩散系数
c_surface = 1.0;   % 表面浓度
c_distribution = simulate_gas_diffusion_2d(D_no2, c_surface, grid_size, voxel_size);

%% 3. 温度响应仿真
T = 10:10:110;
R_temp = R0 * (1 + alpha * (T - 25));

figure('Name','温度响应特性');
plot(T, R_temp, 'r-', 'LineWidth', 2);
xlabel('温度 (°C)'); ylabel('电阻 (Ω)');
title('VOX/LIG 温度响应');
grid on;

%% 4. 多参数解耦示例
% 封装模式：仅温度变化
T_env = 25 + 10*randn(100,1);
R_pdms = R0 * (1 + alpha * (T_env - 25));

% 自加热模式：仅NO2浓度变化
T_heater = 50;
NO2_conc = linspace(0, 5, 100);
R_gas = R0 * (1 + beta * NO2_conc);

figure('Name','多参数解耦验证');
subplot(2,1,1);
plot(T_env, (R_pdms - R0)./R0, 'r.');
title('PDMS封装模式：仅温度响应');
xlabel('温度(°C)'); ylabel('ΔR/R0');
grid on;

subplot(2,1,2);
plot(NO2_conc, (R_gas - R0)./R0, 'b-');
title('自加热模式：仅气体响应');
xlabel('NO₂浓度(ppm)'); ylabel('ΔR/R0');
grid on;

%% 5. 传感器解耦输出测试
[gas_ppm, temp_c] = decouple_signal(950, 1);  % 示例值
disp(['解耦结果：温度 = ', num2str(temp_c), '°C, NO₂浓度 = ', num2str(gas_ppm), ' ppm']);

%% ============== 子函数 ==============
function [lig_matrix, vox_matrix] = generate_heterojunction(grid_size, porosity, vox_ratio)
    % 生成LIG多孔骨架
    lig_matrix = rand(grid_size, grid_size, grid_size) > porosity;
    % 嵌入VOX颗粒
    vox_mask = rand(size(lig_matrix)) < vox_ratio;
    vox_matrix = lig_matrix & vox_mask;
end

function c_distribution = simulate_gas_diffusion_2d(D, c_surface, grid_size, voxel_size)
    model = createpde();
    L = grid_size * voxel_size;
    R1 = [3,4, 0,L,L,0, 0,0,L,L]';
    gm = decsg(R1, 'R1', ('R1')');
    geometryFromEdges(model, gm);

    specifyCoefficients(model, 'm',0, 'd',1, 'c',D, 'a',0, 'f',0);
    applyBoundaryCondition(model, 'dirichlet','Edge',1:4,'u',c_surface);
    setInitialConditions(model, 0);
    tlist = linspace(0, 50, 20);
    generateMesh(model, 'Hmax', voxel_size * 5);  % 根据单位尺寸创建网格
    result = solvepde(model, tlist);

    c_distribution = result.NodalSolution(:, end);

    figure('Name','NO₂扩散二维分布');
    pdeplot(model, 'XYData', c_distribution, 'Contour','on');
    title('NO₂扩散稳态分布');
    xlabel('X'); ylabel('Y');
end

function [gas_conc, temp] = decouple_signal(R_measured, heating_mode)
    R0 = 1000; alpha = -0.003; beta = -0.02929;
    T_heater = 50;
    if heating_mode == 0
        temp = (R_measured/R0 - 1)/alpha + 25;
        gas_conc = 0;
    else
        temp = T_heater;
        gas_conc = (R_measured/R0 - 1)/beta;
    end
end
