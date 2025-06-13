
% 修复物理模型和特征表示问题
clear; clc; close all; rng(42);

%% ========== 参数设置 ==========
grid_size = 36;          % 网格尺寸
voxel_size = 0.1e-6;    % 单元尺寸 (0.1μm)
base_porosity = 0.75;    % 目标孔隙率
vox_ratio = 0.15;       % VOX掺杂比例

% 机器学习参数
num_samples = 300;      % 增加样本数量
num_epochs = 40;        % 增加训练轮次
validation_ratio = 0.2;  % 验证集比例

% 物理模型参数 - 基于真实材料数据校准
density = 5.2e3;        % 材料密度 (kg/m³) 调整为石墨烯复合材料密度
base_conductivity = 4e4; % 基础电导率 (S/m)

%% ========== 主程序开始 ==========
fprintf('==== VO/LIG高精度建模系统 ====\n');
fprintf('网格尺寸: %dx%dx%d | 样本数: %d\n', grid_size, grid_size, grid_size, num_samples);

%% 1. 精确结构生成
fprintf('生成精确结构...\n');
[lig, vox] = generate_accurate_structure(grid_size, base_porosity, vox_ratio);

% 计算结构参数
filler_ratio = sum(vox(:)) / numel(vox);
fprintf('材料参数: 孔隙率=%.3f, VOX比例=%.3f, 填充比例=%.3f\n',...
        1 - filler_ratio, vox_ratio, filler_ratio);

% 可视化结构
plot_accurate_structure(vox);

%% 2. 精确性能计算
fprintf('计算精确性能...\n');
[cond_real, sa_real] = calculate_accurate_performance(vox, voxel_size, density);
fprintf('实际性能: 电导率=%.2f S/m, 表面积=%.2f m²/g\n', cond_real, sa_real);

%% 3. 创建高精度数据集
fprintf('生成%d个高精度样本...\n', num_samples);

% 预分配数据
X_train = zeros(grid_size, grid_size, 6, num_samples); % 6通道输入
Y_train = zeros(num_samples, 2); % [电导率, 表面积]

% 记录数据范围用于归一化
cond_min = Inf; cond_max = -Inf;
sa_min = Inf; sa_max = -Inf;

progress_bar = waitbar(0, '生成样本中...');

for i = 1:num_samples
    % 参数变化 - 更广泛的参数空间
    porosity_var = max(0.60, min(0.90, base_porosity + 0.08*randn()));
    ratio_var = max(0.08, min(0.25, vox_ratio + 0.04*randn()));
    
    % 生成结构
    [~, vox_sample] = generate_accurate_structure(grid_size, porosity_var, ratio_var);
    
    % 计算精确性能
    [cond_val, sa_val] = calculate_accurate_performance(vox_sample, voxel_size, density);
    
    % 记录数据范围
    cond_min = min(cond_min, cond_val);
    cond_max = max(cond_max, cond_val);
    sa_min = min(sa_min, sa_val);
    sa_max = max(sa_max, sa_val);
    
    % 提取高级特征
    X_train(:,:,:,i) = extract_advanced_features(vox_sample);
    
    % 保存性能参数
    Y_train(i,:) = [cond_val, sa_val];
    
    % 进度更新
    progress = i/num_samples;
    waitbar(progress, progress_bar, sprintf('生成样本 %d/%d (%.0f%%)', i, num_samples, progress*100));
    
    % 定期显示样本信息
    if mod(i, 20) == 0
        fprintf('样本 %d: 电导率=%.2f, 表面积=%.2f\n', i, cond_val, sa_val);
    end
end
close(progress_bar);
fprintf('数据集创建完成! 范围: 电导率[%.2f, %.2f] S/m, 表面积[%.2f, %.2f] m²/g\n',...
        cond_min, cond_max, sa_min, sa_max);

% 归一化目标值 - 解决数量级差异问题
Y_train_normalized = normalize_performance(Y_train, cond_min, cond_max, sa_min, sa_max);

%% 4. 构建物理增强型CNN模型
fprintf('构建物理增强型CNN模型...\n');

% 输入尺寸
input_size = [grid_size, grid_size, 6];

% 深度残差网络架构
layers = [
    imageInputLayer(input_size, 'Name', 'input', 'Normalization', 'rescale-zero-one')
    
    % 残差块1
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    additionLayer(2, 'Name', 'add1')
    reluLayer('Name', 'relu_add1')
    
    % 残差块2
    convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', 2, 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv6')
    batchNormalizationLayer('Name', 'bn6')
    additionLayer(2, 'Name', 'add2')
    reluLayer('Name', 'relu_add2')
    
    % 残差块3
    convolution2dLayer(3, 128, 'Padding', 'same', 'Stride', 2, 'Name', 'conv7')
    batchNormalizationLayer('Name', 'bn7')
    reluLayer('Name', 'relu7')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv8')
    batchNormalizationLayer('Name', 'bn8')
    reluLayer('Name', 'relu8')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv9')
    batchNormalizationLayer('Name', 'bn9')
    additionLayer(2, 'Name', 'add3')
    reluLayer('Name', 'relu_add3')
    
    % 全局池化
    globalAveragePooling2dLayer('Name', 'global_pool')
    
    % 回归头
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.4, 'Name', 'dropout1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu_fc2')
    dropoutLayer(0.3, 'Name', 'dropout2')
    fullyConnectedLayer(64, 'Name', 'fc3')
    reluLayer('Name', 'relu_fc3')
    fullyConnectedLayer(2, 'Name', 'output')
    regressionLayer('Name', 'regression')
];

% 添加残差连接
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu1', 'add1/in2');
lgraph = connectLayers(lgraph, 'relu4', 'add2/in2');
lgraph = connectLayers(lgraph, 'relu7', 'add3/in2');

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', num_epochs, ...
    'MiniBatchSize', 16, ...
    'ValidationData', {X_train(:,:,:,1:floor(num_samples*validation_ratio)), ...
                       Y_train_normalized(1:floor(num_samples*validation_ratio), :)}, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 0.0005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.75, ...
    'LearnRateDropPeriod', 10, ...
    'Verbose', true, ...
    'L2Regularization', 2e-4);

%% 5. 训练物理增强型CNN模型
fprintf('训练CNN模型 (%d epochs)...\n', num_epochs);
t0 = tic;
cnn_model = trainNetwork(X_train, Y_train_normalized, lgraph, options);
training_time = toc(t0);
fprintf('模型训练完成! 耗时: %.1f秒\n', training_time);

% 保存模型
save('physical_cnn_model.mat', 'cnn_model', '-v7.3');

%% 6. 高精度预测
fprintf('执行高精度预测...\n');

% 生成测试结构
[~, vox_test] = generate_accurate_structure(grid_size, base_porosity, vox_ratio);
features_test = extract_advanced_features(vox_test);

% CNN预测 (归一化空间)
predicted_normalized = predict(cnn_model, features_test);

% 反归一化到真实物理量
predicted_cond = denormalize_value(predicted_normalized(1), cond_min, cond_max);
predicted_sa = denormalize_value(predicted_normalized(2), sa_min, sa_max);

% 实际性能
[actual_cond, actual_sa] = calculate_accurate_performance(vox_test, voxel_size, density);

% 输出结果
fprintf('\n=== 高精度预测 ===\n');
fprintf('材料属性: 孔隙率=%.3f, VOX比例=%.3f\n', base_porosity, vox_ratio);
fprintf('电导率: 预测=%.2f S/m, 实际=%.2f S/m, 误差=%.2f%%\n',...
        predicted_cond, actual_cond, abs(predicted_cond/actual_cond-1)*100);
fprintf('表面积: 预测=%.2f m²/g, 实际=%.2f m²/g, 误差=%.2f%%\n',...
        predicted_sa, actual_sa, abs(predicted_sa/actual_sa-1)*100);

% 可视化预测精度
visualize_prediction_accuracy(Y_train, predict(cnn_model, X_train),...
                               cond_min, cond_max, sa_min, sa_max,...
                               [predicted_cond, predicted_sa],...
                               [actual_cond, actual_sa]);

fprintf('\n==== 系统运行完成 ====\n');

%% ========== 物理模型修正子函数 ==========
function [lig, vox] = generate_accurate_structure(grid_size, porosity, ratio)
    % 高精度结构生成器
    
    % 1. 创建LIG骨架 - 考虑梯度分布
    [X, Y, Z] = meshgrid(1:grid_size, 1:grid_size, 1:grid_size);
    center = grid_size/2;
    dist = sqrt((X-center).^2 + (Y-center).^2 + (Z-center).^2);
    density_factor = exp(-dist/(grid_size/2)); % 中心密度高
    
    % 梯度孔隙率
    porosity_map = porosity * (1 - 0.3*density_factor);
    lig = rand(grid_size, grid_size, grid_size) > porosity_map;
    
    % 2. 添加VOX颗粒 - 基于概率分布
    num_clusters = max(1, round(ratio * grid_size^3 / 8));
    vox = false(size(lig));
    
    for i = 1:num_clusters
        % 簇中心 - 倾向于高密度区域
        center_prob = density_factor / max(density_factor(:));
        [cx, cy, cz] = weighted_center(center_prob);
        
        % 随机簇半径 (1-4体素)
        radius = 1 + 3*rand();
        
        % 创建球形簇
        cluster = false(size(lig));
        for x = max(1, floor(cx-radius)):min(grid_size, ceil(cx+radius))
            for y = max(1, floor(cy-radius)):min(grid_size, ceil(cy+radius))
                for z = max(1, floor(cz-radius)):min(grid_size, ceil(cz+radius))
                    if sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2) <= radius
                        cluster(x,y,z) = true;
                    end
                end
            end
        end
        
        vox = vox | cluster;
    end
    
    % 3. 确保电气连通性
    % 只保留主连通簇
    CC = bwconncomp(lig | vox, 6);
    if CC.NumObjects > 0
        cluster_sizes = cellfun(@numel, CC.PixelIdxList);
        [~, idx] = max(cluster_sizes);
        
        mask = false(size(lig));
        mask(CC.PixelIdxList{idx}) = true;
        
        lig = mask & lig;
        vox = mask & vox;
    end
end

function [cx, cy, cz] = weighted_center(prob_map)
    % 根据概率图选择簇中心
    total = sum(prob_map(:));
    r = rand() * total;
    cumulative = 0;
    
    for x = 1:size(prob_map,1)
        for y = 1:size(prob_map,2)
            for z = 1:size(prob_map,3)
                cumulative = cumulative + prob_map(x,y,z);
                if cumulative >= r
                    cx = x; cy = y; cz = z;
                    return;
                end
            end
        end
    end
    
    % 如果失败，返回中心
    cx = size(prob_map,1)/2;
    cy = size(prob_map,2)/2;
    cz = size(prob_map,3)/2;
end

function [conductivity, surface_area] = calculate_accurate_performance(vox, res, density)
    % 物理准确的性能计算
    
    % 1. 基本参数
    total_voxels = sum(vox(:));
    grid_volume = numel(vox);
    filler_fraction = total_voxels / grid_volume;
    
    % 2. 电导率模型 (基于渗流理论和有效介质近似)
    % 检查连通性
    CC = bwconncomp(vox, 6);
    percolated = false;
    if CC.NumObjects > 0
        % 检查是否形成渗透路径
        max_cluster_size = max(cellfun(@numel, CC.PixelIdxList));
        max_path_length = max_path_distance(vox, CC);
        percolated = (max_path_length >= max(size(vox)));
    end
    
    % 电导率模型
    if percolated
        % 使用Bruggeman有效介质近似
        base_cond = 5e4; % VO₂高导电相电导率
        cond_lig = 1e3;  % LIG电导率
        phi_c = filler_fraction; % VO₂体积分数
        phi_m = 1 - phi_c;      % 基体体积分数
        conductivity = base_cond * phi_c^1.5 * cond_lig * phi_m^1.5;
    else
        % 未渗透时的电导率
        base_cond = 500;  % VO₂低导电相电导率
        cond_lig = 1e3;   % LIG电导率
        conductivity = cond_lig * (1 - filler_fraction) + base_cond * filler_fraction;
    end
    
    % 3. 表面积计算 - 物理精确模型
    % 检测表面体素 (暴露面)
    surface_voxels = false(size(vox));
    for z = 1:size(vox,3)
        for y = 1:size(vox,2)
            for x = 1:size(vox,1)
                if ~vox(x,y,z), continue; end
                
                % 6邻域检查
                if x == 1 || ~vox(x-1,y,z), surface_voxels(x,y,z) = true; end
                if x == size(vox,1) || ~vox(x+1,y,z), surface_voxels(x,y,z) = true; end
                if y == 1 || ~vox(x,y-1,z), surface_voxels(x,y,z) = true; end
                if y == size(vox,2) || ~vox(x,y+1,z), surface_voxels(x,y,z) = true; end
                if z == 1 || ~vox(x,y,z-1), surface_voxels(x,y,z) = true; end
                if z == size(vox,3) || ~vox(x,y,z+1), surface_voxels(x,y,z) = true; end
            end
        end
    end
    
    % 计算总表面积 (m²)
    surface_voxel_count = sum(surface_voxels(:));
    voxel_area = res^2;
    total_surface = surface_voxel_count * voxel_area;
    
    % 计算比表面积 (m²/g)
    mass = grid_volume * res^3 * density; % 总质量 (kg)
    surface_area = total_surface / (mass / 1000); % 转化为g单位
end

function max_len = max_path_distance(vox, CC)
    % 计算最大路径长度
    max_len = 0;
    if CC.NumObjects == 0, return; end
    
    cluster_sizes = cellfun(@numel, CC.PixelIdxList);
    [~, max_idx] = max(cluster_sizes);
    cluster = false(size(vox));
    cluster(CC.PixelIdxList{max_idx}) = true;
    
    % 选择边界点
    boundary_points = boundary_points_finder(cluster);
    if size(boundary_points, 1) < 2, return; end
    
    % 计算最远点对之间的欧几里得距离
    D = pdist2(boundary_points, boundary_points);
    max_len = max(D(:));
end

function points = boundary_points_finder(cluster)
    % 寻找簇的边界点
    eroded = imerode(cluster, ones(3,3,3));
    boundary = cluster & ~eroded;
    [x, y, z] = ind2sub(size(cluster), find(boundary));
    points = [x, y, z];
end

function features = extract_advanced_features(vox)
    % 提取物理相关特征
    
    % 1. 密度分布投影
    xy_view = max(vox, [], 3);
    xz_view = squeeze(max(vox, [], 2));
    yz_view = squeeze(max(vox, [], 1));
    
    % 标准化尺寸
    target_size = [size(vox,1), size(vox,2)];
    xz_view = imresize(xz_view, target_size);
    yz_view = imresize(yz_view, target_size);
    
    % 2. 连通性特征
    CC = bwconncomp(vox, 6);
    connectivity_map = zeros(size(vox));
    if CC.NumObjects > 0
        [~, idx] = max(cellfun(@numel, CC.PixelIdxList));
        connectivity_map(CC.PixelIdxList{idx}) = 1;
    end
    
    conn_xy = max(connectivity_map, [], 3);
    conn_xz = squeeze(max(connectivity_map, [], 2));
    conn_yz = squeeze(max(connectivity_map, [], 1));
    
    % 标准化尺寸
    conn_xz = imresize(conn_xz, target_size);
    conn_yz = imresize(conn_yz, target_size);
    
    % 组合特征 (6通道)
    features = cat(3, ...
        xy_view, xz_view, yz_view, ...
        conn_xy, conn_xz, conn_yz);
end
function plot_accurate_structure(vox)
    % 高精度结构可视化
    figure('Position', [100,100,1000,400]);
    
    % XY平面
    subplot(1,4,1);
    imagesc(vox(:,:,round(size(vox,3)/2)));
    title('XY平面');
    axis equal; axis off;
    colormap(jet);
    
    % XZ平面
    subplot(1,4,2);
    imagesc(squeeze(vox(round(size(vox,1)/2),:,:))');
    title('XZ平面');
    axis equal; axis off;
    
    % YZ平面
    subplot(1,4,3);
    imagesc(squeeze(vox(:,round(size(vox,2)/2),:))');
    title('YZ平面');
    axis equal; axis off;
    
    % 3D可视化
    subplot(1,4,4);
    vox = double(vox);
    p = patch(isosurface(vox,0.5));
    p.FaceColor = 'red';
    p.EdgeColor = 'none';
    daspect([1 1 1]);
    view(3);
    axis tight;
    camlight;
    lighting gouraud;
    title('3D结构');
    
    saveas(gcf, 'accurate_structure.png');
end

function Y_norm = normalize_performance(Y, cond_min, cond_max, sa_min, sa_max)
    % 归一化性能参数
    % 使用min-max归一化到[0,1]范围
    
    % 电导率归一化
    cond_norm = (Y(:,1) - cond_min) / (cond_max - cond_min);
    
    % 表面积归一化
    sa_norm = (Y(:,2) - sa_min) / (sa_max - sa_min);
    
    Y_norm = [cond_norm, sa_norm];
end

function value = denormalize_value(norm_val, min_val, max_val)
    % 反归一化
    value = norm_val * (max_val - min_val) + min_val;
end

function visualize_prediction_accuracy(Y_train, Y_pred_norm, cond_min, cond_max, sa_min, sa_max, predicted, actual)
    % 可视化预测精度
    
    % 反归一化训练集预测值
    Y_pred_cond = denormalize_value(Y_pred_norm(:,1), cond_min, cond_max);
    Y_pred_sa = denormalize_value(Y_pred_norm(:,2), sa_min, sa_max);
    
    figure('Position', [100,100,1000,400]);
    
    % 电导率预测精度
    subplot(1,2,1);
    scatter(Y_train(:,1), Y_pred_cond, 20, 'filled');
    hold on;
    plot(actual(1), predicted(1), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
    plot([min(Y_train(:,1)) max(Y_train(:,1))], [min(Y_train(:,1)) max(Y_train(:,1))], 'r--');
    title('电导率预测精度');
    xlabel('真实值 (S/m)');
    ylabel('预测值 (S/m)');
    grid on;
    legend('训练样本', '测试点', '理想线', 'Location', 'northwest');
    
    % 表面积预测精度
    subplot(1,2,2);
    scatter(Y_train(:,2), Y_pred_sa, 20, 'filled');
    hold on;
    plot(actual(2), predicted(2), 'ro', 'MarkerSize', 12, 'LineWidth', 2);
    plot([min(Y_train(:,2)) max(Y_train(:,2))], [min(Y_train(:,2)) max(Y_train(:,2))], 'r--');
    title('表面积预测精度');
    xlabel('真实值 (m²/g)');
    ylabel('预测值 (m²/g)');
    grid on;
    legend('训练样本', '测试点', '理想线', 'Location', 'northwest');
    
    saveas(gcf, 'prediction_accuracy.png');
end
