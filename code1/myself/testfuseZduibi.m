clear all
clc
close all

%% 读取2d立体视觉数据
ab = load('./result_mat/point_filter_stereo9_2d.mat');
pcloud = ab.pcloud;
Location = pcloud.Location;
XYZ_World = ab.XYZ_World;
figure;
pcshow(pcloud);

%scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')

hold on;
plot3(XYZ_World(1,:),XYZ_World(2,:),XYZ_World(3,:),'r.')
hold on;
%% 读取3d结构光数据 
ac = load('./result_mat/point_filter_stereo9_3d.mat');
xy2 = ac.xy2;
X = ac.X;
for i = 1:size(xy2,1)
    hold on;
    %scatter3(X{i,1}(:,1),X{i,1}(:,2),X{i,1}(:,3),10,'green','filled')
    hold on
    plot3(xy2(i,1),xy2(i,2),xy2(i,3),'r.','MarkerEdgeColor','g','MarkerFaceColor','b')
    hold on;
    % text(xy2(i,1),xy2(i,2),xy2(i,3)+0.005,num2str(i),'color','green','FontWeight','bold','FontSize',12)
end



%% 读取3d结构光LM算法结果数据 

% ad = load('./result_mat/point_filter_stereo1_LM_3d.mat');
% P_fit = ad.P_fit;
% C_fit = ad.C_fit;
% r_fit = ad.r_fit;
% for i = 1:size(xy2,1)
%     hold on;
%      scatter3(P_fit(:,1),P_fit(:,2),P_fit(:,3),5,'y','filled')
%     scatter3(C_fit(:,1),C_fit(:,2),C_fit(:,3),5,'y','filled')
% end

%% 读取3d融合算法后的结果数据演示
ad = load('./result_mat/point_filter_stereo9_Fusion.mat');
P_fit = ad.P_fit;
C_fit = ad.C_fit;
for i = 1:3
    hold on;
     scatter3(P_fit(:,1),P_fit(:,2),P_fit(:,3),5,'w','filled')
    scatter3(C_fit(:,1),C_fit(:,2),C_fit(:,3),5,'w','filled')
end
grid off




