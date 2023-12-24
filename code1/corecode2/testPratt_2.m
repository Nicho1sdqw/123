

% figure(100)
% scatter3(X{1,1}(:,1),X{1,1}(:,2),X{1,1}(:,3),10,'r','filled')
% hold on
% 
% scatter3(X{2,1}(:,1),X{2,1}(:,2),X{2,1}(:,3),10,'r','filled')
% 
% 
% hold on
% scatter3(X{3,1}(:,1),X{3,1}(:,2),X{3,1}(:,3),10,'r','filled')
% 
%  hold on
% scatter3(X{4,1}(:,1),X{4,1}(:,2),X{4,1}(:,3),10,'r','filled')
% 
% finaX = [X{1,:};X{2,:};X{3,:};X{4,:}];

% XY = finaX;
%     centroid = mean(XY,1);   % the centroid of the data set
% 
%     X = XY(:,1) - centroid(1);  %  centering data
%     Y = XY(:,2) - centroid(2);  %  centering data
%     Z = XY(:,3) - centroid(3);  %  centering data
%     P_centered = [X Y Z];
%     [U,S,V]=svd([X Y Z],0);
% 
%     normal = V(:,3);
%     % d = -dot(centroid, normal)
%     P_xy = rodrigues_rot(P_centered, normal, [0,0,1]); % 映射到2d plane通过2d圆检测来找到圆心与半径 以及法向量
%     % P_xy
% 
%     DataX = P_xy(:,1:2);
%     scatter(DataX(:,1),DataX(:,2),10,'r','filled')

%%  
clear all
close all
clc
file_name = 'point_filtered_mono1-3';
load ./result_mat/point_filtered_mono1-3.mat;
P_fit = [];
C_fit = [];
lamda = [];
r_fit =[];
for i = 1:size(xy2,1)
    XY = X{i,1};
    centroid = mean(XY,1);   % the centroid of the data set
    
    XFInal = XY(:,1) - centroid(1);  %  centering data
    YFInal = XY(:,2) - centroid(2);  %  centering data
    ZFInal = XY(:,3) - centroid(3);  %  centering data
    
    P_centered = [XFInal YFInal ZFInal];


    %% test SVD%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [U,S,V]=svd([XFInal YFInal ZFInal],0);
    normal = V(:,3)
    %% test in IRCUR%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % D = [XFInal YFInal ZFInal];
    % rank = 3;  
    % para.beta_init = max(abs(D(:)));
    % para.beta      = para.beta_init;
    % para.tol       = 1e-5;
    % para.con       = 5;
    % para.resample  = false;
    % [C1, pinv_U1, R1, ircur_d_timer, ircur_d_err] = IRCUR( D, rank, '');
    % normal = R1(1:3,3) 
    % recover_err_ircur_r = norm(D - C1 * pinv_U1 * R1, 'fro') / norm(D,'fro');
    

    % ttPoint = [];
    % ttPoint(:,:,1) = [XFInal];
    % ttPoint(:,:,2) = [YFInal];
    % ttPoint(:,:,3) = [ZFInal];
    % [U,S,V,sigmas]=ttr1svd(ttPoint);
    % normal = V{4}


    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % end 
    % normal = V(:,3);
    % d = -dot(centroid, normal)
    P_xy = rodrigues_rot(P_centered, normal, [0,0,1]); % 映射到2d plane通过2d圆检测来找到圆心与半径 以及法向量
    % P_xy
    
    DataX = P_xy(:,1:2);
    figure(i);
    scatter(DataX(:,1),DataX(:,2),10,'r','filled')
    hold on;
    %% fit by Pratt method 
    Par = CircleFitByPratt(DataX);

    xc=Par(1);
    yc=Par(2);
    r =Par(3);
    C = rodrigues_rot(([xc,yc,0]), [0,0,1], normal) + centroid;
    
    t = linspace(0, 2*pi, 100);
    u = X{i,1}(1,:) - C;
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal', u);
    P_fit = [P_fit; P_fitcircle];
    C_fit = [C_fit; C];
    r_fit = [r_fit;r];
    lamda = [lamda;normal'];
end
%% project 2d to 3d back
 
save(strcat('./result_circle/', file_name ,'result_Pratt_boundPoint','.mat'),"P_fitcircle");   

figure(100)
scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')
hold on
for i = 1:size(xy2,1)
    scatter3(X{i,1}(:,1),X{i,1}(:,2),X{i,1}(:,3),10,'green','filled')
    hold on
    
end
grid off
% scatter3(X{1,1}(:,1),X{1,1}(:,2),X{1,1}(:,3),10,'r','filled')
% hold on
% scatter3(X{2,1}(:,1),X{2,1}(:,2),X{2,1}(:,3),10,'r','filled')
% hold on
% scatter3(X{3,1}(:,1),X{3,1}(:,2),X{3,1}(:,3),10,'r','filled')
% hold on
% scatter3(X{4,1}(:,1),X{4,1}(:,2),X{4,1}(:,3),10,'r','filled')
% hold on
% scatter3(xy2(:,1),xy2(:,2),xy2(:,3),10,'r','filled')


scatter3(P_fit(:,1),P_fit(:,2),P_fit(:,3),5,'b','filled')
hold on 
scatter3(C_fit(:,1),C_fit(:,2),C_fit(:,3),5,'r','filled')
text(C_fit(:,1),C_fit(:,2),C_fit(:,3)+0.005,num2str(i),'color','k','FontWeight','bold')
hold on 
% legend('New Center and Circle')

legend([{'New Center and Circle'},{'Origianl Center and Circle'}])

 for i=1:size(xy2,1)
    fprintf('编号%d：圆心坐标为[%.4f,%.4f,%.4f]，半径为%.4f，圆心法向量为[%.4f,%.4f,%.4f]\n',[i,C_fit(i,:),r_fit(i,:),lamda(i,:)])
 end

 
 % 将结果圆心以及半径写入到result_LM文件中
 save_file = ['result_circle/',file_name ,'result_Pratt.txt'];
 fid = fopen( save_file,'a');
  fprintf(fid,'%s\n',file_name);
 for i=1:size(xy2,1)
    fprintf(fid, '[%.4f,%.4f,%.4f, %.4f]\n', [C_fit(i,:),r_fit(i,:)]);  % 将字符串写入文件
 end
 str = '==============================';
 % 每次检测完一个样本后，就用===================来将样本隔开
 fclose(fid);
 