

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
clc;
% load LocationLMda5.mat;
file_name = 'point_filter_stereo6-1';
load 'result_mat/point_filter_stereo6-1.mat';
P_fit = [];
C_fit = [];
lamda = [];
r_fit =[];
rank = 3;        % rank  
for i = 1:size(xy2,1)
    XY = [X{i,1}; XXX00{i,1}(1,1:3)];
    centroid = mean(XY,1);   % the centroid of the data set
    
    XFInal = XY(:,1) - centroid(1);  %  centering data
    YFInal = XY(:,2) - centroid(2);  %  centering data
    ZFInal = XY(:,3) - centroid(3);  %  centering data
    P_centered = [XFInal YFInal ZFInal];
    %% test SVD
    [U,S,V]=svd([XFInal YFInal ZFInal],0);
    normal = V(:,3);
    %% test in IRCUR
    % D = [XFInal YFInal ZFInal];
    % para.beta_init = 1.5*max(abs(D(:)));
    % para.beta      = para.beta_init;
    % para.tol       = 1e-8;
    % para.con       = 10;
    % para.resample  = false;
    % [C1, pinv_U1, R1, ircur_r_timer, ircur_r_err] = IRCUR( D, rank, para);
    % normal = pinv_U1(1:3,3);
    % recover_err_ircur_r = norm(D - C1 * pinv_U1 * R1, 'fro') / norm(D,'fro')
    %end 
    %%
    % d = -dot(centroid, normal);
    P_xy = rodrigues_rot(P_centered(1:end-1,:), normal, [0,0,1]); % 映射到2d plane通过2d圆检测来找到圆心与半径 以及法向量
    % P_xy
    
    DataX = P_xy(1:end-1,1:2);
    figure(i);
    scatter(DataX(:,1),DataX(:,2),10,'r','filled')
    hold on;
    %% fit by LM method 
    ParIni = [P_xy(end,1:2), [0.0001]];
    Par = CircleFitLevenbergMarquardt(DataX,ParIni);

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
 
save(strcat('./result_circle/', file_name ,'result_LM_boundPoint','.mat'),"P_fitcircle");   

figure(100)
scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')
hold on
for i = 1:size(xy2,1)

    scatter3(X{i,1}(:,1),X{i,1}(:,2),X{i,1}(:,3),10,'r','filled')
    hold on

end


scatter3(P_fit(:,1),P_fit(:,2),P_fit(:,3),5,'b','filled')
hold on 
scatter3(C_fit(:,1),C_fit(:,2),C_fit(:,3),5,'b','filled')
% legend('New Center and Circle')

legend([{'New Center and Circle'},{'Origianl Center and Circle'}])

% hold on 

 
%%

x1=[linspace(min(Location(:,1)),max(Location(:,1)),100)];
y1=[linspace(min(Location(:,2)),max(Location(:,2)),100)];
[xx1,yy1]=meshgrid(x1,y1);
%采用5次多项式形式拟合曲面
[b,~,~,~,state]=regress(Location(:,3),[ones(size(Location,1),1),Location(:,1:2),Location(:,1:2).^2,Location(:,1:2).^3,Location(:,1:2).^4,Location(:,1:2).^5]);
zz1=[];
for i=1:size(xx1,1)
    for j=1:size(xx1,2)
        zz1(i,j)=[1,xx1(i,j),yy1(i,j),xx1(i,j)^2,yy1(i,j)^2,xx1(i,j)^3,yy1(i,j)^3,xx1(i,j)^4,yy1(i,j)^4,xx1(i,j)^5,yy1(i,j)^5]*b;
    end
end
%可视化
figure(7)
hold on
scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')
surf(xx1,yy1,zz1)
shading flat
xlabel('x')
ylabel('y')
zlabel('z')
view(70,60)
title(['曲面拟合，R^2=',num2str(state(1))])
%计算法向量
lamda=[];
for i=1:size(xy2,1)
    %取中心点前后左右0.001精度位置，计算z值从而计算x和y方向的斜率，然后反正切计算角度
    o1=[xy2(i,1)-0.001,xy2(i,2)];
    o2=[xy2(i,1)+0.001,xy2(i,2)];
    o3=[xy2(i,1),xy2(i,2)-0.001];
    o4=[xy2(i,1),xy2(i,2)+0.001];
    kx=([1,o2,o2.^2,o2.^3,o2.^4,o2.^5]*b-[1,o1,o1.^2,o1.^3,o1.^4,o1.^5]*b)/0.002;
    ky=([1,o4,o4.^2,o4.^3,o4.^4,o4.^5]*b-[1,o3,o3.^2,o3.^3,o3.^4,o3.^5]*b)/0.002;
    theta1=(atan(kx)+pi/2);
    theta2=(atan(ky)+pi/2);
    %以法向量单位长度为1，分别计算x、y、z方向的单位长度
    lamda(i,:)=[cos(theta1),cos(theta2),sqrt(1-norm(cos(theta1),cos(theta2))^2)];
end
%可视化
figure(8)
hold on
scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')
%scatter3(xy2(:,1),xy2(:,2),xy2(:,3),10,'r','filled')
for i=1:size(X,1)
    scatter3(X{i,1}(:,1),X{i,1}(:,2),X{i,1}(:,3),10,'g','filled')
    scatter3(P_fit(:,1),P_fit(:,2),P_fit(:,3),5,'b','filled')
    scatter3(C_fit(:,1),C_fit(:,2),C_fit(:,3),5,'r','filled')
    %绘制法向量
    plot3([C_fit(i,1),C_fit(i,1)+0.01*lamda(i,1)],[C_fit(i,2),C_fit(i,2)+0.01*lamda(i,2)],[C_fit(i,3),C_fit(i,3)+0.01*lamda(i,3)],'r-')
    text(C_fit(i,1),C_fit(i,2),C_fit(i,3)+0.005,num2str(i),'color','k','FontWeight','bold')
end
xlabel('x')
ylabel('y')
zlabel('z')
view(70,60)
title('结果')


 for i=1:size(xy2,1)
    fprintf('编号%d：圆心坐标为[%.4f,%.4f,%.4f]，半径为%.4f，圆心法向量为[%.4f,%.4f,%.4f]\n',[i,C_fit(i,:),r_fit(i,:),lamda(i,:)])
 end
 %将结果圆心以及半径写入到result_LM文件中
 save_file = ['result_circle/',file_name ,'result_LM.txt'];
 fid = fopen( save_file,'a');
  fprintf(fid,'%s\n',file_name);
 for i=1:size(xy2,1)
    fprintf(fid, '[%.4f,%.4f,%.4f, %.4f]\n', [C(i,:),r(i)]);  % 将字符串写入文件
 end
 str = '==============================';
 % 每次检测完一个样本后，就用===================来将样本隔开
 fclose(fid);