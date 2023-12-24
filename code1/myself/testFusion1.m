% function data = testFusion(X,P_fit,C_fit,XYZ_World) 
clear all
clc
close all
dcenter = [];
file_name = 'point_filter_stereo1';
%文件读取2d信息
ab = load('./result_mat/point_filter_stereo1_2d.mat');
pcloud = ab.pcloud;
Location = pcloud.Location;
XYZ_World = ab.XYZ_World;
%文件读取3d信息
ac = load('./result_mat/point_filter_stereo1_3d.mat');
xy2 = ac.xy2;
X = ac.X;
figure;
dcenter = XYZ_World(:,1:61:200);

P_fit = [];
C_fit = [];
lamda = [];
r_fit =[]; 
DataXFinal = {};
for i  = 1:4
    D=pdist2(dcenter(:, i)',xy2);
    index = find(D<10);
    
    %获取3d中的 第i个孔的边缘点信息
    D3point = X{index};
    %获取2d中的 第i个孔的边缘点信息
    index = i;
    Lindex = (index-1)*60+index+1;
    Rindex = (index)*60+index;
    range = Lindex:1:Rindex;
    
    %选取多少个点
    number = 60;
    randnumber = sort(randperm(numel(range),number));
    range = range(randnumber);
  
    D2point = XYZ_World(:,range)';
    %可视化验证画的孔是否是一致
    ab = load('./result_mat/point_filter_stereo1_2d.mat');
    pcloud = ab.pcloud;
    Location = pcloud.Location;
    XYZ_World = ab.XYZ_World;
    figure(i);
    scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')
    % hold on
    % scatter3(D2point(:,1),D2point(:,2),D2point(:,3),5,'b','filled')
    % hold on
    % scatter3(D3point(:,1),D3point(:,2),D3point(:,3),5,'b','filled')


    %开始处理每一个圆根据 D2信息与D3信息

    DataFinal = [D2point  ; D3point];

   
    XY = DataFinal;
    centroid = mean(XY,1);   % the centroid of the data set
    
    XFInal = XY(:,1) - centroid(1);  %  centering data
    YFInal = XY(:,2) - centroid(2);  %  centering data
    ZFInal = XY(:,3) - centroid(3);  %  centering data
    P_centered = [XFInal YFInal ZFInal];
    %% test SVD
    [U,S,V]=svd([XFInal YFInal ZFInal],0);
    normal = V(:,3);
   
    %%
    % d = -dot(centroid, normal);
    P_xy = rodrigues_rot(P_centered(1:end-1,:), normal, [0,0,1]); % 映射到2d plane通过2d圆检测来找到圆心与半径 以及法向量
    % P_xy
    
    DataX = P_xy(1:end-1,1:2);
    DataXFinal{i} = DataX;
    % fit by LM method 
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
    


    % scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')
    hold on
    scatter3(P_fit(:,1),P_fit(:,2),P_fit(:,3),5,'b','filled')
    hold on
    scatter3(C_fit(:,1),C_fit(:,2),C_fit(:,3),5,'r','filled')



end


save(strcat('./result_mat/', file_name ,'_Fusion.mat'),"P_fit","C_fit");


for i=1:size(xy2,1)
fprintf('编号%d：圆心坐标为[%.4f,%.4f,%.4f]，半径为%.4f,\n',[i,C_fit(i,:),r_fit(i,:)])
% 如果不需要加编号注释后两行
    plot3([C_fit(i,1),C_fit(i,1)+0.01*lamda(i,1)],[C_fit(i,2),C_fit(i,2)+0.01*lamda(i,2)],[C_fit(i,3),C_fit(i,3)+0.01*lamda(i,3)],'r-')
    text(C_fit(i,1),C_fit(i,2),C_fit(i,3)+0.005,num2str(i),'color','k','FontWeight','bold')
end
  
for i=1:size(xy2,1)

    figure(i*100);
    DataY = DataXFinal{i};
    scatter(DataY(:,1),DataY(:,2),10,'r','filled')
    hold off

end
 
 












% end