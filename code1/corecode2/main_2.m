clear all
clc 
close all
warning off
%% Step1：读取数据文件
filpath = './data/';
file_name = 'point_filter_stereo6-1';
file_path_name = strcat(filpath,file_name);
filename = [file_path_name ,'.pcd'];
% cloud = pcread(strcat('./',filename));%文件名称自行更换
cloud = pcread(filename);
mean_ = 0;  % 噪声的均值
std_dev = 0;  % 噪声的标准差 
% 生成与点云数据矩阵相同大小的高斯噪声
% 添加噪声
Location=cloud.Location;
noise = std_dev * randn(size(Location)) + mean_;
Location = Location + noise;
Location=double(Location);
% %% Step2：杂点剔除
%DBSCAN聚类筛除杂点
epsilon=1; %最大间距 需要参考点之间的距离设置合适的间距
MinPts=20; %半径内最少满足纳入集群的个数 
[IDX, isnoise]=dbscan(Location,epsilon,MinPts);%IDX返回类别
tabel=tabulate(IDX);%统计频次
[~,o]=max(tabel(:,2));%确定最大频次的类别
a=find(IDX==tabel(o,1));
Location=Location(a,:);%提取数据点
%计算K距离，剔除距离较大的点
K=10;
C=[];
for i=1:size(Location,1)
    % 计算每个点和其他所有点之间的距离
    d=pdist2(Location(i,:),Location);
    d=sort(d);
    if d(K)>15
        C=[C,i];
    end
end
Location(C,:)=[];
%% Step3：初步定位xy面上孔的位置
%xy面图像化定位孔洞
%网格化，以0.001精度为一个像素点，转化为二维图像（0是黑色，1是白色），便于对圆进行定位
x = linspace(min(Location(:,1)), max(Location(:,1)), 40);
y = linspace(min(Location(:,2)), max(Location(:,2)), 40);
% 
% x=min(Location(:,1)):0.75:max(Location(:,1));
% y=min(Location(:,2)):0.5:max(Location(:,2));
I=zeros(length(y)-1,length(x)-1);
for i=1:length(y)-1
    for j=1:length(x)-1
        a1=[];a2=[];a3=[];a4=[];
        a1=find(Location(:,2)>y(i));
        if length(a1)>0
            a2=find(Location(a1,2)<y(i+1));
            if length(a2)>0
                a3=find(Location(a1(a2),1)>x(j));
                if length(a3)>0
                    a4=find(Location(a1(a2(a3)),1)<x(j+1));
                end
            end
        end
        if length(a4)>0
            I(i,j)=1;
        end
    end
end
%由于点云分布不均，可能会产生一些杂点，这里对杂点过滤一次
for i=2:size(I,1)-1
    for j=2:size(I,2)-1
        if sum(I(i-1:i+1,j-1:j+1),'all')>=7
            I(i,j)=1;
        end
    end
end

I = imbinarize(I);
%I = keepCentralRegion(I);

%% 尝试SVD方法映射到平面 并给予svd去识别圆。
% XY = Location;
% centroid = mean(XY,1);   % the centroid of the data set
% 
% X = XY(:,1) - centroid(1);  %  centering data
% Y = XY(:,2) - centroid(2);  %  centering data
% Z = XY(:,3) - centroid(3);  %  centering data
% P_centered = [X Y Z];
% [U,S,V]=svd([X Y Z],0);
% 
% normal = V(:,3);
% % d = -dot(centroid, normal)
% P_xy = rodrigues_rot(P_centered, normal, [0,0,1]);
% % P_xy
% 
% I = P_xy(:,1:2);

%% %识别圆，套用函数
[centers1,radii1] = imfindcircles(I,[2 15],'ObjectPolarity','dark','Sensitivity',0.8, 'Method','TwoStage');
disp('识别圆成功')
disp(radii1)
%centers返回圆中心点，radii返回对应半径，[2 15]是设置的识别圆的半径范围，0.95是设置的边缘梯度阈值
%'dark'圆形目标比背景暗，'twostage'两阶段圆形 Hough 变换
%检验中心点，分别统计半斤以内的黑点数以及一倍到二倍半斤内的黑点数，设置规则将不合理的剔除
c=[];
s=[];
for i=1:size(centers1,1)
    s1=[];
    s2=[];
    for j=1:size(I,1)
        for k=1:size(I,2)
            if sqrt(sum(([k,j]-centers1(i,:)).^2))<=radii1(i)
                s1=[s1,I(j,k)];
            end
            if sqrt(sum(([k,j]-centers1(i,:)).^2))>radii1(i) & sqrt(sum(([k,j]-centers1(i,:)).^2))<=2*radii1(i)
                s2=[s2,I(j,k)];
            end
        end
    end
    s=[s;sum(s1)/length(s1),sum(s2)/length(s2)];
    if sum(s1)/length(s1)>0.5 | sum(s2)/length(s2)<0.5 | centers1(i,1)<=3 | centers1(i,1)>=size(I,2)-2 | centers1(i,2)<=3 | centers1(i,2)>=size(I,1)-2
        c=[c,i];
    end
end
centers1(c,:)=[];
radii1(c,:)=[];
%可视化
figure(1)
imshow(I)
viscircles(centers1, radii1,'EdgeColor','c');
title('xy面定位')
figure(2)
hold on
scatter(Location(:,1),Location(:,2),5,'c','filled')
centers1=round(centers1);
xy=[x(centers1(:,1))',y(centers1(:,2))'];
scatter(xy(:,1),xy(:,2),10,'r','filled')
xlabel('x')
ylabel('y')
%%  %yz面图像化定位孔洞
%网格化，同理
% y=min(Location(:,2)):0.001:max(Location(:,2));
% z=min(Location(:,3)):0.0002:max(Location(:,3));
%y=min(Location(:,2)):0.5:max(Location(:,2));
% z=min(Location(:,3)):0.1:max(Location(:,3));
y = linspace(min(Location(:,2)), max(Location(:,2)), 40);
z = linspace(min(Location(:,3)), max(Location(:,3)), 40);

I_yz = zeros(length(z)-1,length(y)-1);
for i=1:length(z)-1
    for j=1:length(y)-1
        a1=[];a2=[];a3=[];a4=[];
        a1=find(Location(:,3)>z(i));
        if length(a1)>0
            a2=find(Location(a1,3)<z(i+1));
            if length(a2)>0
                a3=find(Location(a1(a2),2)>y(j));
                if length(a3)>0
                    a4=find(Location(a1(a2(a3)),2)<y(j+1));
                end
            end
        end
        if length(a4)>0
            I_yz(i,j)=1;
        end
    end
end
%过滤杂点，同理
for i=2:size(I_yz,1)-1
    for j=2:size(I_yz,2)-1
        if sum(I_yz(i-1:i+1,j-1:j+1),'all')>=7
            I_yz(i,j)=1;
        end
    end
end
I_yz = imbinarize(I_yz);
%I_yz = keepCentralRegion(I_yz);

%识别圆，同理
[centers2,radii2] = imfindcircles(I_yz,[2 15],'ObjectPolarity','dark','Sensitivity',0.95,'Method','twostage');
%centers返回圆中心点，radii返回对应半径，[1 15]是设置的识别圆的半径范围，0.95是设置的边缘梯度阈值
%'dark'圆形目标比背景暗，'twostage'两阶段圆形 Hough 变换
%检验中心点，同理
c=[];
s=[];
for i=1:size(centers2,1)
    s1=[];
    s2=[];
    for j=1:size(I_yz,1)
        for k=1:size(I_yz,2)
            if sqrt(sum(([k,j]-centers2(i,:)).^2))<=radii2(i)
                s1=[s1,I_yz(j,k)];
            end
            if sqrt(sum(([k,j]-centers2(i,:)).^2))>radii2(i) & sqrt(sum(([k,j]-centers2(i,:)).^2))<=1.2*radii2(i)
                s2=[s2,I_yz(j,k)];
            end
        end
    end
    s=[s;sum(s1)/length(s1),sum(s2)/length(s2)];
    if sum(s1)/length(s1)>0.5 | sum(s2)/length(s2)<0.5 | centers2(i,1)<=3 | centers2(i,1)>=size(I,2)-2 | centers2(i,2)<=3 | centers2(i,2)>=size(I,1)-2
        c=[c,i];
    end
end
centers2(c,:)=[];
radii2(c,:)=[];
%可视化
f = figure(3);
ax = axes(f);
imshow(I_yz, 'Parent', ax);
viscircles(ax, centers2, radii2, 'EdgeColor', 'c');
title('yz面定位')
figure(4)
hold on
scatter(Location(:,2),Location(:,3),5,'c','filled')
centers2=round(centers2);
yz=[y(centers2(:,1))',z(centers2(:,2))'];
scatter(yz(:,1),yz(:,2),10,'r','filled')
xlabel('y')
ylabel('z')

%% Step4：确定三维中心点坐标

%% %%xy面
xy_1=[];
for i=1:size(xy,1)
    %筛选中心周围一定距离的点（减少运算量）
    d=pdist2(xy(i,:),Location(:,1:2));
    [d,index]=sort(d);
    a=find(d<=5*d(1));
    index=index(a);
    xyz=Location(index,:);
    %计算各点相对中心点的角度
    theta=[];
    for j=1:size(xyz,1)
        theta(j)=atan((xyz(j,2)-xy(i,2))/(xyz(j,1)-xy(i,1)));
        if xyz(j,1)<xy(i,1) & xyz(j,2)<xy(i,2)%不同象限的点算出的角度需要进行调整
            theta(j)=theta(j)+pi;
        elseif xyz(j,1)<xy(i,1) & xyz(j,2)>xy(i,2)
            theta(j)=theta(j)+pi;
        elseif xyz(j,1)>xy(i,1) & xyz(j,2)<xy(i,2)
            theta(j)=theta(j)+2*pi;
        end
    end
    %计算各点到中心点的距离
    d=pdist2(xy(i,:),xyz(:,1:2));
    %以每30°划分，在每个角度区域找寻边界点，记录最小距离1.05倍范围的最近的几个点
    t=[0:30:360]./180.*pi;
    xx=[];
    for k=1:length(t)-1
        a1=find(theta>t(k));
        if length(a1)>0
            a2=find(theta(a1)<t(k+1));
            if length(a2)>0
                d1=min(d(a1(a2)));
                a3=find(d(a1(a2))<=d1*1.05);
                xx=[xx;xyz(a1(a2(a3)),:)];
            end
        end
    end
    xy_1(i,:)=[xy(i,:),(max(xx(:,3))+min(xx(:,3)))/2];%用所属边界点的坐标均值更新中心点（三维坐标）
end
%% %yz面，计算同理
yz_1=[];
for i=1:size(yz,1)
    %筛选中心周围一定距离的点（减少运算量）
    d=pdist2(yz(i,:),Location(:,2:3));
    [d,index]=sort(d);
    a=find(d<=5*d(1));
    index=index(a);
    xyz=Location(index,:);
    %计算各点相对中心点的角度
    theta=[];
    for j=1:size(xyz,1)
        theta(j)=atan((xyz(j,3)-yz(i,2))/(xyz(j,2)-yz(i,1)));
        if xyz(j,2)<yz(i,1) & xyz(j,3)<yz(i,2)
            theta(j)=theta(j)+pi;
        elseif xyz(j,2)<yz(i,1) & xyz(j,3)>yz(i,2)
            theta(j)=theta(j)+pi;
        elseif xyz(j,2)>yz(i,1) & xyz(j,3)<yz(i,2)
            theta(j)=theta(j)+2*pi;
        end
    end
    %计算各点到中心点的距离
    d=pdist2(yz(i,:),xyz(:,2:3));
    %以每60°划分
    t=[0:60:360]./180.*pi;
    xx=[];
    for k=1:length(t)-1
        a1=find(theta>t(k));
        if length(a1)>0
            a2=find(theta(a1)<t(k+1));
            if length(a2)>0
                d1=min(d(a1(a2)));
                a3=find(d(a1(a2))<=d1);
                xx=[xx;xyz(a1(a2(a3)),:)];
            end
        end
    end
    yz_1(i,:)=[mean(xx(:,1)),yz(i,:)];%用所属边界点的坐标均值更新中心点（三维坐标）
end
%物件是近似平放的，因此xy面提取到的中心点最多，则以xy面找到的中心点为主，yz面进行补充
%xy面和yz面找到的中心有重复的，将不重复的添加进来
xy1=xy_1;
for i=1:size(yz_1,1)
    d=pdist2(xy1,yz_1(i,:))
    if min(d)>0.01
        xy1=[xy1;yz_1(i,:)];
    end
end
xy1 = mean(xy1,1);
disp(xy1)
%可视化
figure(5)
scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')
hold on
scatter3(xy1(:,1),xy1(:,2),xy1(:,3),10,'r','filled')
xlabel('x')
ylabel('y')
zlabel('z')
view(70,60)
title('三维圆心定位')

%%  Step5：在三维矫正孔中心位置
X=[];
xy2=xy1;%初始化矩阵
r=[];
for i=1:size(xy2,1)
    %筛选中心周围的点（减少运算量）
    d=pdist2(xy2(i,:),Location);
    [d,index]=sort(d);
    a=find(d<=5*min(d));
    index=index(a);
    xyz=Location(index,:);
    %计算各点相对中心点的在xy方向的角度
    theta1=[];
    for j=1:size(xyz,1)
        theta1(j)=atan((xyz(j,2)-xy2(i,2))/(xyz(j,1)-xy2(i,1)));
        if xyz(j,1)<xy2(i,1) & xyz(j,2)<xy2(i,2)
            theta1(j)=theta1(j)+pi;
        elseif xyz(j,1)<xy2(i,1) & xyz(j,2)>xy2(i,2)
            theta1(j)=theta1(j)+pi;
        elseif xyz(j,1)>xy2(i,1) & xyz(j,2)<xy2(i,2)
            theta1(j)=theta1(j)+2*pi;
        end
    end
    %计算各点相对中心点的在yz方向的角度
    theta2=[];
    for j=1:size(xyz,1)
        theta2(j)=atan((xyz(j,3)-xy2(i,3))/(xyz(j,2)-xy2(i,2)));
        if xyz(j,2)<xy2(i,2) & xyz(j,3)<xy2(i,3)
            theta2(j)=theta2(j)+pi;
        elseif xyz(j,2)<xy2(i,2) & xyz(j,3)>xy2(i,3)
            theta2(j)=theta2(j)+pi;
        elseif xyz(j,2)>xy2(i,2) & xyz(j,3)<xy2(i,3)
            theta2(j)=theta2(j)+2*pi;
        end
    end
    %xy和yz面分别以每10°、10°划分寻找边界点
    t1=[0:10:360]./180.*pi;
    t2=[0:10:360]./180.*pi;
    xx=[];
    for k1=1:length(t1)-1
        for k2=1:length(t2)-1
            a1=find(theta1>=t1(k1));
            if length(a1)>0
                a2=find(theta1(a1)<=t1(k1+1));
                if length(a2)>0
                    a3=find(theta2(a1(a2))>=t2(k2));
                    if length(a3)>0
                        a4=find(theta2(a1(a2(a3)))<=t2(k2+1));
                        if length(a4)>0
                            d1=min(d(a1(a2(a3(a4)))));
                            a5=find(d(a1(a2(a3(a4))))<=d1);
                            xx=[xx;xyz(a1(a2(a3(a4(a5)))),:)];
                        end
                    end
                end
            end
        end
    end
    %因为角度原因可能会有距离较远的点，这里取距离的中位数作为阈值，剔除距离较远的点
    d=pdist2(xy2(i,:),xx);
    d1=sort(d);
    n=length(d);
    xx(find(d>d1(fix(n/2))),:)=[];
    %拟合三维圆公式（非线性最小二乘法，其中拟合算法是信赖域算法）
    f=@(a,x) (x(:,1)-a(1)).^2+(x(:,2)-a(2)).^2+(x(:,3)-a(3)).^2-a(4)^2;
    x0=[xy2(i,:),mean(pdist2(xy2(i,:),xx))];
    lb=[xy1(i,1:2)-0.001,xy1(i,3)-0.0001,0];
    ub=[xy1(i,1:2)+0.001,xy1(i,3)+0.0001,15];
    option=[];option.Display='off';
    a=lsqcurvefit(f,x0,xx,zeros(size(xx,1),1),lb,ub,option); 
    
    % figure(6)
    % times = linspace(xx(1),xx(end));
    % plot(xx,zeros(size(xx,1),1),'ko',times,f(a,times),'b-')
    
    xy2(i,:)=a(1:3);%圆心
    r(i)=a(4);%圆半径
    disp('======')
    disp(r(i));
    X{i,1}=xx;%记录边界点
    XXX00{i,1} = x0;% 记录初始点
    save(strcat('./result_mat/', file_name ,'.mat'),"XXX00","X","xy2",'Location');
end
%可视化
figure(6)
hold on
scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')
scatter3(xy2(:,1),xy2(:,2),xy2(:,3),10,'r','filled')

for i=1:size(X,1)
    scatter3(X{i,1}(:,1),X{i,1}(:,2),X{i,1}(:,3),10,'g','filled') ;
end
xlabel('x')
ylabel('y')
zlabel('z')
view(70,60)
title('三维圆心矫正')
%points_3d
points_3d = zeros(size(X{1,1},1), 3);
%disp("size (X,1):")
%disp(size(X{1,1},1))
for i=1:size(X{1,1},1)
    points_3d(i,:) =X{1,1}(i,:); 
end
%points_3d
%%
x = points_3d(:,1);
y = points_3d(:,2);
z = points_3d(:,3);
fun = @(cir) (cir(1)-x).^2 + (cir(2)-y).^2 + (cir(3)-z).^2 - cir(4)^2;
%fun = @(cir) sum((sqrt((cir(1)-x).^2 + (cir(2)-y).^2 + (cir(3)-z).^2) - cir(4)).^2);

x0 = [0, 0, 0, 1];

%[re_cir,resnorm] = lsqnonlin(fun,x0);

options = optimset('Algorithm','levenberg-marquardt','Display','off');
re_cir = lsqnonlin(fun,x0,[],[],options);

center = re_cir(1:3);
radius = abs(re_cir(4));
% disp(['圆心坐标：', num2str(center)]);
% disp(['圆半径：', num2str(radius)]);
figure(10);
hold on;
scatter3(Location(:,1),Location(:,2),Location(:,3),5,'c','filled')
scatter3(xy2(:,1),xy2(:,2),xy2(:,3),10,'r','filled')
for i=1:size(points_3d,1)
    scatter3(points_3d(i,1),points_3d(i,2),points_3d(i,3),10,'g','filled')  
end

%scatter3(xy2(:,1),xy2(:,2),xy2(:,3),10,'r','filled')
xlabel('x')
ylabel('y')
zlabel('z')
view(70,60)
title('三维圆心矫正之后')
%%


%% Step6：求法向量
%首先拟合曲面(最小二乘法)，用于计算圆心所在的斜率和角度，方便计算法向量
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
scatter3(xy2(:,1),xy2(:,2),xy2(:,3),10,'r','filled')
for i=1:size(X,1)
    scatter3(X{i,1}(:,1),X{i,1}(:,2),X{i,1}(:,3),10,'g','filled')

    t = linspace(0, 2*pi, 100);
    u = X{1,1}(1,:) - xy2(1,:);
    P_fitcircle = generate_circle_by_vectors(t, xy2, r(1), lamda', u);
    scatter3(P_fitcircle(:,1),P_fitcircle(:,2),P_fitcircle(:,3),5,'b','filled');

    %绘制法向量
    plot3([xy2(i,1),xy2(i,1)+0.01*lamda(i,1)],[xy2(i,2),xy2(i,2)+0.01*lamda(i,2)],[xy2(i,3),xy2(i,3)+0.01*lamda(i,3)],'r-')
    text(xy2(i,1),xy2(i,2),xy2(i,3)+0.005,num2str(i),'color','k','FontWeight','bold')
end
xlabel('x')
ylabel('y')
zlabel('z')
view(70,60)
title('结果')

%输出结果
for i=1:size(xy2,1)
    fprintf('编号%d：圆心坐标为[%.4f,%.4f,%.4f]，半径为%.4f，圆心法向量为[%.4f,%.4f,%.4f]\n',[i,xy2(i,:),radius,lamda(i,:)])
end
%%    后面加进来保存数据
% 保存边缘点

x_center = xy2(1,1);
y_center = xy2(1,2);
z_center = xy2(1,3);
radius = r(1);
radius = 4.7273;
theta_values = linspace(0, 2*pi, 100); % 经度范围
phi_values = [0];      % 纬度范围
% 初始化坐标数组
x_coords = [];
y_coords = [];
z_coords = [];
% 生成圆上的点的直角坐标
for i = 1:length(theta_values)
    for j = 1:length(phi_values)
        theta = theta_values(i);
        phi = phi_values(j);
        x = x_center + radius * cos(theta) ;
        y = y_center + radius * sin(theta) ;
        z = z_center ;  
        x_coords = [x_coords, x];
        y_coords = [y_coords, y];
        z_coords = [z_coords, z];
    end
end
P_fitcircle = [x_coords' y_coords' z_coords'];

save(strcat('./result_circle/', file_name ,'result_LS_boundPoint','.mat'),"P_fitcircle");
 %将结果圆心以及半径写入到result_LS文件中
 save_file = ['result_circle/',file_name ,'result_LS.txt'];
 fid = fopen( save_file,'a');
  fprintf(fid,'%s\n',file_name);
 for i=1:size(xy2,1)
    fprintf(fid, '[%.4f,%.4f,%.4f, %.4f]\n', [xy2(i,:),r(i)]);  % 将字符串写入文件
 end
 % str = '==============================';
 % 每次检测完一个样本后，就用===================来将样本隔开
 fclose(fid);