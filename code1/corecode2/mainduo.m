clear all
clc
close all
warning off
%% Step1：读取数据文件
filpath = './data/';
file_name = 'point_filtered_mono8';
file_path_name = strcat(filpath,file_name);
filename = [file_path_name ,'.pcd'];
cloud = pcread(filename);
Location=cloud.Location;

%%
% 提取 XY 坐标
xy = Location(:, 1:2);

% 计算图像的尺寸
xMin = min(xy(:,1));
xMax = max(xy(:,1));
yMin = min(xy(:,2));
yMax = max(xy(:,2));

% 计算范围和图像大小
xRange = xMax - xMin;
yRange = yMax - yMin;
imgWidth = ceil(xRange);
imgHeight = ceil(yRange);

% 初始化图像矩阵，增加1是为了防止索引为0
imageMatrix = zeros(imgHeight + 1, imgWidth + 1);

% 投影点云到图像矩阵
for i = 1:size(xy, 1)
    x = round(xy(i, 1) - xMin + 1);
    y = round(xy(i, 2) - yMin + 1);
    if (x > 0 && y > 0 && x <= imgWidth && y <= imgHeight)
        imageMatrix(y, x) = imageMatrix(y, x) + 1; % 累加点云密度
    end
end

% 调整图像对比度
imageMatrix = mat2gray(imageMatrix); % 将矩阵值归一化到 [0, 1]
%imageMatrix = imadjust(imageMatrix); % 增强对比度

% 接下来，您可以对 grayImg 应用 Otsu 方法进行二值化
binaryImg = imageMatrix > 0;
I = binaryImg;
[centers1,radii1] = imfindcircles(binaryImg,[2 8],'ObjectPolarity','dark','Sensitivity',0.65, 'Method','TwoStage');
%可视化
figure(1)
imshow(I)
viscircles(centers1, radii1,'EdgeColor','c');
title('xy面定位')

% 将检测到的圆心坐标按比例缩放回点云坐标系
scaleX = (xMax - xMin) / imgWidth;
scaleY = (yMax - yMin) / imgHeight;
scaledCenters1(:,1) = centers1(:,1) * scaleX + xMin;
scaledCenters1(:,2) = centers1(:,2) * scaleY + yMin;
xy = scaledCenters1;

% 在点云散点图上可视化圆心
figure(2);
scatter(Location(:,1), Location(:,2), 5, 'c', 'filled');
hold on;
scatter(scaledCenters1(:,1), scaledCenters1(:,2), 10, 'r', 'filled'); % 使用调整后的坐标
hold off;
title('点云中的圆心');

%% %%xy面
xy_1=[];
for i=1:size(xy,1)
    %筛选中心周围一定距离的点（减少运算量）
    d=pdist2(xy(i,:),Location(:,1:2));
    [d,index]=sort(d);
    a=find(d<=100*d(1));
    index=index(a);
    xyz=Location(index,:);
    %计算各点相对中心点的角度
    theta=[];
    for j=1:size(xyz,1)
        theta(j)=atan((xyz(j,2)-xy(i,2))/(xyz(j,1)-xy(istd_dev,1)));
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

%物件是近似平放的，因此xy面提取到的中心点最多，则以xy面找到的中心点为主，yz面进行补充
%xy面和yz面找到的中心有重复的，将不重复的添加进来
xy1=xy_1;

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
    xx = double(xx);
    %拟合三维圆公式（非线性最小二乘法，其中拟合算法是信赖域算法）
    f=@(a,x) (x(:,1)-a(1)).^2+(x(:,2)-a(2)).^2+(x(:,3)-a(3)).^2-a(4)^2;
    x0=[xy2(i,:),mean(pdist2(xy2(i,:),xx))];
    lb=[xy1(i,1:2)-0.001,xy1(i,3)-0.0001,0];
    ub=[xy1(i,1:2)+0.001,xy1(i,3)+0.0001,15];
    option=[];option.Display='off';
    a=lsqcurvefit(f,x0,xx,zeros(size(xx,1),1),lb,ub,option);
    xy2(i,:)=a(1:3);%圆心
    r(i)=a(4);%圆半径
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
    scatter3(X{i,1}(:,1),X{i,1}(:,2),X{i,1}(:,3),10,'g','filled')
end
xlabel('x')
ylabel('y')
zlabel('z')
view(70,60)
title('三维圆心矫正')

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
    fprintf('编号%d：圆心坐标为[%.4f,%.4f,%.4f]，半径为%.4f，圆心法向量为[%.4f,%.4f,%.4f]\n',[i,xy2(i,:),r(i),lamda(i,:)])
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
