%based on the paper 

% addpath('./core/');

 
clc;
close all;
clear

%% 相机参数,因为校正已经在c++里完成了，这里只需要三维重建的相关参数
%读取Q矩阵
param = xmlread('校正图、标定\calib_stereo1216_Q.xml');
Qxml=char(param.getElementsByTagName('data').item(10).getFirstChild.getData);
Q=str2double(strsplit(strtrim(Qxml)));
Q=reshape(Q,4,4)';

%读取第一个旋转向量
rxml=char(param.getElementsByTagName('data').item(11).getFirstChild.getData);
r0=str2double(strsplit(strtrim(rxml)));
RE = rotationVectorToMatrix(r0);

txml=char(param.getElementsByTagName('data').item(12).getFirstChild.getData);
TE=str2double(strsplit(strtrim(txml)))';


seq =1; %改这个就行

%% 批量处理
left_img_name = ['.\校正图、标定\L\' num2str(seq) '_rectify.bmp']; %左图像目录
right_img_name =['.\校正图、标定\r\' num2str(seq) '_rectify.bmp']; %右图像目录

% 这是保存最后得到的点的路径，现在是左图像的路径，如果需要右图像则将L改为R
left_file_name = '.\校正图、标定\result\L\';
right_file_name = '.\校正图、标定\result\R\';
left_center = [];
right_center = [];

write_1 = 1;
mm = 1; % 代表处理哪张图片，
for z = mm  %: length(files)

    % disp('read image------------------------------------------------');
%% 读取左右图
    I_left = imread(left_img_name);
    I_right = imread(right_img_name);
%% 立体校正这里就不需要了
    rectifiedLeft = I_left;
    rectifiedRight = I_right;
    %[rectifiedLeft, rectifiedRight] = rectifyStereoImages(I_left, I_right, stereoParams,'OutputView','valid'); %立体校正
    %figure;
    %imshow(stereoAnaglyph(rectifiedLeft, rectifiedRight));
%% 提取圆信息    
   % circles_left = captureCircles(rectifiedLeft, Tac,Tni,center_size); %原来的方法
   % circles_right = captureCircles(rectifiedRight, Tac,Tni,center_size);%原来的方法
   circles_left = captureCircles_Matlab(rectifiedLeft);  %matlab的方法
   circles_right = captureCircles_Matlab(rectifiedRight); %matlab的方法

   dispImg_left = drawCircle(rectifiedLeft,circles_left(:,1:2),circles_left(:,3));     
   dispImg_right = drawCircle(rectifiedRight,circles_right(:,1:2),circles_right(:,3));

%% 画图并写入坐标
%左图
   figure(10);
   imshow(dispImg_left)
    % 写入文件路径
    if write_1 == 1
        files_path = [left_file_name, num2str(seq),'.txt'];
        border_left = writeCircles(circles_left,files_path); %左图圆的边界点
        left_center = circles_left;
    end
%右图
    figure(11);
    imshow(dispImg_right)
    % 写入文件路径
    if write_1 == 1
        files_path = [right_file_name, num2str(seq),'.txt'];
        border_right = writeCircles(circles_right,files_path);%右图圆的边界点
        right_center = circles_right;
    end
end


%% 立体视觉三维重建
%计算视差
disp = border_left(:,1) - border_right(:,1);

%计算三维坐标
xyz = [border_left(:,1)-1 border_left(:,2)-1 disp]'; %构造x y disp
xyz = [xyz;ones(1,size(xyz,2))];%构造x y disp 1
xyz_map = Q*xyz; %相机坐标系下齐次坐标
xyz_camera = xyz_map./xyz_map(4,:);
xyz_camera = xyz_camera(1:3,:); %校正的相机坐标系坐标

%按C++方式进行坐标转换
XYZ_World = RE * (xyz_camera - TE);

%% 比较
file_name = 'point_filter_stereo1';
pcloud= pcread("point_filter_stereo1.pcd");
figure;pcshow(pcloud);
hold on;
plot3(XYZ_World(1,:),XYZ_World(2,:),XYZ_World(3,:),'r.')
save(strcat('./result_mat/', file_name ,'_2d.mat'),"pcloud","XYZ_World");

%写入txt
num = size(XYZ_World,2)/61;
XYZ_World = XYZ_World';
fid = fopen( ".\校正图、标定\result\result.txt",'w');
for i = 1:num
    for j = (i-1)*61+1:i*61
        fprintf(fid,'%.4f,%.4f,%.4f \n',[XYZ_World(j,1),XYZ_World(j,2),XYZ_World(j,3)]);  
    end
    fprintf(fid,'\n');
end


