%based on the paper 

% addpath('./core/');

 
clc;
close all;
clear
Tac = 80;
Tni = 0.85;
center_size = 30;


%% 相机内外参
KL = [4.0414902467046381e+03 0. 6.1548190490078093e+02; 0. 4.0415568521823084e+03 4.8664531136958760e+02; 0. 0. 1.]; %对应camMatL，注意是3x3矩阵，添加分号;
DL = [-7.1679069182923591e-02 1.9391127929011229e+00 0. 0. 0.];%对应DistL
KR = [4.0291844363113541e+03 0. 6.2515187671350520e+02; 0. 4.0282505132823089e+03 4.7900276705160121e+02; 0. 0. 1.]; %camMatR，注意是3x3矩阵，添加分号;
DR = [-3.1869296798769042e-02 6.5667364171136378e-01 0. 0. 0.];%DistR
R = [ 9.6508536437588155e-01 5.9845304948975614e-03 2.6186718935794068e-01;-7.0550119258210560e-03 9.9997015825930213e-01 3.1479195657845441e-03; -2.6184053596454165e-01 -4.8854872450726566e-03 9.6509878548269856e-01];%R,注意是3x3矩阵，添加分号;
T = [-2.0272692865067700e+02 8.2079312896071888e-01 3.2335975909738771e+01];%T
% 立体标定
cameraParameters1 = cameraParameters("IntrinsicMatrix",KL',"RadialDistortion",[DL(1,1:2) DL(5)],"TangentialDistortion",DL(1,3:4),"ImageSize",[1024,1280]);
cameraParameters2 = cameraParameters("IntrinsicMatrix",KR',"RadialDistortion",[DR(1,1:2) DR(5)],"TangentialDistortion",DR(1,3:4),"ImageSize",[1024,1280]);
stereoParams = stereoParameters(cameraParameters1,cameraParameters2,R',T);



%% 批量处理
left_dir_name = '.\testronghexiao\L'; %左图像目录
right_dir_name = '.\testronghexiao\R'; %右图像目录
left_files = dir(fullfile(left_dir_name,'*.bmp'));
right_files = dir(fullfile(right_dir_name,'*.bmp'));
% 这是保存最后得到的点的路径，现在是左图像的路径，如果需要右图像则将L改为R
left_file_name = '.\result_ronghexiao\L\';
right_file_name = '.\result_ronghexiao\R\';
left_center = [];
right_center = [];

write_1 = 1;
mm = 3; % 代表处理哪张图片，
for z = mm  %: length(files)
    left_filename = fullfile(left_dir_name, left_files(z).name);  % 获取完整的左图文件路径
    right_filename = fullfile(right_dir_name, right_files(z).name);  % 获取完整的右图文件路径
    % disp('read image------------------------------------------------');
%% 读取左右图
    I_left = imread(left_filename);
    I_right = imread(right_filename);
%% 立体校正
    [rectifiedLeft, rectifiedRight] = rectifyStereoImages(I_left, I_right, stereoParams,'OutputView','valid'); %立体校正
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
        files_path = [left_file_name, num2str(z),'.txt'];
        writeCircles(circles_left,files_path)
        left_center = circles_left;
    end
%右图
    figure(11);
    imshow(dispImg_right)
    % 写入文件路径
    if write_1 == 1
        files_path = [right_file_name, num2str(z),'.txt'];
        writeCircles(circles_right,files_path)
        right_center = circles_right;
    end
end
