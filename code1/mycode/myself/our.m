%% based on the paper 

% addpath('./core/');

% filename = 'E:/MATLAB程序/边缘检测/大板位置1图像(1)/大板位置1图像/5.bmp'; 
% filename = '1.bmp'; 
% parameters
clear all
clc
Tac = 138;
Tni = 0.35;
% read image
% disp('read image------------------------------------------------');
% I = imread(filename);
% figure;imshow(I);

dir_name = 'D://jianceshujuxiugai//';
files = dir(fullfile(dir_name,'r.bmp'));
for z = 1 : length(files)
    disp(z)
    filename = fullfile(dir_name, files(z).name);  % 获取完整的文件路径
    fname = filename;
    disp('read image------------------------------------------------');
    I = imread(fname);
    noise = 0.001
    I = imnoise(I, 'gaussian',0, noise);
    % circle detections
    disp('circle detetion-------------------------------------------');
    [circles, ~,~] = circleDetectionByArcsupportLS(I, Tac, Tni);
    % display
    disp('show------------------------------------------------------');
    circles;
    % 存放的时满足条件的圆心和半径
    index = [];
    circles_new = [];
    circles_ori = [];
    % 如果两个索引在一定范围内
    for i = 1:size(circles, 1)
        if i == size(circles, 1)
            circles_ori = [circles_ori; circles(i,:)];
        else
            for j = i + 1 : size(circles, 1)
                dis = sqrt(power((circles(i, 1) - circles(j, 1)),2) + power(circles(i, 2) - circles(j, 2),2));
                diff_r = circles(i, 3) - circles(j, 3);
                if (dis >= 0 && dis < 30)
                    % 分别得到x,y的坐标和r的长度
                    % 存放满足条件时候的索引
                    index = [index,i,j];
                    x = (circles(i, 1) + circles(j, 1))/2;
                    y = (circles(i, 2) + circles(j, 2))/2;
                    % r = (circles(i, 3) + circles(j, 3))/2;
                    if diff_r >0
                        if abs(diff_r) > 15
                            r = circles(j, 3);
                        else
                            r = circles(i, 3)*0.1 + circles(j, 3)*0.9;
                        end
                    else
                        if abs(diff_r) > 20
                            r = circles(i, 3);
                        else
                            r = circles(i, 3)*0.9 + circles(j, 3)*0.1;
                        end
                    end
                    cir = [x,y,r];
                    circles_new = [circles_new; cir(:,:)];
                else
                    if j == size(circles, 1)
                        circles_ori = [circles_ori; circles(i,:)];
                    else
                        break;
                    end
                end
            end
        end
    end

    for i = 1: size(circles, 1)
        if isempty(index)
            circles_new = circles;
            break;
        end
        if ~ismember(index, i) 
            if circles(i,3) < 45
                circles_new = [circles_new; circles(i,:)];
            end
        end
    end
    % circles_new
%     [m, n] = size(circles_new);
% 
%     % 设置输出格式
%     formatSpec = repmat('%d,', 1, n-1);
%     formatSpec = [formatSpec, '%d\n'];
%     fprintf(formatSpec, circles_new');
    % disp(['number of circles：',num2str(size(circles,1))]);
    % disp('draw circles----------------------------------------------');
    % dispImg = drawCircle(I,circles(:,1:2),circles(:,3));
    disp(['number of circles：',num2str(size(circles_new,1))]);
    disp('draw circles----------------------------------------------');
    dispImg_new = drawCircle(I,circles_new(:,1:2),circles_new(:,3));
    dispImg = drawCircle(I,circles(:,1:2),circles(:,3));
    writematrix(circles_new,'myData.dat','Delimiter',',')  
    type myData.dat
    % path1 = ['pic_all/output0',num2str(i),'.bmp'];
    % if noise==0.00001
    %     path = ['E:\MATLAB程序\边缘检测\mycode\小板结果图\结果图像31\1噪声/',num2str(z),'.bmp'];
    %     imwrite(dispImg_new, path, 'BMP');
    %     disp('1写入成功')
    % elseif noise==0.00005
    %     path = ['E:\MATLAB程序\边缘检测\mycode\小板结果图\结果图像31\5噪声/',num2str(z),'.bmp'];
    %     imwrite(dispImg_new, path, 'BMP');
    %     disp('5写入成功')
    % elseif noise==0.0001
    %     path = ['E:\MATLAB程序\边缘检测\mycode\小板结果图\结果图像31\10噪声/',num2str(z),'.bmp'];
    %     imwrite(dispImg_new, path, 'BMP');
    %     disp('10写入成功')
    % end%% 画图
    % figure;
        %%%%
    % imshow(dispImg)
    figure;
    imshow(dispImg_new);
    % figure(4);
    % imshow(dispImg_new)
end
%%  

% [ve1, ~] = EES_linear(X_tilde', gt_threshold);
% [inliers1, outliers1] = cut(X_tilde, ve1, gt_threshold);
% inliers1 = X_tilde(:,inliers1);
% outliers1 = X_tilde(:,outliers1);