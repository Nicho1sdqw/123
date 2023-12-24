%% based on the paper 

% addpath('./core/');

% filename = 'E:/MATLAB程序/边缘检测/大板位置1图像(1)/大板位置1图像/5.bmp'; 
% filename = '1.bmp'; 
% parameters
clear allimage
clc
Tac = 168;
Tni = 0.6;

% read image
% disp('read image------------------------------------------------');
% I = imread(filename);
% figure;imshow(I);

dir_name = 'D://myself//testronghexiao//R//';
files = dir(fullfile(dir_name,'1.bmp'));
for z = 1 : length(files)
    disp(z)
    filename = fullfile(dir_name, files(z).name);
    fname = filename;
    disp('read image------------------------------------------------');
    I = imread(fname);
    noise = 0.01;
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
% 为每个圆添加编号
% for i = 1:size(circles_new,1)
%    num_str = num2str(i);
%     x = circles_new(i,1);
%    y = circles_new(i,2);
%     r = circles_new(i,3);
%     theta = rand()*2*pi; % 随机生成编号的角度
%     offset = [r*cos(theta), r*sin(theta)]; % 计算偏移量
%     text_str = num_str; % 不再加上 "No." 前缀
%     I = insertText(I,[x,y] + offset,text_str,'FontSize',20,'AnchorPoint','Center', 'BoxColor', 'white', 'TextColor', 'red'); % 修改为白色背景和红色字体，字体大小改回原来的值
% end

    disp(['number of circles：',num2str(size(circles_new,1))]);
    disp('draw circles----------------------------------------------');
    dispImg_new = drawCircle(I,circles_new(:,1:2),circles_new(:,3));
    dispImg = drawCircle(I,circles(:,1:2),circles(:,3));
    writematrix(circles_new,'myData.dat','Delimiter',',')  

    % 打印输出数据时带上编号
    %for i = 1: size(circles_new, 1)
        %fprintf('%d: %f %f %f\n',i,circles_new(i,1),circles_new(i,2),circles_new(i,3))
    %end

    % 显示图片和圈
    figure;
    imshow(dispImg_new);
end