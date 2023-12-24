%% based on the paper 

addpath('./core/');
filename = 'D://mycode//myself//1.bmp';


% parameters
Tac = 168;
Tni = 0.1;

% read image
disp('read image------------------------------------------------');
I = imread(filename);

figure;imshow(I);
% circle detection
disp('circle detetion-------------------------------------------');
[circles, ~,~] = circleDetectionByArcsupportLOOS(I, Tac, Tni);
% display
disp('show------------------------------------------------------');

% 给每一个圆周添加编号
for i = 1:size(circles,1)
    num_str = num2str(i);
    x = circles(i,1);
    y = circles(i,2);
    r = circles(i,3);
    theta = rand()*2*pi; % 随机生成编号的角度
    offset = [r*cos(theta), r*sin(theta)]; % 计算偏移量
    text_str = num_str; % 不再加上 "No." 前缀
    I = insertText(I,[x,y] + offset,text_str,'FontSize',20,'AnchorPoint','Center', 'BoxColor', 'white', 'TextColor', 'red'); % 修改为白色背景和红色字体，字体大小改回原来的值
end

% circles
disp(['number of circles：',num2str(size(circles,1))]);
disp('draw circles----------------------------------------------');
dispImg = drawCircle(I,circles(:,1:2),circles(:,3));

writematrix(circles,'myData.dat','Delimiter',',')  
type myData.dat

% 打印输出数据时带上编号
for i = 1: size(circles, 1)
    fprintf('%d: %f %f %f\n',i,circles(i,1),circles(i,2),circles(i,3))
end

figure;
imshow(dispImg);