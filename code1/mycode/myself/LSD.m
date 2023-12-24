%% based on the paper 

addpath('./core/');
filename = 'D://xiaoban//weizhi3//xiaoban5/yuantu//5.bmp';


% parameterss
Tac = 168;
Tni = 0.2;

% read image
disp('read image------------------------------------------------');
I = imread(filename);


%figure;imshow(I);
% circle detection
disp('circle detetion-------------------------------------------');
[circles, ~,~] = circleDetectionByArcsupportLOOS(I, Tac, Tni);
% display
disp('show------------------------------------------------------');
% circles
disp(['number of circles£º',num2str(size(circles,1))]);
disp('draw circles----------------------------------------------');
dispImg = drawCircle(I,circles(:,1:2),circles(:,3));
writematrix(circles,'myData.dat','Delimiter',',')  
type myData.dat
figure;
imshow(dispImg);




%%  

% [ve1, ~] = EES_linear(X_tilde', gt_threshold);
% [inliers1, outliers1] = cut(X_tilde, ve1, gt_threshold);
% inliers1 = X_tilde(:,inliers1);
% outliers1 = X_tilde(:,outliers1);