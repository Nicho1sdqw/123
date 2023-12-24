function circles_new = captureCircles_Matlab(I)
%直接用matlab的方法提取圆
circles=[];
circles_new=[];
gray = im2bw(I,0.2); %二值化
%figure;imshow(gray);
[centers, radii, metric] = imfindcircles(~gray,[10 300]) ;%用matlab的函数提取圆 
% figure;
% imshow(I);
% hold on;
% viscircles(centers, radii, 'EdgeColor', 'b');
% title('Detected Circles');
for i = 1:size(centers,1)
    circles = [circles;centers(i,:) radii(i)];

circles_new = circles;
circles_new = sortrows(circles_new,1);
circles_new = sortrows(circles_new,2);
end

