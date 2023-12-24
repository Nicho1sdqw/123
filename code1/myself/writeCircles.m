function boder = writeCircles(circles_new,filename)
%返回边界点坐标
boder=[];
start_angle = 0;  % 起始角度
end_angle = 2 * pi;  % 结束角度
num_point = 60;  % 生成的点的个数
theta = linspace(start_angle, end_angle, num_point);  % 生成一些角度值
fid = fopen( filename,'w');
center_xy = []; % 用于存放x,y坐标
for  m =1: size(circles_new,1)
    circles_new(m,:);
    boder = [boder;circles_new(m,1:2) ];
    fprintf(fid,'[%.4f,%.4f,%.4f]\n',[circles_new(m,:)]); % 第一行保存的是代表的是 圆中心 与 半径 eg：246.1084   70.4799   17.7772
    % 生成角度范围内的点坐标,并同时进行转置操作
    x = (circles_new(m,1) + circles_new(m,3) * cos(theta))';
    y = (circles_new(m,2) + circles_new(m,3) * sin(theta))';
    % 将坐标进行拼接
    center_xy = [x,y];
    for jjjj = 1:size(center_xy,1)
        fprintf(fid,'[%.4f,%.4f]\n',[center_xy(jjjj,1),center_xy(jjjj,2)]);     
     end
            %fprintf(fid,'[%.4f,%.4f]\n',[center_xy(:,1)',center_xy(:,2)']) % 第二行开始 往后 60行  每一行是 每一个圆被取了 60个点，eg:263.8855,263.7848
    hold on

    plot(center_xy(:, 1), center_xy(:, 2), 'r+');
    boder = [boder;center_xy ];
    text(round(circles_new(m, 1)+circles_new(m, 3)+5),round(circles_new(m, 2)+circles_new(m, 3)+5),num2str(m), 'Color', 'red', 'FontSize', 14);
    hold on;
end

disp('写入成功')
fclose(fid);


end
