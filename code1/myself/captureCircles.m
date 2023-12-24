function circles_new = captureCircles(I,Tac,Tni,center_size)
%将demo2主程序里的提取circles的函数提取出来
    [circles, ~,~] = circleDetectionByArcsupportLS(I, Tac, Tni);
    % circles
    % 根据得到的圆心进行过滤
    % 遍历圆心
    %center_size = 30; % 圆半径的大小(超过这个范围就认为是噪声圆)
    circles_new = [];
    index = [];
    for i = 1 : size(circles,1)
        % 说明这些数据已经是需要消除了的
        if ismember(i, index)
            continue
        end
        storage_circle = [];  % 定义一个存放圆的地方
        % 如果圆的半径超过了center_size,那么就认为这个圆是噪声圆
        if circles(i,3) > center_size
            continue
        end
        % 如果不是噪声圆，我就将这个圆先存放到零时变量storage_circle中
        storage_circle = [storage_circle; circles(i,:)];
        for j = i:size(circles,1)
            if ismember(j, index)
                continue
            end
            dis = sqrt(power((circles(i, 1) - circles(j, 1)),2) + power(circles(i, 2) - circles(j, 2),2));
            diff_r = circles(i,3) - circles(j,3);
            % 如果是同一个圆的话，就跳过保存
            if dis == 0 && diff_r == 0
                continue
            % 如果相等，那么说明是同一个点，此时直接跳过
            elseif (dis >= 0 && dis < 20)  % 如果圆心之间的差距在范围:[0,20)内，那么就说明这两个圆可能是同一个圆
                % 那么我就将这个圆保存下来
                storage_circle = [storage_circle;circles(j,:)];
                % 并且此时保存的数据的索引进行保存，方便后续的消除
                index = [index,j];
            end
        end
        % 根据上面保存下来的数据来计算最后的平均值
        % 按照列的方式来求均值,并且将求出来的均值存放到最后的结果中
        if size(storage_circle,1) == 1
            % 表示只有一个样本，此时就不用求均值
            circles_new = [circles_new; storage_circle];
        else
            % 表示有多个样本，此时就需要求均值
            %circles_new = [circles_new; mean(storage_circle)];
            x = mean(storage_circle(:,1));
            y = mean(storage_circle(:,2));
            r_min = min(storage_circle(:,3));
            r_sum = sum(storage_circle(:,3));
            r = 0;
            size_storage = size(storage_circle,1);
            for m = 1:size_storage
                if storage_circle(m,3) == r_min
                    r = r + 0.8*storage_circle(m,3);
                else
                    r = r + (size_storage-1)/size_storage * 0.2 * storage_circle(m,3);
                end
                    
            end
            circles_new = [circles_new; [x,y,r]];
        end
    end
end