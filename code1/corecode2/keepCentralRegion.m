function centralRegionImg = keepCentralRegion(bwImg)
    % 反转图像，因为连通区域函数在像素值为 1 时检测连通区域
    bwImg = ~bwImg;

    % 查找连通区域
    connComp = bwconncomp(bwImg);

    % 获取每个连通区域的统计信息
    stats = regionprops(connComp, 'Centroid', 'BoundingBox');

    % 图像中心
    imgCenter = size(bwImg) / 2;

    % 初始化最小距离为一个很大的数
    minDist = inf;
    centralRegionIdx = 0;

    % 找到最接近中心的区域
    for i = 1:numel(stats)
        centroid = stats(i).Centroid;
        distToCenter = norm(centroid - imgCenter);
        if distToCenter < minDist
            minDist = distToCenter;
            centralRegionIdx = i;
        end
    end

    % 创建一个新的图像，保留最接近中心的连通区域
    centralRegionImg = false(size(bwImg));
    centralRegionImg(connComp.PixelIdxList{centralRegionIdx}) = true;

    % 反转图像回到原始状态（连通区域为 0）
    centralRegionImg = ~centralRegionImg;
end
