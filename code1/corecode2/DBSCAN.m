function [IDX, isnoise]=DBSCAN(X,epsilon,MinPts)  
%首先定义个函数ExpandCluster
    function ExpandCluster(i,Neighbors,C)  
        IDX(i)=C;  
        k = 1;  
        while true  
            j = Neighbors(k);  
            if ~visited(j)  
                visited(j)=true;  
                Neighbors2=find(D(j,:)<=epsilon);  
                if numel(Neighbors2)>=MinPts  
                %numel函数用于计算数组中满足指定条件的元素个数
                    Neighbors=[Neighbors Neighbors2]; 
                end  
            end  
            if IDX(j)==0  
                IDX(j)=C;  
            end  
            k = k + 1;  
            if k > numel(Neighbors)
            %numel函数用于计算数组中满足指定条件的元素个数
                break;  
            end  
        end  
    end  
    C=0;  %初始化参数
    n=size(X,1);  
    IDX=zeros(n,1);  
    D=pdist2(X,X);  %计算各个点之间的距离
    visited=false(n,1);  %false：创建逻辑矩阵（0和1，0表示真，1表示假）
    isnoise=false(n,1);  
    %% 下面这段程序是每次循环先生成各个小集群，然后在以这些小集群为基础逐步扩大范围
    for i=1:n  
        if ~visited(i)  
            visited(i)=true;  %true相当于0，表示事件正确
            %先定初始集群
            Neighbors=find(D(i,:)<=epsilon);  
            if numel(Neighbors)<MinPts
            %numel函数用于计算数组中满足指定条件的元素个数
                isnoise(i)=true;  
            else  
                C=C+1;  
            %扩大集群
                ExpandCluster(i,Neighbors,C);  
            end  
        end  
    end    
end