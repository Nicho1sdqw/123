import numpy as np
import json
import scipy.io as scio
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
center = []
redis_sum = 0
count_redis = 0

# %%%%%%%%%%%%%%%%%%%%%%%%main

# # print(center)
# predict_centers = []
# # 这里是
# boudpoint = []
# LS_file_path = "C:/Users/gxxu/Downloads/corecode/result_circle/dapoint_filter_stereo1-16result_LS.txt"
# with open(LS_file_path, "r") as lm_f:
#     for line in lm_f:
#         line = line.rstrip()
#         # 如果不是分隔符，那么就开始将数据存储起来
#         # 表示是当前的pcd文件，因此读取对应的pcd文件
#         if line[0] != '[':
#             # 读取文件
#             json_file = "C:/Users/gxxu/Downloads/corecode/point_filter_stereo6_1.json"
#             with open(json_file, 'r') as f:
#                 result = json.load(f)
#                 for key, values in result.items():
#                     if key == 'center':
#                         center = values
#                         center = np.array(center)
#                     else:  # 开始计算半径
#                         values_r = np.array(values)
#                         boudpoint.append(values)
#
#                         redis_sum += np.sqrt(np.sum((center - values_r) ** 2))
#                         count_redis += 1
#                 # 得到半径
#                 redis = redis_sum / count_redis
#         else:
#             predict_centers.append(eval(line))
#             print(predict_centers)
#             # 转换成numpy类型
#             predict_centers = np.array(predict_centers)
#             distance = float('inf')
#             index_ = 0
#             for i, predict_center in enumerate(predict_centers):
#                 diff = np.sqrt(np.sum((predict_center[:3] - center) ** 2))
#                 if diff < distance:
#                     distance = diff
#                     index_ = i
#
#             print("ad_c", distance)
#             print("ad_r", np.abs(redis - predict_centers[index_][3]))
#
#             # print('bound point',boudpoint)
#             # print(len(boudpoint))
#
#             mat_file = "C:/Users/gxxu/Downloads/corecode/result_circle/point_filter_stereo6_1result_LS_boundPoint.mat"
#             bpoint_LS = scio.loadmat(mat_file)
#             bpoint_LS_data = bpoint_LS['P_fitcircle']
#
#             # print(len(bpoint_LS_data))
#             # print('Get LS data',bpoint_LS_data)
#             boudpoint = np.array(boudpoint)
#             distances = cdist(boudpoint, bpoint_LS_data)
#
#             closest_indices = np.argmin(distances, axis=0)
#
#             # print(closest_indices)
#             closest_points = boudpoint[closest_indices]
#             distances = np.sqrt(np.sum((closest_points - bpoint_LS_data)**2, axis=1))
#
#             # 计算平均距离误差
#             average_distance_error = np.mean(distances)
#
#             print(f"LS平均距离误差为: {average_distance_error}")



# %%%%%%%%%%%%%%%%%%%%%%%%% LM

# print(center)
predict_centers = []
boudpoint = []
# 这里是
LM_file_path = "D:/corecode2/result_circle/point_filtered_mono4-2result_LM.txt"
with open(LM_file_path, "r") as lm_f:
    for line in lm_f:
        line = line.rstrip()
        # 如果不是分隔符，那么就开始将数据存储起来
        # 表示是当前的pcd文件，因此读取对应的pcd文件
        if line[0] != '[':
            # 读取文件
            json_file = "D:/corecode2/data/point_filtered_mono4-2.json"
            with open(json_file, 'r') as f:
                result = json.load(f)
                for key, values in result.items():
                    if key == 'center':
                        center = values
                        center = np.array(center)
                    else:  # 开始计算半径
                        values_r = np.array(values)
                        boudpoint.append(values)
                        redis_sum += np.sqrt(np.sum((center - values_r) ** 2))
                        count_redis += 1
                # 得到半径
                redis = redis_sum / count_redis
        else:
            predict_centers.append(eval(line))
            print(predict_centers)
            # 转换成numpy类型
            predict_centers = np.array(predict_centers)
            distance = float('inf')
            index_ = 0
            for i, predict_center in enumerate(predict_centers):
                diff = np.sqrt(np.sum((predict_center[:3] - center) ** 2))
                if diff < distance:
                    distance = diff
                    index_ = i

            print("ad_c", distance)
            print("ad_r", np.abs(redis - predict_centers[index_][3]))
            
            # print('bound point',boudpoint)
            # print(len(boudpoint))

            mat_file = "D:/corecode2/result_circle/point_filtered_mono4-2result_LM_boundPoint.mat"
            bpoint_LM = scio.loadmat(mat_file)
            bpoint_LM_data = bpoint_LM['P_fitcircle']
 
            # print(len(bpoint_LS_data))
            # print('Get LM data',bpoint_LS_data)
            boudpoint = np.array(boudpoint)
            distances = cdist(boudpoint, bpoint_LM_data)
            # print(len(distances))
            closest_indices = np.argmin(distances, axis=0)

            # print(closest_indices)
            closest_points = boudpoint[closest_indices]
            distances = np.sqrt(np.sum((closest_points - bpoint_LM_data)**2, axis=1))

            # 计算平均距离误差
            average_distance_error = np.mean(distances)

            print(f"LM平均距离误差为: {average_distance_error}")



# # %%%%%%%%%%%%%%%%%%%% Pratt
# center = []
# redis_sum = 0
# count_redis = 0
# # print(center)
# predict_centers = []
# boudpoint = []
# Pratt_file_path = "C:/Users/gxxu/Downloads/corecode/result_circle/point_filter_stereo6_1result_Pratt.txt"
# with open(Pratt_file_path, "r") as lm_f:
#     for line in lm_f:
#         line = line.rstrip()
#         # 如果不是分隔符，那么就开始将数据存储起来
#         # 表示是当前的pcd文件，因此读取对应的pcd文件
#         if line[0] != '[':
#             # 读取文件
#             json_file = "C:/Users/gxxu/Downloads/corecode/point_filter_stereo6_1.json"
#             with open(json_file, 'r') as f:
#                 result = json.load(f)
#                 for key, values in result.items():
#                     if key == 'center':
#                         center = values
#                         center = np.array(center)
#                     else:  # 开始计算半径
#                         values_r = np.array(values)
#                         boudpoint.append(values)
#                         redis_sum += np.sqrt(np.sum((center - values_r) ** 2))
#                         count_redis += 1
#                 # 得到半径
#                 redis = redis_sum / count_redis
#         else:
#             predict_centers.append(eval(line))
#             print(predict_centers)
#             # 转换成numpy类型
#             predict_centers = np.array(predict_centers)
#             distance = float('inf')
#             index_ = 0
#             for i, predict_center in enumerate(predict_centers):
#                 diff = np.sqrt(np.sum((predict_center[:3] - center) ** 2))
#                 if diff < distance:
#                     distance = diff
#                     index_ = i
#
#             print("ad_c", distance)
#             print("ad_r", np.abs(redis - predict_centers[index_][3]))
#             mat_file = "C:/Users/gxxu/Downloads/corecode/result_circle/point_filter_stereo6_1result_Pratt_boundPoint.mat"
#             bpoint_Pratt = scio.loadmat(mat_file)
#             bpoint_Pratt_data = bpoint_Pratt['P_fitcircle']
#
#             # print(len(bpoint_LS_data))
#             # print('Get LM data',bpoint_LS_data)
#             boudpoint = np.array(boudpoint)
#             distances = cdist(boudpoint, bpoint_Pratt_data)
#             # print(distances)
#             closest_indices = np.argmin(distances, axis=0)
#
#             # print(closest_indices)
#             closest_points = boudpoint[closest_indices]
#             distances = np.sqrt(np.sum((closest_points - bpoint_Pratt_data)**2, axis=1))
#
#             # 计算平均距离误差
#             average_distance_error = np.mean(distances)
#
#             print(f"Pratt平均距离误差为: {average_distance_error}")
#
#
# # %%%%%%%%%%%%%%%%%%%% ransac
# center = []
# redis_sum = 0
# count_redis = 0
# # print(center)
# predict_centers = []
# boudpoint = []
# Pratt_file_path = "C:/Users/gxxu/Downloads/corecode/result_circle/point_filter_stereo6_1result_RANSAC.txt"
# with open(Pratt_file_path, "r") as lm_f:
#     for line in lm_f:
#         line = line.rstrip()
#         # 如果不是分隔符，那么就开始将数据存储起来
#         # 表示是当前的pcd文件，因此读取对应的pcd文件
#         if line[0] != '[':
#             # 读取文件
#             json_file = "C:/Users/gxxu/Downloads/corecode/point_filter_stereo6_1.json"
#             with open(json_file, 'r') as f:
#                 result = json.load(f)
#                 for key, values in result.items():
#                     if key == 'center':
#                         center = values
#                         center = np.array(center)
#                     else:  # 开始计算半径
#                         values_r = np.array(values)
#                         boudpoint.append(values)
#                         redis_sum += np.sqrt(np.sum((center - values_r) ** 2))
#                         count_redis += 1
#                 # 得到半径
#                 redis = redis_sum / count_redis
#         else:
#             predict_centers.append(eval(line))
#             print(predict_centers)
#             # 转换成numpy类型
#             predict_centers = np.array(predict_centers)
#             distance = float('inf')
#             index_ = 0
#             for i, predict_center in enumerate(predict_centers):
#                 diff = np.sqrt(np.sum((predict_center[:3] - center) ** 2))
#                 if diff < distance:
#                     distance = diff
#                     index_ = i
#
#             print("ad_c", distance)
#             print("ad_r", np.abs(redis - predict_centers[index_][3]))
#             mat_file = "C:/Users/gxxu/Downloads/corecode/result_circle/point_filter_stereo6_1result_RANSAC_boundPoint.mat"
#             bpoint_Pratt = scio.loadmat(mat_file)
#             bpoint_Pratt_data = bpoint_Pratt['P_fitcircle']
#
#             # print(len(bpoint_LS_data))
#             # print('Get LM data',bpoint_LS_data)
#             boudpoint = np.array(boudpoint)
#             distances = cdist(boudpoint, bpoint_Pratt_data)
#             # print(distances)
#             closest_indices = np.argmin(distances, axis=0)
#
#             # print(closest_indices)
#             closest_points = boudpoint[closest_indices]
#             distances = np.sqrt(np.sum((closest_points - bpoint_Pratt_data)**2, axis=1))
#
#             # 计算平均距离误差
#             average_distance_error = np.mean(distances)
#
#             print(f"RANSAC平均距离误差为: {average_distance_error}")