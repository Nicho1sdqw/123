import cv2
import numpy as np

# 读取图像
image = cv2.imread('D://daban//weizhi2//yuantu//1.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 预处理图像：应用高斯模糊和Canny边缘检测
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# 轮廓检测
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 寻找圆形轮廓
circle_contours = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(approx) >= 3:  # 圆形轮廓通常有6个以上的顶点
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity > 0.2:  # 根据圆形度进行筛选
            circle_contours.append(contour)

# 获取圆心和半径
circles = []
centerold = 0
for contour in circle_contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    circles.append((center, radius))

# ccircles = circles
# newcircles = []
# # circles = np.unique(circles)
# for eee in circles:
#     ab = eee
#     for i in range(0,len(circles)):
#         # print(ab[0][0], abbb[0][0])
#         if ab[0][0] != circles[i][0]:
#             newcircles.append(ab)

# print(newcircles)    
# print(circles)

# 绘制圆形轮廓和标记圆心-> 在这里提取圆心和半径
for (center, radius) in circles:
    print(center[0],',', center[1],',',radius,',')
    cv2.circle(image, center, radius, (0, 255, 0), 1)
    cv2.circle(image, center, 2, (0, 0, 255), 1)

# 显示结果图像
cv2.imshow('Circle Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()