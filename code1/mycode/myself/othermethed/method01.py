import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread('D://xiaoban//weizhi1//xiaoban8//1zao//4.bmp')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(gray)
_,thres = cv2.threshold(gray, 50,260, cv2.THRESH_TOZERO)
res = cv2.Canny(thres, 100, 200, L2gradient=True)
# res = gray
# plt.imshow(res.astype(np.uint8))

# plt.title('img'), plt.xticks([]), plt.yticks([])
# plt.show()
circle1 = cv2.HoughCircles(res,cv2.HOUGH_GRADIENT,1,10,param1=4,param2=36,minRadius=23,maxRadius=50)
print(circle1)
circles = circle1[0, :, :]  # 提取为二维
circles = np.uint16(np.around(circles))  # 四舍五入，取整
for i in circles[:]:
    print(i[0],',', i[1],',', i[2],',')
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画圆
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画圆心

plt.imshow(img)
plt.title('circle'), plt.xticks([]), plt.yticks([])
plt.show()



# pic= cv2.imread('F://A1//SubExp//mycode//DATA//DaPos1Image//1.bmp')
# gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
# # print(gray)
# _,thres = cv2.threshold(gray, 50,260, cv2.THRESH_TOZERO)
# res = cv2.Canny(thres, 100, 200, L2gradient=True)
# res = gray
# plt.imshow(res.astype(np.uint8))

# plt.title('img'), plt.xticks([]), plt.yticks([])
# plt.show()
# circles = cv2.HoughCircles(res,cv2.HOUGH_GRADIENT,1,10,param1=1,param2=1,minRadius=0,maxRadius=100)
# print(circles)
# for i in circles[0,:]:
    
#     i = i.astype(int)
#     crop = res[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]
#     mask = np.zeros(crop.shape)
#     mask = cv2.circle(mask, (i[2], i[2]), i[2], (255, 255, 255), -1)
#     final_im = mask * crop
#     final_im = final_im.astype(np.uint8)
#     print(final_im )
#     print(final_im.dtype)
#     print(len(final_im) )
#     # if len(final_im) != 0:
#     #     circles2 = cv2.HoughCircles(final_im,cv2.HOUGH_GRADIENT,1,100,param1=1,param2=1,minRadius=0,maxRadius=100)
#     # for j in circles2[0,:]:
#     #     cv2.circle(final_im,(int(j[0]),int(j[1])),int(j[2]),(255,0,0),2)

# cv2.imshow('Hole',final_im)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()