import msvcrt
import cv2 #读的是BGR不是RGB！
import matplotlib
import numpy
import numpy as np

'''
第一部分：数字模板提取数字
第一步：读入图片
第二步：进行灰度化和二值化处理，这里的二值化使用的cv2.THRESH_BINARY_INV， 将黑色的数字转换为白色
第三步：使用cv2.findContours获得轮廓信息
第四步：对contours根据外接矩阵的x的位置，从左到右进行排序
第五步：遍历contours，使用cv2.boudingRect外接矩形获得轮廓的位置信息，提取数字轮廓的图片，与索引组成轮廓信息的字典
'''

'''
img_num = cv2.imread("card.png")
gray_num = cv2.cvtColor(img_num,cv2.COLOR_BGR2GRAY)
ret,gray_2num = cv2.threshold(gray_num,150,255,cv2.THRESH_BINARY_INV)
contours,hierarchy=cv2.findContours(gray_2num,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
res = cv2.drawContours(img_num,contours,-1,(0,0,255),2)
listx = []
listy = []
n = 0
for i in contours:
    x, y, w, h = cv2.boundingRect(i)
    listx.append(x)
    listy.append(y)
    img = cv2.rectangle(img_num,(x,y),(x+83,y+130),(0,255,0),2)
    n += 1
listx.sort()
listy.sort()
print(listx)
print(listy)
piclist = []
k = 0
for l in listx:
    y = listy[k]
    x = listx[k]
    piclist.append(gray_2num[y:y+130,x:x+83])
    cv2.imshow("img", piclist[k])
    cv2.waitKey(0)#数字为毫秒，0为按键消失
    k += 1
'''

'''
第二部分：对图片进行预处理，提取包含数字信息的4个轮廓的位置信息
第一步：读入图片
第二步：调用函数，扩大图片的面积，并进行灰度化
第三步：使用礼帽tophat 原始图片 - 先腐蚀后膨胀的图片，为了去除背景，使得线条更加的突出
第四步：使用sobel算子cv2.Sobel 找出图片中的边缘信息，即进行图像的梯度运算
第五步：使用闭运算 先膨胀再腐蚀， 将图片上的一些相近的数字进行相连，使得其连成一块
第六步：使用cv2.threshold 将图片进行二值化操作
第七步：再次使用闭运算对图片中的内部缺失的位置再次进行填充,使用不同的卷积核
第八步：重新计算轮廓值，遍历轮廓，根据长宽比和长宽的数值，筛选出符合条件的轮廓的locs，并对locs根据x的大小进行排序
'''
img_data = cv2.imread("post.jpg")

gray_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2GRAY)


kernel = np.ones((3,9),np.uint8)
img1 = cv2.morphologyEx(gray_data,cv2.MORPH_TOPHAT,kernel)

img2 = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=-1)
img2 = np.absolute(img2)
img3 = 255 * (img2 - img2.min()) / (img2.max() - img2.min())
img3 = np.uint8(img3)

img4 = cv2.morphologyEx(img3,cv2.MORPH_CLOSE,kernel)
img5 = cv2.threshold(img4, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
img6 = cv2.morphologyEx(img5, cv2.MORPH_CLOSE, kernel)
contours,hierarchy= cv2.findContours(img6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
res = cv2.drawContours(img_data,contours,-1,(0,0,255),2)
cv2.imshow("img", res)
cv2.waitKey(0)  # 数字为毫秒，0为按键消失
locs = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    acr = int(w / h)
    if acr > 2.5 and acr < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))
print(locs)
res = cv2.drawContours(img_data,locs,-1,(0,0,255),2)
'''
第三部分：遍历每个locs，提取其中的数字，与模板数字做匹配，判断数字属于模板中的哪个数字
第一步：遍历locs，使用loc中的x,y, w, h 获得信用卡中的对应图片
第二步：对图片进行二值化操作
第三步：使用cv2.findContours,找出其中的轮廓，对轮廓进行排序
第四步：循环轮廓，使用外接矩形的位置信息, x1, y1, w1, h1, 获得当前轮廓对应的数字，此时已经获得了需要预测数字的单个图片
第五步：循环数字模板，使用cv2.matchTemplate进行模板匹配，使用cv2.minMaxLoc获得最大的得分值，使用np.argmax输出可能性最大的数字
'''
lx = []
ly = []
n = 0
for i in locs:
    im = gray_data[i[1]:i[1]+i[3],i[0]:i[0]+i[2]]
    im1 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(im1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        lx.append(x)
        ly.append(y)
        print(x,y,w,h)
        n += 1
cv2.imshow("img", res)
cv2.waitKey(0)  # 数字为毫秒，0为按键消失

lx.sort()
ly.sort()
print(lx)
print(ly)
piclist = []
'''