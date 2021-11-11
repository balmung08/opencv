import msvcrt
import cv2 #读的是BGR不是RGB！
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np

'''
#用摄像头读取
cap = cv2.VideoCapture(1)

#读图片
#以默认方式读取图像，彩色，1,0，-1：彩色，灰色，包括alpha通道
img0 = cv2.imread("1.jpg",1)
size = img0.shape   #高/宽/颜色组
print(size)
cv2.imshow("img0",img0)
cv2.waitKey(0)#数字为毫秒，0为按键消失

#切图片
im = img0[0:200,0:200]
cv2.imshow("im",im)
cv2.waitKey(0)#数字为毫秒，0为按键消失

#提取像素
print(img[100,100])
#只返回单通道的值
print(img[100,100,0])
#修改像素点的值
img[100,100] = (255,255,3)

#切分rgb数据
b,g,r = cv2.split(img0)
print(b)
b_shape = b.shape
print(b_shape)
imgc = img0.copy()
imgc[:,:,1]=0
imgc[:,:,2]=0#只保留b通道
cv2.imshow("imgc",imgc)
cv2.waitKey(0)#数字为毫秒，0为按键消失

#改成灰度图存放
img1 = cv2.imread("7.jpg",cv2.IMREAD_GRAYSCALE)#读成灰度图
size = img1.shape   #高/宽/颜色组（灰度没有此值）
print(size)
cv2.imshow("img1",img1)
cv2.waitKey(0)#数字为毫秒，0为按键消失
cv2.imwrite("darksouls.png",img1)#保存

#转换图片格式
gray = cv.cvtColor(blood, cv.COLOR_BGR2GRAY)#cv.COLOR_BGR2GRAY是将彩图转化为灰度图
hsv = cv.cvtColor(blood, cv.COLOR_BGR2HSV)#cv.COLOR_BGR2HSV是将彩图转化为HSV格式图

#读视频
video = cv2.VideoCapture("test.mp4")
if video.isOpened():
    open,frame = video.read()
else:
    open = (False)
while open:
    sta,frame = video.read()
    if frame is None:
        break
    if sta == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("result",gray)
        if cv2.waitKey(10)&0xFF==27:
            break
video.release()
cv2.destroyAllWindows()

#自定义视频输出
cap = cv2.VideoCapture("test.mp4")
width = int(cap.get(3))#获取视频宽度
height = int(cap.get(4))#获取视频高度
#输出地址，输出视频格式，帧数，输出视频大小
out = cv.VideoWriter("python_venv_test/pyVenvTest/video/friend.avi",cv.VideoWriter_fourcc("M", "J", "P", "G"),60,(width,height))
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True:
        out.write(frame)#将每一帧图片输出
    else:
        break
cap.release()
out.release()
cv.destroyAllWindows()





#边界填充
top_size,bottom_size,left_size,right_size = (50,50,50,50)
method = {cv2.BORDER_REPLICATE,#复制最边上的像素填充
          cv2.BORDER_REFLECT,#反射法
          cv2.BORDER_REFLECT_101,#另一种反射法
          cv2.BORDER_WRAP，#外包装法
cv2.BORDER_CONSTANT#常数填充，须在最后加入value=______选定颜色
}
new_img=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,method)

#数值计算
img2 = img1 + 10   #给img1的所有元素都加10
#相同shape的图片才可以直接相加！拿resize先改一下
img3  = img1 + img2       #加出来的结果超过255的减去256
img3 = cv2.add(img1,img2) #加出来的结果超过255的等于255
img.shape #读取图像像素
img1 = cv2.resize(img1,(500,400))#重设尺寸
img = cv2.addWeighted(img1,0.4,img2,0.6,0)#分权相加后加常数b实现图像叠加 

#图像阈值
ret,dst = cv2.threshold(src,thresh,maxval,type)
#ret是函数格式，dst是输出图，src是输入图，只能为灰度图
#thresh为阈值，maxval为超过阈值以后的赋值
#cv2.THRESH_BINARY  超过阈值取maxval，否则取0
#cv2.THRESH_BINARY_INV 上一种的反转
#cv2.THRESH_TRUNC   大于阈值设为阈值，小于阈值不变
#cv2.THRESH_TOZERO  大于阈值不变，小于阈值变为0
#函数后加_INV表示反转

#图像平滑处理
blur = cv2.blur(img,(3,3))#均值滤波，使每个像素点都是周围3*3矩阵的均值
box1 = cv2.boxFilter(img,-1,(3,3),normalize=True) #方框滤波，-1不用改，normalize写ture时效果同均值滤波
#True时矩阵和除以9，False是不除以9，会出现大量255的白点
gaussian=cv2.GaussianBlur(img,(5,5),1)#高斯滤波，构造了权重矩阵，越近的权重越大
median=cv2.medianBlur(img,5)#中值滤波，n*n矩阵内的中位数作为结果
#中值滤波处理噪音点效果最好

#将图片拼在一起，可以一起输出
img = np.hstack(img1,img2,img3)#横着
img = np.vstack(img1,img2,img3)#竖着

#腐蚀操作 操作前先二值化！
kernel = np.ones((5,5),np.uint8)#n*n矩阵范围内有黑色，白点就变黑
img1 = cv2.erode(img,kernel,iterations=1) #iterations控制操作执行次数

#膨胀操作 操作前先二值化！ 是腐蚀操作的逆操作
 #n*n矩阵范围内有白色，黑点就变白
img1 = cv2.dilate(img,kernel,iterations=1) #操作同上

#开运算：先腐蚀再膨胀，使图形主体（白）不变，消掉毛刺（白），背景黑
kernel = np.ones((5,5),np.uint8)
img1 = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

#闭运算：先膨胀再腐蚀，毛刺不变但主体轮廓变胖
kernel = np.ones((5,5),np.uint8)
img1 = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

#梯度运算：膨胀一次图-腐蚀一次图，可以得到一个边框
kernel = np.ones((5,5),np.uint8)
img1 = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)

#礼帽：原始输入减开运算结果   剩下毛刺
kernel = np.ones((5,5),np.uint8)
img1 = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)

#黑帽：闭运算结果减原始输入    剩下主体轮廓
kernel = np.ones((5,5),np.uint8)
img1 = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)





#提取边缘
#sobel算子
[-1  0  1
 -2  0  2
 -1  0  1]
 #x向，y向顺时针旋转九十度
img1 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
#cv2.CV_64F是限制通道数，这么写就行，x,y,矩阵阶数
#scharr算子
#-1变为-3，-2变为-10，敏感程度更高
img1 = cv2.Scharr(img,cv2.CV_64F,1,0,ksize=3)
#拉普拉斯算子
[0  1   0
 1  -4  0
 0  1   0]
 #不分x,y
img1 = cv2.Laplacian(img,cv2.CV_64F)
#将两个方向都转化成unit8格式
absx = cv.convertScaleAbs(x)
absy = cv.convertScaleAbs(y)
#最后合并
res = cv.addWeighted(absx, 0.5, absy, 0.5, 0)

#canny边缘检测
#流程：对图像高斯滤波，sobel算子提取梯度，非极大值抑制，双阈值检定
#非极大值抑制：不是最大梯度的滤去
#双阈值：大于maxval的认定为边缘，小于maxval大于minval且与边缘相连的像素认定为边缘
img1 = cv2.Canny(img,100,200)#min/maxval

#高斯金字塔
img = cv2.pyrUp(img)#放大
img = cv2.pyrDown(img)#缩小
#放大缩小都会失真，所以先放大后缩小不是回原样而是更加失真

#轮廓检测 最好先二值化！
binary,contours,hierarchy=cv2.findContours(img,mode,method)
#binary原图 contours边缘的结构组 hierarchy层级
#mode:RETR_TREE重新构筑所有轮廓，返回一个数组
#method:CHAIN_APPROX_NONE直接存   CHAIN_APPROX_SIMPLE压缩一下
#绘制轮廓
img = img1.copy()#复制副本
cnt0 = contours[0]
res = cv2.drawContours(img,[cnt0],0,(0,0,255),2)
#变量为 原图 边缘数据 轮廓索引（即序号，写-1直接画所有轮廓） 颜色bgr 线条厚度
#轮廓特征
cnt0 = contours[0]
S = cv2.contourArea(cnt0)#面积
L = cv2.arcLength(cnt0，True)#周长/T表示闭合

#轮廓近似
alpha = 0.05 * cv2.arcLength(cnt0，True)
approx = cv2.approxPolyDP(cnt,alpha,True)#距离线在alpha以上的点略去/闭合
res = cv2.drawContours(img,[approx],0,(0,0,255),2)

#绘制图形
cv.line(img, (0,0), (511,511), (255,0,0),5)#直线line(画在哪里(背景)，起始位置，终点位置，颜色(b(蓝),g(绿),r(红)),线条宽度)
cv.circle(img, (256,256), 60, (0,0,255),-1)#圆形cricle(背景，圆心位置，半径，颜色，线条宽度(-1代表填充))
cv.rectangle(img, (100,100), (400,400), (0,255,0),5)#矩形rectangle(背景，左上角位置，右下角位置，颜色，线条宽度)

#写文字
cv.putText(img, "hello", (50,250), cv.FONT_HERSHEY_COMPLEX, 5, (255,255,255),3)
#文字putText(背景，文本，文本框左下角位置，字体，字号大小，颜色，线条宽度，线型)

#轮廓外接图形
cnt0 = contours[0]
#轮廓矩形
x,y,w,h = cv2.boundingRect(cnt0)
img = cv2.rectangle(img0,(x,y),(x+w,y+h),(0,255,0),2)
#轮廓⚪
(x,y),radius = cv2.minEnclosingCircle(cnt0)
center = (int(x),int(y))
radius = int(radius)
img = cv.circle(img,center,radius,(0,255,0),2)

#模板匹配
method= {cv2.TTM_CCOEFF_NORMED}#归一化相关系数，越接近于1越相关
res = cv2.matchTemplate(img,template,cv2.TTM_CCOEFF_NORMED)#原图+要匹配的小图
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)#相关最小值/最大值及最小最大值的坐标位置

#霍夫线检测(霍夫变换)—在canny边缘提取的基础上检测直线
lines = cv.HoughLines(edg, 0.8, np.pi/180, 130)#参数，原图，rho的单位精度，θ的单位精度，阈值
#绘制直线
for line in lines:
    print(line)
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho*a
    y0 = rho*b
    #再找两个点，绘制直线，使其尽量沾满整个图像
    """https://www.it610.com/article/1291807161583738880.htm
    详细讲解，为何乘以1000"""
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*a)
    cv.line(detect_line, (x1,y1), (x2,y2), (0,255,0))
#霍夫圆检测—用中值滤波去噪，因为霍夫圆检测对噪声比较敏感且只能做灰度图
cricle_d = cv.HoughCircles(cricle_s, cv.HOUGH_GRADIENT, 1,200,param1 = 100,param2 = 50,minRadius = 0,maxRadius = 200)
#绘制圆
for i in cricle_d[0]:
    print(i)
    cv.circle(cricle, (int(i[0]),int(i[1])), int(i[2]), (0,255,0),3)#圆
    cv.circle(cricle, (int(i[0]),int(i[1])), 2, (0,0,255),-1)#圆心

#图像按像素转换为直方图
hist = cv2.calcHist([img],[0],None,[256],[0,256])#图像/颜色通道（灰度0+bgr012）/mask，用于切图像/直方图横坐标个数/颜色范围
plt.hist(img.ravel(),256)#转换过程
plt.show()
#mask创建与使用
mask = np.zeros(img.shape[:2],np.uint8)#按输入图像shape构造一个二维区域
mask[100:300,100:400] = 1#指定区域为1，其他区域为0、
masked_img = cv2.bitwise_and(img,img,mask = mask)#图像与mask与操作，留下1的部分
hist = cv2.calcHist([img],[0],mask,[256],[0,256])#只转换mask区域内为直方图
#直方图均衡化
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(),256)#将结果图转换为直方图
plt.show()
#自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))#创建网格：对比度阈值/单位网格的大小
clahe = clahe.apply(img)#应用网格均衡化
cv2.imshow('imgs', clahe))
plt.hist(clahe.ravel(),256)#将结果图转换为直方图
plt.show()

#特征点提取
#Harris角点检测-只能作用于格式为float32的图像且必须为灰度图 (img0 = np.float32(img1))
#返回的是每个像素点的dst值，因此需要遍历筛选后显示
dst = cv.cornerHarris(img0, 2, 3, 0.04)#原图，角点检测的矩形框大小，sobel算子的卷积核大小，α的取值(0.04,0.06)
img0[dst>0.01*dst.max()] = (0,0,255)#将这个里面所有R数值大于dst中最大值的0.01的焦点全部变成红色

#shi_Tomas角点检测(不需要转换成float32)但是要灰度,返回值是满足条件的角点的坐标
corners = cv.goodFeaturesToTrack(img0, 1000, 0.01, 10)#原图，最多有多少个角点，门限值，角点间的最小距离
for i in corners:
    x,y = i.ravel() #将多维数组转变成一维数组
    cv2.circle(img0, (int(x),int(y)), 2, (0,0,255),-1)#画圆的圆心参数不能是小数
    
#sift算法-可以为彩图
sift = cv2.xfeatures2d.SIFT_create()#创建一个sift对象
"""https://blog.csdn.net/qq_36387683/article/details/80559237 讲这两个函数的"""
kp,des = sift.detectAndCompute(img0,None)#返回关键点和其描述符,后面一个参数是掩膜
cv2.drawKeypoints(img0,kp, img0,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#绘制关键点，输入图像，关键点，输出图像，关键点的表现形式

#fast角点检测
fast = cv2.FastFeatureDetector_create(threshold = 30,nonmaxSuppression = 1)#threshold=None(阈值), nonmaxSuppression=None（是否进行非最大化抑制）
#不加极大值抑制会导致角点更加密集
kp2 = fast.detect(img0,None)#没有特征点的描述，只有检测,可以用彩图
img2 = cv2.drawKeypoints(img0, kp2, None,(0,0,255))

#orb算法(对小尺寸图像效果不佳)
orb = cv2.ORB_create(nfeatures = 5000)#表示采取5000个特征点
kp,des = orb.detectAndCompute(img0, None)#可以用彩图
img2 = cv2.drawKeypoints(img0, kp, None,flags = 0)#画特征点

#meanshift目标追踪 ???????????????????????
cap = cv2.VideoCapture(0)
ret,frame = cap.read()
r,h,c,w = 0,330,369,392#要追踪位置的行高列宽，行和高对应纵坐标，列和宽对应横坐标
cv2.rectangle(frame, (369,0), ((369+392),(0+330)), (0,0,255),2)
win = (c,r,w,h)#追踪窗口，列行宽高
roi = frame[r:r+h,c:c+w]#先高在宽，先纵坐标，在横坐标
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])#hsv图像中的h亮度这个通道，范围就到180
cv2.normalize(roi_hist, roi_hist,0,255,cv2.NORM_MINMAX)#归一化函数
term = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)#设置终止条件，最大迭代次数，中心飘离最小次数
while(True):
    ret,frame = cap.read()
    if ret == True:
        hst = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hst], [0] , roi_hist, [0,180], 1)#返向直方图，原图，通道，模板直方图，范围，组距
        ret,win = cv2.meanShift(dst, win, term)#在哪里做meanshift，窗口，终止条件
        x,y,w,h = win
        img = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),1)
        cv2.imshow("frame", img)
        if cv2.waitKey(60) & 0xff == ord("q"):
            break
cap.release()




#傅里叶变换
# 第一步读取图片
img = cv2.imread('lena.jpg', 0)
# 第二步：进行float32形式转换
float32_img = np.float32(img)
# 第三步: 使用cv2.dft进行傅里叶变化
dft_img = cv2.dft(float32_img, flags=cv2.DFT_COMPLEX_OUTPUT)
# 第四步：使用np.fft.shiftfft()将变化后的图像的低频转移到中心位置
dft_img_ce = np.fft.fftshift(dft_img)
# 第五步：使用cv2.magnitude将实部和虚部转换为实部，乘以20是为了使得结果更大
img_dft = 20 * np.log(cv2.magnitude(dft_img_ce[:, :, 0], dft_img_ce[:, :, 1]))
# 第六步：进行画图操作
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(img_dft, cmap='gray')
plt.show()
#低通滤波
#第四六步中间插入
# 第五步：定义掩模：生成的掩模中间为1周围为0
crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2) # 求得图像的中心点位置
mask = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
# 第六步：将掩模与傅里叶变化后图像相乘，保留中间部分
mask_img = dft_center * mask
# 第七步：使用np.fft.ifftshift(将低频移动到原来的位置
img_idf = np.fft.ifftshift(mask_img)
# 第八步：使用cv2.idft进行傅里叶的反变化
img_idf = cv2.idft(img_idf)
# 第九步：使用cv2.magnitude转化为空间域内
img_idf = cv2.magnitude(img_idf[:, :, 0], img_idf[:, :, 1])
#高通滤波
#仅第五步不一样# 第五步：定义掩模：生成的掩模中间为0周围为1
crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2) # 求得图像的中心点位置
mask = np.ones((img.shape[0], img.shape[1], 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0





#级联分类器作目标检测
cap = cv2.VideoCapture(1)
while(1):
    ret, frame = cap.read()
    face_cas = cv2.CascadeClassifier("classifier/haarcascade_frontalface_default.xml")
    face_rects = face_cas.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=11)
    #scaleFactor是每次缩小图像的比例，默认是1.1，minNeighbors匹配成功所需要的周围矩形框的数目，每一个特征匹配到的区域都是一个矩形框，只有多个矩形框同时存在的时候，才认为是匹配成功
    for face_rect in face_rects:
        x, y, w, h = face_rect
        # 画出人脸
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("capture", frame)
    if cv2.waitKey(10)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
'''


cap = cv2.VideoCapture(0)
ret,frame = cap.read()
r,h,c,w = 0,330,369,392#要追踪位置的行高列宽，行和高对应纵坐标，列和宽对应横坐标
cv2.rectangle(frame, (369,0), ((369+392),(0+330)), (0,0,255),2)
win = (c,r,w,h)#追踪窗口，列行宽高
roi = frame[r:r+h,c:c+w]#先高在宽，先纵坐标，在横坐标
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])#hsv图像中的h亮度这个通道，范围就到180
cv2.normalize(roi_hist, roi_hist,0,255,cv2.NORM_MINMAX)#归一化函数
term = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)#设置终止条件，最大迭代次数，中心飘离最小次数
while(True):
    ret,frame = cap.read()
    if ret == True:
        hst = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hst], [0] , roi_hist, [0,180], 1)#返向直方图，原图，通道，模板直方图，范围，组距
        ret,win = cv2.meanShift(dst, win, term)#在哪里做meanshift，窗口，终止条件
        x,y,w,h = win
        img = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),1)
        cv2.imshow("frame", img)
        if cv2.waitKey(60) & 0xff == ord("q"):
            break
cap.release()