import numpy as np
try:
	import cv2
except:
	import sys
	sys.path.remove(sys.path[1])
import matplotlib.pyplot as plt
import glob

fig, ax = plt.subplots(2, 2)
fig2, ax2 = plt.subplots(1,6)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')


plt.ion()


def warpInv(p):
	inverse_output = np.matrix([[0.1]] * 6)
	val = (1 + p[0, 0]) * (1 + p[3, 0]) - p[1, 0] * p[2, 0]
	inverse_output[0, 0] = (-p[0, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
	inverse_output[1, 0] = (-p[1, 0]) / val
	inverse_output[2, 0] = (-p[2, 0]) / val
	inverse_output[3, 0] = (-p[3, 0] - p[0, 0] * p[3, 0] + p[1, 0] * p[2, 0]) / val
	inverse_output[4, 0] = (-p[4, 0] - p[3, 0] * p[4, 0] + p[2, 0] * p[5, 0]) / val
	inverse_output[5, 0] = (-p[5, 0] - p[0, 0] * p[5, 0] + p[1, 0] * p[4, 0]) / val
	return inverse_output

def checkBoxInBoundary(img, x, y, size):
	if (((y + size) > len(img)) or ((x + size) > len(img[0]))): 
		return True
	else:
		return False

def checkInBoundary(img, rect):
	x,y,w,h = rect
	if (((y + h) > img.shape[0]) or ((x + w) > img.shape[1])): 
		return False
	else:
		return True
# (jac, old_gray, frame_gray, tmp, rect, ss, gradNewX, gradNewY, gradOriginalX, gradOriginalY, l, debug)
def get_New_Coordinate(jac1, jac2, Original, frame, T, rect, size, l, debug):
	if not checkInBoundary(Original, rect): 
		return np.matrix([[-1], [-1]])

	p1, p2, p3, p4, p5, p6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	k = 0
	bad_itr = 0
	min_cost = -1
	minW = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

	W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	frame = cv2.equalizeHist(frame)
	while True:

		x,y,w,h = rect   
		k+=1
		# print('W=')
		# print(W)
		
		blur = cv2.GaussianBlur(frame,(3,3),0)
		gradNewX = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
		gradNewY = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
		gradNewX_w = cv2.warpAffine(gradNewX, W, (gradNewX.shape[1], gradNewX.shape[0]))
		gradNewY_w = cv2.warpAffine(gradNewY, W, (gradNewY.shape[1], gradNewY.shape[0]))

		gnxw_t = gradNewX_w[y:y+h, x:x+w]
		gnyw_t = gradNewY_w[y:y+h, x:x+w]
		gxT = cv2.resize(gnxw_t, (size,size))
		gyT = cv2.resize(gnyw_t, (size,size))


		gx = gxT.reshape(-1,1)
		gy = gyT.reshape(-1,1)
		sd1 = jac1*gx
		sd2 = jac2*gy
		sd = sd1 + sd2

		frame_w = cv2.warpAffine(frame, W, (frame.shape[1], frame.shape[0]))
		fw_t = frame_w[y:y+h, x:x+w]
		fwT = cv2.resize(fw_t, (size,size))		
		# fwT = cv2.equalizeHist(fwT)		

		if debug:
			ax[0,0].imshow(gnxw_t, cmap="gray")
			ax[0,0].set_title("Gradient X")
			ax[0,1].imshow(gnyw_t, cmap="gray")
			ax[0,1].set_title("Gradient Y")
			ax[1,0].imshow(fwT, cmap="gray")
			ax[1,0].set_title("Warped")
			ax[1,1].imshow(T, cmap="gray")
			ax[1,1].set_title("Template")
			plt.pause(0.0001)
		
		err = (T - fwT).reshape(-1,1)
		# stp_err = np.sum((sd*err).T, axis=1).reshape(-1,1)
		stp_err = np.matmul(sd.T,err)

		Hess = np.matmul(sd.T, sd)
		# print(Hess)
		# det = np.linalg.det(Hess)
		# print(det)
		Heinv = np.linalg.inv(Hess)
		dp = np.matmul(Heinv, stp_err)
		# sd = [gxT*jac1[0], gyT*jac2[1], gxT*jac1[2], gyT*jac2[3], gxT*jac1[4], gyT*jac2[5]]

		# stp_err = np.array([np.sum(np.matmul(sd[0].T,err)), np.sum(np.matmul(sd[1].T,err)), np.sum(np.matmul(sd[2].T,err)),
		# 				np.sum(np.matmul(sd[3].T,err)), np.sum(np.matmul(sd[4].T,err)), np.sum(np.matmul(sd[5].T,err))  ])

		# stp_err = stp_err.reshape(6,1)



		# Hess = np.array([[np.sum(np.matmul(sd[0].T,sd[0])), np.sum(np.matmul(sd[0].T,sd[1])), np.sum(np.matmul(sd[0].T,sd[2])),
		# 				np.sum(np.matmul(sd[0].T,sd[3])), np.sum(np.matmul(sd[0].T,sd[4])), np.sum(np.matmul(sd[0],sd[5]))],

		# 				[np.sum(np.matmul(sd[1].T,sd[0])), np.sum(np.matmul(sd[1].T,sd[1])), np.sum(np.matmul(sd[1].T,sd[2])),
		# 				np.sum(np.matmul(sd[1].T,sd[3])), np.sum(np.matmul(sd[1].T,sd[4])), np.sum(np.matmul(sd[1].T,sd[5]))],

		# 				[np.sum(np.matmul(sd[2].T,sd[0])), np.sum(np.matmul(sd[2].T,sd[1])), np.sum(np.matmul(sd[2].T,sd[2])),
		# 				np.sum(np.matmul(sd[2].T,sd[3])), np.sum(np.matmul(sd[2].T,sd[4])), np.sum(np.matmul(sd[2].T,sd[5]))],

		# 				[np.sum(np.matmul(sd[3].T,sd[0])), np.sum(np.matmul(sd[3].T,sd[1])), np.sum(np.matmul(sd[3].T,sd[2])),
		# 				np.sum(np.matmul(sd[3].T,sd[3])), np.sum(np.matmul(sd[3].T,sd[4])), np.sum(np.matmul(sd[3].T,sd[5]))],

		# 				[np.sum(np.matmul(sd[4].T,sd[0])), np.sum(np.matmul(sd[4].T,sd[1])), np.sum(np.matmul(sd[4].T,sd[2])),
		# 				np.sum(np.matmul(sd[4].T,sd[3])), np.sum(np.matmul(sd[4].T,sd[4])), np.sum(np.matmul(sd[4].T,sd[5]))],

		# 				[np.sum(np.matmul(sd[5].T,sd[0])), np.sum(np.matmul(sd[5].T,sd[1])), np.sum(np.matmul(sd[5].T,sd[2])),
		# 				np.sum(np.matmul(sd[5].T.T,sd[3])), np.sum(np.matmul(sd[5].T,sd[4])), np.sum(np.matmul(sd[5].T,sd[5]))]  ])

		# Heinv = np.linalg.inv(Hess)

		# dp = np.matmul(Heinv, stp_err)
		 
		# if debug:
		# 	ax[1,1].imshow(err, cmap="gray")
		# 	ax[1,1].set_title("Warped")
		# 	plt.pause(0.001)
		# name = "debug/Error_frame-"+str(l)+"_"+str(k)+".jpg"
		# k+=1
		
		# cv2.imwrite(name, err)


		if debug:
			sdd = sd.reshape(-1,6)
			for i in range(6):
				img = sdd[:,i].reshape(size,size)
				ax2[i].imshow(img, cmap="gray")
			plt.pause(0.0001)
	
		p1, p2, p3, p4, p5, p6 = p1 + dp[0, 0] + (p1 * dp[0, 0]) + (p3 * dp[1, 0]), \
								 p2 + dp[1, 0] + (p2 * dp[0, 0]) + (p4 * dp[1, 0]), \
								 p3 + dp[2, 0] + (p1 * dp[2, 0]) + (p3 * dp[3, 0]), \
								 p4 + dp[3, 0] + (p2 * dp[2, 0]) + (p4 * dp[3, 0]), \
								 p5 + dp[4, 0] + (p1 * dp[4, 0]) + (p3 * dp[5, 0]), \
								 p6 + dp[5, 0] + (p2 * dp[4, 0]) + (p4 * dp[5, 0])

		W = np.array([[1+p1,p3,p5], [p2,1+p4,p6]])
		mean_cost = np.sum(np.absolute(stp_err))
		absErr = np.sum(np.absolute(dp))
		# print('mean_cost: ', mean_cost)
		if (min_cost == -1):
			min_cost = absErr
		elif (min_cost > absErr):
			min_cost = absErr
			bad_itr = 0
			minW = W
		else:
			bad_itr += 1
		# if (bad_itr == 15):
		# 	W = minW
		# 	print("Bad itr")
		# 	return W.dot(np.matrix([[x], [y], [1.0]]))
		# print("Itr count : " +str(k)+" -- Bad itrs : "+str(bad_itr))

		
		print('abs err= ', absErr)
		if (absErr <= 0.031 and k>5):
			print("Low error" + "--"*5)
			# input('enter any key')
			H= np.vstack((W, [0,0,1]))
			Hinv = np.linalg.inv(H)
			Winv = Hinv[:2,:]
			c1= W.dot(np.matrix([[rect[0]], [rect[1]], [1.0]]))
			c2= W.dot(np.matrix([[rect[0]+rect[2]], [rect[1]+rect[3]], [1.0]]))
			return c1,c2,fwT
		elif(bad_itr>5):
			print("Iteration done" + "--"*5)
			# input('enter any key')
			frame_w = cv2.warpAffine(frame, minW, (frame.shape[1], frame.shape[0]))
			fw_t = frame_w[y:y+h, x:x+w]
			fwT = cv2.resize(fw_t, (size,size))
			H= np.vstack((minW, [0,0,1]))
			Hinv = np.linalg.inv(H)
			Winv = Hinv[:2,:]
			c1= minW.dot(np.matrix([[rect[0]], [rect[1]], [1.0]]))
			c2= minW.dot(np.matrix([[rect[0]+rect[2]], [rect[1]+rect[3]], [1.0]]))
			return c1,c2,fwT


# image_names = glob.glob("Car4/img/*")
image_names = glob.glob("Bolt2/img/*")
# image_names = glob.glob("DragonBaby/DragonBaby/img/*")
image_names.sort()

## X, y, w, h
bolt = [274, 78, 32, 35]
car = [68, 52, 108, 87]
baby = [160, 83, 56, 65]
color = np.random.randint(0, 255, (100, 3))
old_frame = cv2.imread(image_names[0])

# r = cv2.selectROI(old_frame)
# xx, yy = r[0], r[1]
# ss = max(r[3],r[2])
# print(xx,yy,ss)


# debug = True
debug = False


old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# old_frame = cv2.equalizeHist(old_frame)
# old_frame = cv2.blur(old_frame,(3,3))
rows, cols = len(old_gray), len(old_gray[0])


size= 20
# x,y = car[0], car[1]
# shapeX, shapeY = car[2], car[3]
# xx, yy = int(car[0]+shapeX), int(car[1]+shapeY)


shapeX, shapeY = bolt[2], bolt[3]
xx, yy = int(bolt[0] + shapeX), int(bolt[1] + shapeY)
x, y = bolt[0], bolt[1]

# xx,yy = baby[0], baby[1]
# shapeX, shapeY = baby[2], baby[3]
# ss = 50
feature_point = [x,y]
old_gray = cv2.equalizeHist(old_gray)
vw = cv2.VideoWriter("trackbolt.avi", fourcc, 30, (old_gray.shape[1],old_gray.shape[0]))
tmp = old_gray[y:yy, x:xx]
# tmp = cv2.equalizeHist(tmp)
cv2.imwrite('boltTemp.png', tmp)
rect1= [x,y,shapeX,shapeY]
rect= rect1


x1 = np.arange(0, size)
y1 = np.arange(0, size)
x11, y11 = np.meshgrid(x1,y1)

jx = x11.reshape(-1,1)
jy = y11.reshape(-1,1)
j1 = np.ones([jy.shape[0],1])
j0 = np.zeros([jy.shape[0],1])

js1 = np.hstack((jx, j0, jy, j0, j1, j0))
js2 = np.hstack((j0, jx, j0, jy, j0, j1))

# jac1 = [x11, np.zeros((size,size)), y11, np.zeros((size,size)), np.ones((size,size)), np.zeros((size,size))]
# jac2 = [np.zeros((size,size)), x11, np.zeros((size,size)), y11, np.zeros((size,size)), np.ones((size,size))]
# jac = np.vstack((jac1,jac2))
T = cv2.resize(tmp, (size,size)) # interpolation = cv2.INTER_AREA
l=1
for i in range(1,len(image_names)):
	frame = cv2.imread(image_names[i])
	l+=1
	
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# frame_gray = cv2.equalizeHist(frame_gray)

	# frame_gray = cv2.blur(frame_gray, (3,3))
	# gradOriginalX = cv2.Sobel(old_gray, cv2.CV_32F, 1, 0, ksize=5)
	# gradOriginalY = cv2.Sobel(old_gray, cv2.CV_32F, 0, 1, ksize=5)

	good_new, newShape, T = get_New_Coordinate(js1, js2, old_gray, frame_gray, T, rect, size, l, debug)
	
	a, b = feature_point
	c, d = int((good_new[0,0])), int((good_new[1,0]))
	cc, dd = int((newShape[0,0])), int((newShape[1,0]))
	if(cc-c<10 or dd-d<10):
		cc = rect[0] + rect[2]
		dd = rect[1] + rect[3]
	rect = [c, d, cc-c, dd-d]
	# tmp = frame_gray[d:dd, c:cc]
	frame = cv2.rectangle(frame, (int(c),int(d)), (int(cc), int(dd)), (0,0,255))
	if (0 <= c < cols and 0 <= d < rows):
		newfeature_point = [c, d]
		# img = cv2.add(frame,mask)
		cv2.imshow('frame',frame)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		vw.write(frame)
		old_gray = frame_gray.copy()
		# print(newfeature_point)
		# print(newfeature_point)
		feature_point = newfeature_point
		print("----"*10)
	else:
		cv2.imshow('frame',frame)
		vw.write(frame)
		old_gray = frame_gray.copy()
		T = cv2.resize(tmp, (size,size))
		rect= rect1


cv2.destroyAllWindows()
vw.release()