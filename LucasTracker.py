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
# vw = cv2.VideoWriter("track.avi", fourcc, 30, (640,360))

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

def get_New_Coordinate(Original, frame, x, y, size, gradOriginalX, gradOriginalY, l, debug):
    if checkBoxInBoundary(Original, x, y, size): 
        return np.matrix([[-1], [-1]])

    # Original = Original/255.0
    # frame = frame/255.0
    
    T = np.matrix(Original[y:y+size, x:x+size])

    x1 = np.matrix([[q for q in range(size)] for z in range(size)])
    y1 = np.matrix([[z] * size for z in range(size)])

     
    gradOriginalX = np.matrix(gradOriginalX[y:y+size, x:x+size])
    gradOriginalY = np.matrix(gradOriginalY[y:y+size, x:x+size])

    if debug:
        ax[0,0].imshow(gradOriginalX, cmap="gray")
        ax[0,0].set_title("Gradient X")
        ax[0,1].imshow(gradOriginalY, cmap="gray")
        ax[0,1].set_title("Gradient Y")
        ax[1,0].imshow(T, cmap="gray")
        ax[1,0].set_title("Patch")
        plt.pause(0.0001)
        plt.show()

    gradOriginalP = [np.multiply(x1, gradOriginalX), 
                     np.multiply(x1, gradOriginalY), 
                     np.multiply(y1, gradOriginalX),
                     np.multiply(y1, gradOriginalY), 
                     gradOriginalX, 
                     gradOriginalY]
    # print("--"*20)
    # print(len(gradOriginalP))
    # print("--"*20)
    
    if debug:
        for i in range(len(gradOriginalP)):
            # print(gradOriginalP[i].shape)
            ax2[i].imshow(gradOriginalP[i], cmap="gray")
        plt.pause(0.0001)

    HessianOriginal = [[np.sum(np.multiply(gradOriginalP[a], gradOriginalP[b])) for a in range(6)] for b in range(6)]
    
    Hessianinv = np.linalg.pinv(HessianOriginal)

    p1, p2, p3, p4, p5, p6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    k = 0
    bad_itr = 0
    min_cost = -1
    minW = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    W = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    while True:
        position = [[W.dot(np.matrix([[x + i], [y + j], [1]], dtype='float')) for i in range(size)] for j in range(size)]
        # print("Position "+"*"*20)
        # print(len(position))
        if not (0 <= (position[0][0])[0, 0] < cols and 0 <= (position[0][0])[1, 0] < rows and 0 <= position[size - 1][0][
            0, 0] < cols and 0 <= position[size - 1][0][1, 0] < rows and 0 <= position[0][size - 1][0, 0] < cols and 0 <=
            position[0][size - 1][1, 0] < rows and 0 <= position[size - 1][size - 1][0, 0] < cols and 0 <=
            position[size - 1][size - 1][1, 0] < rows):
            return np.matrix([[-1], [-1]])

        # I = cv2.warpAffine(frame, W, frame.shape)[y:y+size,x:x+size]
        I = np.matrix([[frame[int((position[i][j])[1, 0]), int((position[i][j])[0, 0])] for j in range(size)] for i in range(size)])
        
        
        # error = np.absolute(np.matrix(I, dtype='float') - np.matrix(T, dtype='float'))
        error = np.matrix(I, dtype='int') - np.matrix(T, dtype='int')
        if debug:
            ax[1,1].imshow(error, cmap="gray")
            ax[1,1].set_title("Warped")
            plt.pause(0.001)
        name = "debug/Error_frame-"+str(l)+"_"+str(k)+".jpg"
        k+=1
        cv2.imwrite(name, error)
        steepest_error = np.matrix([[np.sum(np.multiply(g, error))] for g in gradOriginalP])
        # print(steepest_error.shape)
        mean_cost = np.sum(np.absolute(steepest_error))
        deltap = Hessianinv.dot(steepest_error)
        dp = warpInv(deltap)

        p1, p2, p3, p4, p5, p6 = p1 + dp[0, 0] + p1 * dp[0, 0] + p3 * dp[1, 0], \
                                 p2 + dp[1, 0] + p2 * dp[0, 0] + p4 * dp[1, 0], \
                                 p3 + dp[2, 0] + p1 * dp[2, 0] + p3 * dp[3, 0], \
                                 p4 + dp[3, 0] + p2 * dp[2, 0] + p4 * dp[3, 0], \
                                 p5 + dp[4, 0] + p1 * dp[4, 0] + p3 * dp[5, 0], \
                                 p6 + dp[5, 0] + p2 * dp[4, 0] + p4 * dp[5, 0]
        # p1, p2, p3, p4, p5, p6 = p1 + dp[0, 0] , \
        #                          p2 + dp[1, 0] , \
        #                          p3 + dp[2, 0] , \
        #                          p4 + dp[3, 0] , \
        #                          p5 + dp[4, 0] , \
        #                          p6 + dp[5, 0] 
        W = np.matrix([[1+p1,p3,p5], [p2,1+p4,p6]])
        print(mean_cost)
        print(np.sum(np.absolute(deltap)))
        if (min_cost == -1):
            min_cost = mean_cost
        elif (min_cost > mean_cost):
            min_cost = mean_cost
            bad_itr = 0
            minW = W
        else:
            bad_itr += 1
        if (bad_itr == 15):
            W = minW
            print("Bad itr")
            return W.dot(np.matrix([[x], [y], [1.0]]))
        # print("Itr count : " +str(k)+" -- Bad itrs : "+str(bad_itr))
        if (np.sum(np.absolute(deltap)) < 0.001):
            print("Low error" + "--"*20)
            return W.dot(np.matrix([[x], [y], [1.0]]))


# image_names = glob.glob("Car4/img/*")
# image_names = glob.glob("Bolt2/img/*")
image_names = glob.glob("DragonBaby/DragonBaby/img/*")
image_names.sort()

## X, y, w, h
bolt = [269, 75, 34, 64]
car = [70, 51, 108, 108]
baby = [160, 83, 56, 65]
debug = True
color = np.random.randint(0, 255, (100, 3))
old_frame = cv2.imread(image_names[0])

# r = cv2.selectROI(old_frame)
# xx, yy = r[0], r[1]
# ss = max(r[3],r[2])
# print(xx,yy,ss)

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# old_frame = cv2.equalizeHist(old_frame)
# old_frame = cv2.blur(old_frame,(3,3))
rows, cols = len(old_gray), len(old_gray[0])

ss = 20
# x,y = car[0], car[1]
# shapeX, shapeY = car[2], car[3]
# xx, yy = int(car[0]+shapeX/2), int(car[1]+shapeY/2)

# shapeX, shapeY = bolt[2], bolt[3]
# xx, yy = int(bolt[0] + shapeX/2), int(bolt[1] + shapeY/2)
# xx, yy = bolt[0], bolt[1]

xx,yy = baby[0], baby[1]
shapeX, shapeY = baby[2], baby[3]
ss = 50
feature_point = [xx,yy]


l=1

for i in range(1,len(image_names)):
    frame = cv2.imread(image_names[i])
    l+=1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)

    # frame_gray = cv2.blur(frame_gray, (3,3))
    gradOriginalX = cv2.Sobel(old_gray, cv2.CV_32F, 1, 0, ksize=5)
    gradOriginalY = cv2.Sobel(old_gray, cv2.CV_32F, 0, 1, ksize=5)
    good_new = get_New_Coordinate(old_gray, frame_gray, int(feature_point[0]), int(feature_point[1]), ss, gradOriginalX, gradOriginalY, l, debug)
    
    a, b = feature_point
    c, d = int((good_new[0,0])), int((good_new[1,0]))
    frame = cv2.rectangle(frame, (int(c-shapeX/2),int(d-shapeY/2)), (int(c+shapeX/2), int(d+shapeY/2)), (0,0,255))
    if (0 <= c < cols and 0 <= d < rows):
        newfeature_point = [c, d]
        # img = cv2.add(frame,mask)
        cv2.imshow('frame',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # vw.write(frame)
        old_gray = frame_gray.copy()
        # print(newfeature_point)
        # print(newfeature_point)
        feature_point = newfeature_point
        print("----"*20)
    else:
        break


cv2.destroyAllWindows()
# vw.release()