#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def feature_match(img1,img2):
    img1_g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create(500)
    kp1, des1 = sift.detectAndCompute(img1_g,None)
    kp2, des2 = sift.detectAndCompute(img2_g,None)
    
    matches = []
    for i in range(len(kp1)):
        distances=[]
        for j in range(len(kp2)):
            dist=np.sqrt(sum(np.square(des1[i]-des2[j])))
            distances.append(dist)
        sorted_dist = distances.copy()
        sorted_dist.sort()
        min1=sorted_dist[0]
        min2=sorted_dist[1]
        k1 = distances.index(min1)
        k2 = distances.index(min2)
        matches.append([cv2.DMatch(i,k1,min1),cv2.DMatch(i,k2,min2)])
    matchesMask = [[0,0] for i in range(len(matches))]
    good=[]    
    
    # Choosing only the good matches by setting threshold to 0.7
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)             
    match_count=20
    if len(good)>match_count:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()   
    else:
        print("No overlap")
        return                       
    
    return good, H    


def stitch_background(img1, img2, savepath=''):
    
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    good, H = feature_match(img1,img2)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
  
    points_1 = np.float32([[0,0], [0, h2],[w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
      
    # Change field of view
    points_2 = cv2.perspectiveTransform(temp_points, H)
  
    points = np.concatenate((points_1,points_2), axis=0)
  
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min,-y_min]
    
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
  
    stitched_img = cv2.warpPerspective(img1, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    
    # Compute the padding required to make the images of the same size.
    top = -y_min
    bottom = y_max-h2
    left = -x_min
    right = x_max-w2
    img2 = cv2.copyMakeBorder(img2,top,bottom,left,right,borderType=cv2.BORDER_CONSTANT)
    
    h3, w3 = stitched_img.shape[:2]    
    black = np.zeros(3)
    for i in range(h3):
        for j in range(w3):
            p1=stitched_img[i,j,:]
            p2=img2[i,j,:]
            if not np.array_equal(p1, black) and np.array_equal(p2, black):
                stitched_img[i, j, :] = p1
            elif np.array_equal(p1, black) and not np.array_equal(p2, black):
                stitched_img[i, j, :] = p2
            elif not np.array_equal(p1, black) and not np.array_equal(p2, black):
                if sum(p1)>sum(p2):
                    stitched_img[i, j, :] = p1
                else:
                    stitched_img[i, j, :] = p2
            else:
                pass
            
    #show_image(stitched_img)  
    cv2.imwrite(f'.{savepath}',stitched_img)
    return stitched_img
    
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = '/task1.png'
    stitch_background(img1, img2, savepath=savepath)

