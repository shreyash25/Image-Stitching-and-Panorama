# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def stitch(imgmark, N, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)        
    "Start you code here"
    
    overlap_arr = np.zeros((N,N))
    # Check which Images overlap
    for i in range(N):
        for j in range(N):
            __, __, overlap = feature_match(imgs[i],imgs[j],400)
            overlap_arr[i][j]=overlap
        
    # Get initial image that overlaps with most images
    k = np.argmax(np.sum(overlap_arr,axis=0))
    panorama = imgs[k]
    stitch_idx=[k]
    temp=[]
    for i in range(len(imgs)):
        if i!=k:
            if np.any([overlap_arr[i][l] for l in stitch_idx]):
                panorama = stitch_background(panorama,imgs[i],600)
                stitch_idx.append(i)
            elif not np.any([overlap_arr[i][l] for l in stitch_idx]):
                temp.append(i)
        
    if len(temp)!=0: 
        i=0        
        while i<len(temp):                                   
            if np.any([overlap_arr[temp[i]][l] for l in stitch_idx]):                
                panorama = stitch_background(imgs[temp[i]],panorama,800)                
                stitch_idx.append(temp[i])
            elif temp[i]!=temp[i-1]:
                temp.append(temp[i])                
            i+=1    
            
    cv2.imwrite(f'./{savepath}',panorama)    
    return overlap_arr

# A function to show an image
def show_image(img, delay=1000):    
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# A function that computes matches between two input images by generating given number of keypoints.
def feature_match(img1,img2,kps):           
    img1_g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create(kps)
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
    
    # Checking if the images overlap and computing homography
    min_match_count=20
    if len(good)>min_match_count:        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()   
        overlap = 1
    else:
        
        H=0
        overlap = 0                       
    
    return good, H, overlap

# A function that stitches 2 images together using given number of keypoints    
def stitch_background(img1, img2,kps):
    
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    
    # Get Good matches and Homography for img1 and img2.
    good, H, overlap = feature_match(img1,img2,kps)    
    
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
    stitched_img[translation_dist[1]:h2+translation_dist[1], translation_dist[0]:w2+translation_dist[0]] = img2
        
    return stitched_img

if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # #bonus
    overlap_arr2 = stitch('t3', N=5, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
