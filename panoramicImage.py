import numpy as np
import cv2

#Creating a Panoramic Image
#March 18 2020
#Virginia Saurer


#A trimming function to trim the extra black space around the image
def trim(image):
    #crop top
    if not np.sum(image[0]):
        return trim(image[1:])
    #crop top
    if not np.sum(image[-1]):
        return trim(image[:-2])
    #crop top
    if not np.sum(image[:,0]):
        return trim(image[:,1:])
    #crop top
    if not np.sum(image[:,-1]):
        return trim(image[:,:-2])
    return image


#reading in images as grey scale
imageA = cv2.imread('C:/Users/Virginia Saurer/PycharmProjects/A1/keble_a_half.bmp', 0)
image_B = cv2.imread('C:/Users/Virginia Saurer/PycharmProjects/A1/keble_b_long.bmp', 0)
imageB = trim(image_B)#trimming the extra black space around this image
imageC = cv2.imread('C:/Users/Virginia Saurer/PycharmProjects/A1/keble_c_half.bmp', 0)

#This function takes in 2 images and warps them together based on their homography relation
def addImagesTogether(imgA, imgB, showMatches=False):
    ratio = 0.75
    reprojThresh = 4.0

    (imageB, imageA) = imgA,imgB
    # Detect Keypoints and get the invariant descriptors from the 2 images
    (kpsA, featuresA) = imageFeatureDetection(imageA)
    (kpsB, featuresB) = imageFeatureDetection(imageB)

    #Matching the featrures between the 2 images
    M = matchingKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

    # Returns None if no keypoint Matches
    if M is None:
        return None

    # If there are matches the perspective warp is applied to both images with the calculated homography
    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    #the final warped image is then returned
    return result

#This function takes in an image and returns the keypoints and features of the image
#Using the BRISK descriptor
def imageFeatureDetection(image):
    descriptor = cv2.BRISK_create()
    #descriptor = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    (kp1, des1) = descriptor.detectAndCompute(image, None)

    #converting the keypoints
    kps = np.float32([kp.pt for kp in kp1])

    return (kps, des1)

#Function takes in image features and calculates matches and the homography
def matchingKeypoints(keyPointsA, keyPointsB, featuresA, featuresB,ratio, repThreshold):
    #using BruteForce match descriptor
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    # First getting the base matches
    baseMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    # looping over the base Matvhes
    for m in baseMatches:
        #checking the distance is in a particular ratio of eachother
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
        print("length of matches :")
        print(len(matches))
    # To compute homography, we need at least 4 matches
    if len(matches) > 4:
        # constructing the two sets of points
        ptsA = np.float32([keyPointsA[i] for (_, i) in matches])
        ptsB = np.float32([keyPointsB[i] for (i, _) in matches])

        # computing the homography between the 2 sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, repThreshold)

        # return the matches along with the homograpy matrix and status of each matched point
        return (matches, H, status)



# warp/add the images together to create a panorama
#First put together images A and B
result1 = addImagesTogether( imageA, imageB, showMatches=True)
result1Trimmed = trim(result1)
#Then put together images B and C
result2 = addImagesTogether(imageB, imageC, showMatches=True)
result2Trimmed = trim(result2)
#Finally put together the resulting warped images to create the final panoramic image
result = addImagesTogether( result1Trimmed,result2Trimmed, showMatches=True)
finalResult = trim(result)

#Show the final panoramic result
cv2.imshow("Final Result", finalResult)
cv2.imwrite("FinalMergeResult3.png",finalResult)
cv2.waitKey(0)