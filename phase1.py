import cv2
import numpy
from scipy.stats import skew
from PIL import Image
from pymongo import MongoClient
import os
from collections import OrderedDict


# function to calculate mean for color moments
def getmean(param):
    avg_color_per_row = numpy.average(param, axis=0)
    avg_color = numpy.average(avg_color_per_row, axis=0)
    return avg_color


# function to calculate standard deviation for color moments
def getstd(param):
    (means, stddev) = cv2.meanStdDev(param)
    return stddev


# function to calculate skewness for color moments
def getskew(param):
    skewYUV = skew(param.reshape(-1, 3))
    return skewYUV


# function to load feature descriptors in mongodb
def insertData(file, mean, std, skew):
    client = MongoClient(
        "mongodb+srv://srinivethini18:mongo1234@cluster0-sffpx.mongodb.net/test?retryWrites=true&w=majority")
    # print("conn ",client)
    db = client.get_database("hand")
    data = {'hand': file,
            'avgY': mean[0],
            'avgU': mean[1],
            'avgV': mean[2],
            'stdY': std[0],
            'stdU': std[1],
            'stdV': std[2],
            'skewY': skew[0],
            'skewU': skew[1],
            'skewV': skew[2]
            }
    query = db.images.insert_one(data)


def insertDataSIFT(file, keypoints):
    client = MongoClient(
        "mongodb+srv://srinivethini18:mongo1234@cluster0-sffpx.mongodb.net/test?retryWrites=true&w=majority")
    db = client.get_database("hand")
    data = {'hand': file,
            'keypoints': keypoints
            }
    query = db.sift.insert_one(data)


# function to compute color moments for image
def findCM(file, option):
    # Load image
    img_bgr = cv2.imread(file)
    summean = [0, 0, 0]
    sumstd = 0
    sumskew = 0
    block_len = 0
    # Convert Image from RGB to YUV
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    # splitting the blocks into 100*100 windows to calculate the mean, standard deviation and skewness
    for r in range(0, img_yuv.shape[0], 100):
        for c in range(0, img_yuv.shape[1], 100):
            summean = numpy.add(summean, getmean(img_yuv[r:r + 30, c:c + 30, :]))
            sumstd = numpy.add(sumstd, getstd(img_yuv[r:r + 30, c:c + 30, :]))
            sumskew = numpy.add(sumskew, getskew(img_yuv[r:r + 30, c:c + 30, :]))
            block_len += 1
            mean = summean / block_len
            std = (sumstd / block_len).flatten()
            skew = sumskew / block_len
    print("COLOR MOMENT 1(MEAN): \n", mean)
    print("COLOR MOMENT 2(STANDARD DEVIATION): \n", std)
    print("COLOR MOMENT 3(SKEWNESS): \n", skew)
    # For similarity comparision purpose taking vales of individual channel
    meanY = mean[0]
    meanU = mean[1]
    meanV = mean[2]
    stdY = std[0]
    stdU = std[1]
    stdV = std[2]
    skewY = skew[0]
    skewU = skew[1]
    skewV = skew[2]
    if option == '2':
        # Folder of images are given, after computing the color moments they will be added to the DB
        insertData(file, mean, std, skew)
    elif option == '3':
        # Given an image, after finding it's feature descriptor they will be compared with existing images
        k = int(input("Enter threshold value: "))
        findSimilarImages(file, meanY, meanU, meanV, stdY, stdU, stdV, skewY, skewU, skewV, k)
    pass


def findSIFT(file, option):
    img = cv2.imread(file)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoint, descriptors = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, keypoint, None)
    keypoints = []
    for i in range(len(keypoint)):
        x, y = keypoint[i].pt
        angle = keypoint[i].angle
        scale = keypoint[i].size
        descriptor = []
        for j in range(len(descriptors[i])):
            descriptor.append(descriptors[i][j].item())
        keypoints.append([x, y, angle, scale, descriptor])
    if option == '1':
        print("SIFT KEYPOINT VALUES: \n", keypoints)
        print("SIFT DESCRIPTOR VALUES: \n", descriptors)
    elif option == '2':
        insertDataSIFT(file, keypoints)
    elif option == '3':
        print("SIFT KEYPOINT VALUES: \n", keypoints)
        print("SIFT DESCRIPTOR VALUES: \n", descriptors)
        rank = int(input("Enter threshold value: "))
        findSimilarImagesSIFT(file, keypoints, rank)


def findSimilarImages(file, meanY, meanU, meanV, stdY, stdU, stdV, skewY, skewU, skewV, k):
    client = MongoClient(
        "mongodb+srv://srinivethini18:mongo1234@cluster0-sffpx.mongodb.net/test?retryWrites=true&w=majority")
    db = client.get_database("hand")
    cols = db["images"]
    similarimage = {}
    similarimage_ranked = {}
    similarimage_ranked_k = {}
    # Comparing values of given image and existing images
    for values in cols.find():
        meanY_values = values['avgY']
        meanU_values = values['avgU']
        meanV_values = values['avgV']
        stdY_values = values['stdY']
        stdU_values = values['stdU']
        stdV_values = values['stdV']
        skewY_values = values['skewY']
        skewU_values = values['skewU']
        skewV_values = values['skewV']
        y_similar_value = abs((meanY_values - meanY) + (stdY_values - stdY) + (skewU_values - skewY))
        u_similar_value = abs((meanU_values - meanU) + (stdU_values - stdU) + (skewU_values - skewU))
        v_similar_value = abs((meanV_values - meanV) + (stdV_values - stdV) + (skewV_values - skewV))
        # Y has dominance over U and V
        similarity_value = (2*y_similar_value) + (1*u_similar_value) + (1*v_similar_value)
        similarimage.update({values['hand']: similarity_value})
    similarimage_ranked = OrderedDict(sorted(similarimage.items(), key=lambda x: x[1]))
    imagecount = 0
    # Displaying the highly ranked images
    for key in similarimage_ranked:
        if imagecount < k:
            similarimage_ranked_k.update({key: similarimage_ranked[key]})
            print("Matching Image", key)
        imagecount += 1


def findSimilarImagesSIFT(file, keypoints, rank):
    client = MongoClient("mongodb+srv://srinivethini18:mongo1234@cluster0-sffpx.mongodb.net/test?retryWrites=true&w=majority")
    db = client.get_database("hand")
    cols = db["sift"]
    similarSIFTimages = []
    for eachimg in cols.find():
        match = 0
        for i in range(len(keypoints)):
            edistance = []
            for j in range(len(eachimg["keypoints"])):
                euclideanValue = 0
                for k in range(128):
                    euclideanValue = euclideanValue + (keypoints[i][4][k] - eachimg["keypoints"][j][4][k]) ** 2
                euclideanValue = numpy.math.sqrt(euclideanValue)
                edistance.append(euclideanValue)
            edistance.sort()
            threshold = 0.5
            # if 1/(1+edistance[i]) <= threshold:
            if (edistance[0] / edistance[1] <= threshold):
                match = match + 1
        similarSIFTimages.append({"Name": eachimg["hand"], "MatchingIndex": match})
    sortedImageList = sorted(similarSIFTimages, key=lambda y: y['MatchingIndex'], reverse=True)
    imagecountSIFT = 0
    for i in range(len(sortedImageList)):
        if imagecountSIFT < rank:
            print(sortedImageList[i]["Name"])
            # img_match_SIFT = cv2.imread(sortedImageList[i]["Name"])
            # cv2.imshow('Matching Image', img_match_SIFT)
            # cv2.waitKey()
        imagecountSIFT += 1


# main
option = input(
    "Select an option: \n 1.Feature Descriptor of an image \n 2.Featuure descriptor of a folder of images \n 3.Finding Similarity \n ")
if option == '1':
    # Display the feature descriptor of given image in color moments or SIFT
    print("Enter filename: ")
    file = input()
    model = input("Select a model: \n 1. Color Moments 2.Scale-invariant feature transform: ")
    if model == '1':
        findCM(file, option)

    elif model == '2':
        findSIFT(file, option)


elif option == '2':
    # Load feature descriptor values of images in folder to DB
    print("Enter folder name : ")
    filefolderpath = input()
    fmodel = input("Select a model: \n 1. Color Moments 2.Scale-invariant feature transform: ")
    if fmodel == '1':
        for files in os.listdir(filefolderpath):
            filepath = filefolderpath + '\\' + files
            findCM(filepath, option)

    elif fmodel == '2':
        for files in os.listdir(filefolderpath):
            filepath = filefolderpath + '\\' + files
            findSIFT(filepath, option)

elif option == '3':
    # Finding similar images for the given image
    print("Enter filename: ")
    file = input()
    models = input("Select a model: \n 1. Color Moments \n 2.Scale-invariant feature transform: ")
    if models == '1':
        findCM(file, option)
    elif models == '2':
        findSIFT(file, option)
