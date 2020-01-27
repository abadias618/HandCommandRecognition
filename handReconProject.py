#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:25:46 2020

@author: abaldiviezo

based on gogul's programm
https://gogul.dev/software/hand-gesture-recognition-p1
https://gogul.dev/software/hand-gesture-recognition-p2
"""

# -*- coding: utf-8 -*-
"""
recognize the number of fingers in your hand
and perform certain tasks when you hold the
number of fingers for a certain time

After Running PRESS 'Q' to QUIT
"""

# organize imports

#openCV version 4.6 i think

import cv2

#for managing images
import imutils

#for handling the bunch of arrays openCV returns
import numpy as np

#for some math
from sklearn.metrics import pairwise

#to open urls
import webbrowser


#global variable

bg= None

#--------------------------------------------------

# To find the running average over the background

#--------------------------------------------------

def running_average(image, accumWeight):
    
    global bg
    # initialize backgorund
    if bg is None:
        bg = image.copy().astype("float")
        return
    
    #calculate weighted avg, accumulate it and update the background
    """cv2.accumulateWeighted(src, dst, alpha[, mask]) → None
    Parameters:	

        src – Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
        dst – Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.
        alpha – Weight of the input image.
        mask – Optional operation mask.
    The function calculates the weighted sum of the input image src and the accumulator 
    dst so that dst becomes a running average of a frame sequence:

    """
    cv2.accumulateWeighted(image, bg, accumWeight)
#---------------------------------------------

# To segment the region of hand in the image

#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    # "uint8 is a type of image"
    """cv2.absdiff(src1, src2[, dst]) → dst
        Parameters:	

        src1 – first input array or a scalar.
        src2 – second input array or a scalar.
        src – single input array.
        value – scalar value.
        dst – output array that has the same size and type as input arrays.

    The function absdiff calculates:
    Absolute difference between two arrays when they have the same size and type:
    """
    diff = cv2.absdiff(bg.astype("uint8"), image)
    
    # threshold the diff image so that we get the foreground
    """cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst
    Parameters:	

        src – input array (single-channel, 8-bit or 32-bit floating point).
        dst – output array of the same size and type as src.
        thresh – threshold value.
        maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV 
                 thresholding types.
        type – thresholding type (see the details below).

    The function applies fixed-level thresholding to a single-channel array. 
    The function is typically used to get a bi-level (binary) image out of a 
    grayscale image ( compare() could be also used for this purpose) or for 
    removing a noise, that is, filtering out pixels with too small or too large 
    values. There are several types of thresholding supported by the function.
    """

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]



    # get the contours in the thresholded image
    """cv.FindContours(image, storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE, offset=(0, 0)) → contours
    Parameters:	

        image – Source, an 8-bit single-channel image. Non-zero pixels are treated 
        as 1’s. Zero pixels remain 0’s, so the image is treated as binary . You can 
        use compare() , inRange() , threshold() , adaptiveThreshold() , Canny() , 
        and others to create a binary image out of a grayscale or color one. The 
        function modifies the image while extracting the contours. If mode equals 
        to CV_RETR_CCOMP or CV_RETR_FLOODFILL, the input can also be a 32-bit 
        integer image of labels (CV_32SC1).
        contours – Detected contours. Each contour is stored as a vector of points.
        hierarchy – Optional output vector, containing information about the image 
        topology. It has as many elements as the number of contours. For each i-th 
        contour contours[i] , the elements hierarchy[i][0] , hiearchy[i][1] , 
        hiearchy[i][2] , and hiearchy[i][3] are set to 0-based indices in contours 
        of the next and previous contours at the same hierarchical level, the first 
        child contour and the parent contour, respectively. If for the contour i 
        there are no next, previous, parent, or nested contours, the corresponding 
        elements of hierarchy[i] will be negative.
        mode –

        Contour retrieval mode (if you use Python see also a note below).
            CV_RETR_EXTERNAL retrieves only the extreme outer contours. It sets 
            hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
            
        method –

        Contour approximation method (if you use Python see also a note below).
            CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal 
            segments and leaves only their end points. For example, an up-right 
            rectangular contour is encoded with 4 points.
        offset – Optional offset by which every contour point is shifted. 
        This is useful if the contours are extracted from the image ROI and 
        then they should be analyzed in the whole image context.

    The function retrieves contours from the binary image using the algorithm 
    [Suzuki85]. The contours are a useful tool for shape analysis and object 
    detection and recognition. See squares.c in the OpenCV sample directory.
    """

    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    # return None, if no contours detected

    if len(cnts) == 0:

        return

    else:

        # based on contour area, get the maximum contour which is the hand
        """cv.ContourArea(contour, slice=CV_WHOLE_SEQ) → float
        Parameters:	

            contour – Input vector of 2D points (contour vertices), stored in 
            std::vector or Mat.
            oriented – Oriented area flag. If it is true, the function returns 
            a signed area value, depending on the contour orientation (clockwise 
            or counter-clockwise). Using this feature you can determine orientation
            of a contour by taking the sign of an area. By default, the parameter 
            is false, which means that the absolute value is returned.

        The function computes a contour area. Similarly to moments() , the area 
        is computed using the Green formula. Thus, the returned area and the number
        of non-zero pixels, if you draw the contour using drawContours() or 
        fillPoly() , can be different. Also, the function will most certainly 
        give a wrong results for contours with self-intersections.
        """

        segmented = max(cnts, key=cv2.contourArea)

        return (thresholded, segmented)
#--------------------------------------------------------------

# To count the number of fingers in the segmented hand region

#--------------------------------------------------------------

def count(thresholded, segmented):

    # find the convex hull of the segmented hand region
    """cv.ConvexHull2(points, storage, orientation=CV_CLOCKWISE, return_points=0) → convexHull
    The functions find the convex hull of a 2D point set using the Sklansky’s 
    algorithm [Sklansky82] that has O(N logN) complexity in the current 
    implementation. See the OpenCV sample convexhull.cpp that demonstrates 
    the usage of different function variants.
    """

    chull = cv2.convexHull(segmented)



    # find the most extreme points in the convex hull
    """argmin(a, axis=None, out=None)
    Returns the indices of the minimum values along an axis.
        
        argmax(a, axis=None, out=None)
    Parameters:	

        a : array_like

    Input array.
        axis : int, optional

    By default, the index is into the flattened array, otherwise along the specified axis.
        out : array, optional

    If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.

    Returns the indices of the maximum values along an axis.
    
    """

    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])

    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])

    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])

    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])



    # find the center of the palm

    cX = int((extreme_left[0] + extreme_right[0]) / 2)

    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)



    # find the maximum euclidean distance between the center of the palm

    # and the most extreme points of the convex hull
    
    """skylearn.metrics.pairwise.euclidean_distances
    (X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None)
    
    Parameters

        X{array-like, sparse matrix}, shape (n_samples_1, n_features)
        Y{array-like, sparse matrix}, shape (n_samples_2, n_features)
        Y_norm_squaredarray-like, shape (n_samples_2, ), optional
    
            Pre-computed dot-products of vectors in Y (e.g., (Y**2).sum(axis=1)) 
            May be ignored in some cases, see the note below.
            squaredboolean, optional
    
            Return squared Euclidean distances.
        X_norm_squaredarray-like of shape (n_samples,), optional
    
            Pre-computed dot-products of vectors in X (e.g., (X**2).sum(axis=1)) 
            May be ignored in some cases, see the note below.

    Returns

        distancesarray, shape (n_samples_1, n_samples_2)
    
    """

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]

    maximum_distance = distance[distance.argmax()]



    # calculate the radius of the circle with 80% of the max euclidean distance obtained

    radius = int(0.8 * maximum_distance)



    # find the circumference of the circle

    circumference = (2 * np.pi * radius)



    # take out the circular region of interest which has 

    # the palm and the fingers
    
    """numpy.zeros(shape, dtype=float, order='C')
    Parameters:	

        shape : int or tuple of ints

            Shape of the new array, e.g., (2, 3) or 2.
        dtype : data-type, optional

            The desired data-type for the array, e.g., numpy.int8. Default is numpy.
            float64.
        order : {‘C’, ‘F’}, optional, default: ‘C’

            Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.

    Returns:	

            out : ndarray

            Array of zeros with the given shape, dtype, and order.

    """

    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

	

    # draw the circular ROI
    """cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0) → None
    Parameters:	

        img – Image where the circle is drawn.
        center – Center of the circle.
        radius – Radius of the circle.
        color – Circle color.
        thickness – Thickness of the circle outline, if positive. Negative thickness means that a filled circle is to be drawn.
        lineType – Type of the circle boundary. See the line() description.
        shift – Number of fractional bits in the coordinates of the center and in the radius value.

    The function circle draws a simple or filled circle with a given center and radius.
    """

    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)



    # take bit-wise AND between thresholded hand using the circular ROI as the mask

    # which gives the cuts obtained using mask on the thresholded hand image
    
    """cv2.bitwise_and(src1, src2[, dst[, mask]]) → dst
    Parameters:	

        src1 – first input array or a scalar.
        src2 – second input array or a scalar.
        src – single input array.
        value – scalar value.
        dst – output array that has the same size and type as the input arrays.
        mask – optional operation mask, 8-bit single channel array, that specifies elements of the output array to be changed.
    In case of floating-point arrays, their machine-specific bit representations 
    (usually IEEE754-compliant) are used for the operation. In case of multi-channel
    arrays, each channel is processed independently. In the second and third cases 
    above, the scalar is first converted to the array type.
    """

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)



    # compute the contours in the circular ROI
    
    """See line 145"""

    cnts, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



    # initalize the finger count

    count = 0



    # loop through the contours found

    for c in cnts:

        # compute the bounding box of the contour
        """cv2.boundingRect(points) → retval
            Parameters:	points – Input 2D point set, stored in std::vector 
               or Mat.

            The function calculates and returns the minimal up-right bounding 
            rectangle for the specified point set. 
        """

        (x, y, w, h) = cv2.boundingRect(c)



        # increment the count of fingers only if -

        # 1. The contour region is not the wrist (bottom area)

        # 2. The number of points along the contour does not exceed

        #     25% of the circumference of the circular ROI

        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):

            count += 1



    return count
    
#-----------------

# MAIN FUNCTION

#-----------------

if __name__ == "__main__":

    # initialize accumulated weight

    accumWeight = 0.5



    # get the reference to the webcam
    
    """cv2.VideoCapture(device) → <VideoCapture object>
    Parameters:	

        filename – name of the opened video file (eg. video.avi) or image sequence (eg. img_%02d.jpg, which will read samples like img_00.jpg, img_01.jpg, img_02.jpg, ...)
        device – id of the opened video capturing device (i.e. a camera index). If there is a single camera connected, just pass 0.

    """

    camera = cv2.VideoCapture(0)
    



    # region of interest (ROI) coordinates

    #top, right, bottom, left = 10, 350, 225, 590
    top, right, bottom, left = 50, 350, 300, 590



    # initialize num of frames

    num_frames = 0



    # calibration indicator

    calibrated = False



    # time of fingers mantained up counter

    count_one = 0

    count_two = 0

    count_three = 0

    count_four = 0

    # keep looping, until interrupted

    while(True):

        # get the current frame

        (grabbed, frame) = camera.read()



        # resize the frame

        frame = imutils.resize(frame, width=700)



        # flip the frame so that it is not the mirror view, 
        # cause the camera recieves as mirror

        frame = cv2.flip(frame, 1)



        # clone the frame

        clone = frame.copy()



        # get the height and width of the frame

        (height, width) = frame.shape[:2]



        # get the ROI

        roi = frame[top:bottom, right:left]



        # convert the roi to grayscale and blur it
        
        """cv2.cvtColor(src, code[, dst[, dstCn]]) → dst
            Parameters:	

        src – input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or 
                single-precision floating-point.
        dst – output image of the same size and depth as src.
        code – color space conversion code (see the description below).
        dstCn – number of channels in the destination image; if the parameter 
                is 0, the number of the channels is derived automatically from src and code .

        The function converts an input image from one color space to another. 
        In case of a transformation to-from RGB color space, the order of the 
        channels should be specified explicitly (RGB or BGR). Note that the default 
        color format in OpenCV is often referred to as RGB but it is actually BGR 
        (the bytes are reversed). So the first byte in a standard (24-bit) color 
        image will be an 8-bit Blue component, the second byte will be Green, and 
        the third byte will be Red. The fourth, fifth, and sixth bytes would then 
        be the second pixel (Blue, then Green, then Red), and so on.
        """

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        """cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) → dst
        Parameters:	

        src – input image; the image can have any number of channels, which are 
        processed independently, but the depth should be CV_8U, CV_16U, CV_16S, 
        CV_32F or CV_64F.
        dst – output image of the same size and type as src.
        ksize – Gaussian kernel size. ksize.width and ksize.height can differ but 
        they both must be positive and odd. Or, they can be zero’s and then they 
        are computed from sigma* .
        sigmaX – Gaussian kernel standard deviation in X direction.
        sigmaY – Gaussian kernel standard deviation in Y direction; if sigmaY is 
        zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are 
        computed from ksize.width and ksize.height , respectively (see 
        getGaussianKernel() for details); to fully control the result regardless 
        of possible future modifications of all this semantics, it is recommended 
        to specify all of ksize, sigmaX, and sigmaY.
        borderType – pixel extrapolation method (see borderInterpolate() for 
        details).

        The function convolves the source image with the specified Gaussian kernel. 
        In-place filtering is supported.
        """

        gray = cv2.GaussianBlur(gray, (7, 7), 0)



        # to get the background, keep looking till a threshold is reached

        # so that our weighted average model gets calibrated

        if num_frames < 30:

            running_average(gray, accumWeight)

            if num_frames == 1:

                print("[STATUS] please wait! calibrating...")

            elif num_frames == 29:

                print("[STATUS] calibration successfull...")

        else:

            # segment the hand region

            hand = segment(gray)



            # check whether hand region is segmented

            if hand is not None:

                # if yes, unpack the thresholded image and

                # segmented region

                (thresholded, segmented) = hand



                # draw the segmented region and display the frame
                """cv2.drawContours(image, contours, contourIdx, color[, 
                    thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) → None
                Parameters:	

                    image – Destination image.
                    contours – All the input contours. Each contour is stored as 
                    a point vector.
                    contourIdx – Parameter indicating a contour to draw. If it is 
                    negative, all the contours are drawn.
                    color – Color of the contours.
                    thickness – Thickness of lines the contours are drawn with. 
                    If it is negative (for example, thickness=CV_FILLED ), the 
                    contour interiors are drawn.
                    lineType – Line connectivity. See line() for details.
                    hierarchy – Optional information about hierarchy. It is only 
                    needed if you want to draw only some of the contours 
                    (see maxLevel ).
                    maxLevel – Maximal level for drawn contours. If it is 0, 
                    only the specified contour is drawn. If it is 1, the function 
                    draws the contour(s) and all the nested contours. If it is 2, 
                    the function draws the contours, all the nested contours, all 
                    the nested-to-nested contours, and so on. This parameter is 
                    only taken into account when there is hierarchy available.
                    offset – Optional contour shift parameter. Shift all the drawn 
                    contours by the specified \texttt{offset}=(dx,dy) .
                    contour – Pointer to the first contour.
                    externalColor – Color of external contours.
                    holeColor – Color of internal contours (holes).

                The function draws contour outlines in the image 
                if \texttt{thickness} \ge 0 or fills the area bounded by the 
                contours if \texttt{thickness}<0 . The example below shows how 
                to retrieve connected components from the binary image and label 
                them:
                """

                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))



                # count the number of fingers

                fingers = count(thresholded, segmented)

                

                #control which finger is held at least 100 frames

                if fingers == 1:
                    
                    count_one = count_one + 1;
                    count_two = count_two - 1;
                    count_three = count_three - 1;
                    count_four = count_four - 1;
                
                elif fingers == 2:

                    count_one = count_one - 1;
                    count_two = count_two + 1;
                    count_three = count_three - 1;
                    count_four = count_four - 1;
                    
                elif fingers == 3:

                    count_one = count_one - 1;
                    count_two = count_two - 1;
                    count_three = count_three + 1;
                    count_four = count_four - 1;
                    
                elif fingers == 4:

                    count_one = count_one - 1;
                    count_two = count_two - 1;
                    count_three = count_three - 1;
                    count_four = count_four + 1;
                    


                # reset the count of fingers every 200 frames
                
                if num_frames % 50 == 0:
                    
                    count_one = 0;
                    count_two = 0;
                    count_three = 0;
                    count_four = 0;



                # if we reach the the 50 frame mark open the file
                
                if count_one==49:
                    
                    cv2.putText(clone, str(" Opening a URL "), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    webbrowser.open('https://www.ldsbc.edu/')

                elif count_two==49:
                    
                    cv2.putText(clone, str(" Opening a .txt "), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    webbrowser.open('https://www.pyimagesearch.com/2015/02/02/just-open-sourced-personal-imutils-package-series-opencv-convenience-functions/')
                    
                elif count_three==49:
                    
                    cv2.putText(clone, str(" Opening a URL "), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    webbrowser.open('https://www.churchofjesuschrist.org/?lang=eng')
                    
                elif count_four==49:
                    
                    cv2.putText(clone, str(" Opening an image "), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    webbrowser.open('https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html')

                
                # print the number of the fingers to the screen

                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                

                # show the thresholded image to the user

                cv2.imshow("Thesholded", thresholded)



        # draw the segmented hand
        
        """cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) 
        → None
        Parameters:	

            img – Image.
            pt1 – Vertex of the rectangle.
            pt2 – Vertex of the rectangle opposite to pt1 .
            rec – Alternative specification of the drawn rectangle.
            color – Rectangle color or brightness (grayscale image).
            thickness – Thickness of lines that make up the rectangle. 
                        Negative values, like CV_FILLED , mean that the function 
                        has to draw a filled rectangle.
            lineType – Type of the line. See the line() description.
            shift – Number of fractional bits in the point coordinates.

        The function rectangle draws a rectangle outline or a filled rectangle 
        whose two opposite corners are pt1 and pt2, or r.tl() and r.br()-Point(1,1).
        """

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)



        # increment the number of frames

        num_frames += 1



        # display the frame with segmented hand to the user
    
        cv2.imshow("Video Feed", clone)



        # observe the keypress by the user
        """cv2.waitKey([delay]) → retval
        Parameters:	delay – Delay in milliseconds. 0 is the special value 
        that means “forever”.

        The function waitKey waits for a key event infinitely 
        (when \texttt{delay}\leq 0 ) or for delay milliseconds, when it is 
        positive. Since the OS has a minimum time between switching threads, 
        the function will not wait exactly delay ms, it will wait at least 
        delay ms, depending on what else is running on your computer at that 
        time. It returns the code of the pressed key or -1 if no key was 
        pressed before the specified time had elapsed.
        """

        keypress = cv2.waitKey(1) & 0xFF



        # if the user pressed "q", then stop looping

        if keypress == ord("q"):

            break
#free up memory

camera.release()

"""cv2.destroyAllWindows() → None
The function destroyAllWindows destroys all of the opened HighGUI windows.
"""

cv2.destroyAllWindows()
