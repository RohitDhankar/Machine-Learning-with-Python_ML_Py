# conda activate env_tf2

# TODO -- get_img_blur( # get the VALUE of Image BLUR -- categorize as too_blurred ( reject ) or ok_blurred ( can be used in data)

# DATA_1 -- https://www.inf.ufpr.br/vri/databases/vehicle-reid/data.tgz

# TODO - https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/
# TODO - https://github.com/icarofua/vehicle-rear

# SOURCE -- https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
# Source -- https://pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
# SOURCE -- https://pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/

from PIL import Image
import pytesseract
import argparse , cv2 , os 
import numpy as np
import imutils
from imutils import perspective , contours

# WRITE TEXT 
dict_colors ={
    "color_blue" : (255, 0, 0),
    "color_green" : (0, 255, 0),
    "color_yellow" : (0, 255, 255),
    "color_red" : (0, 0, 255),
    "color_white" : (255, 255, 255),
    "color_pink" : (255,0,255)
}

def init_ocr():
    """
    """
    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,	help="path to input image to be OCR'd")
    ap.add_argument("-p", "--preprocess", type=str, default="thresh",help="type of preprocessing to be done")
    args = vars(ap.parse_args())

    path_img = "./data_dir/input_dir/img_ocr_1.png"

    # load the example image and convert it to grayscale
    #image = cv2.imread(args["image"])

    image = cv2.imread(path_img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("--type(gray---",type(gray))
    cv2.imwrite("gray_only.png", gray)
    # check to see if we should apply thresholding to preprocess the image ##if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imwrite("gray_thresh.png", gray)
    # make a check to see if median blurring should be done to remove noise ##elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)
    cv2.imwrite("gray_thresh_m_blur.png", gray)
    # write the grayscale image to disk as a temporary file so we can # apply OCR to it
    cv2.imwrite(f"do_ocr.png", gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete # the temporary file
    text = pytesseract.image_to_string(Image.open("do_ocr.png"))
    #os.remove(filename)
    print(text)
    # show the output images
    #cv2.imshow("Image", image)
    # cv2.imshow("Output", gray)
    # cv2.waitKey(0)

    # python test_ocr_1.py -i img_ocr_1.png -p thresh



def correct_skew(image_init,img_name):
    """
    Image Angle Rotation 

    https://arxiv.org/pdf/1109.3317.pdf
    https://arxiv.org/pdf/1801.00824.pdf
    https://arxiv.org/pdf/2305.14672.pdf

    getRotationMatrix2D --> 
            Parameters
            center	Center of the rotation in the source image.
            angle	Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
            scale	Isotropic scale factor. 

    """
    # convert the image to grayscale and flip the foreground     # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image_init, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    img_thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # grab the (x, y) coordinates of all pixel values that  are greater than zero, 
    # then use these coordinates to compute a rotated bounding box that contains all coordinates
    coords = np.column_stack(np.where(img_thresh > 0))
    print("---np.shape-Coordinates---",np.shape(coords))
    print("---np.shape-Coordinates---coords.dtype--",coords.shape, coords.dtype)
    
    #imageIntegral = cv2.integral(src=coords)
    # division results in values ranging from 0.0 to 1.0
    # type is floating point array (float64)
    # presentable = imageIntegral / imageIntegral.max()

    # cv2.imshow("imageIntegral", presentable)
    # cv2.waitKey(5000) # 5 Secs
    # cv2.destroyWindow('imageIntegral')


    # image_coords = coords.astype(np.uint8)
    # cv2.imshow('test_coords', image_coords)
    # cv2.waitKey(5000) # 5 Secs
    # cv2.destroyWindow('test_coords')


    # the `cv2.minAreaRect` function returns values in the     # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we     # need to add 90 degrees to the angle
    angle = cv2.minAreaRect(coords)[-1]
    print("--Coordinates--Angle-",angle)

    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make     # it positive
    else:
        angle = -angle

    # rotate the image_init to deskew it
    (h, w) = image_init.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_init, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # cv2.imshow('show_rotated', rotated)
    # cv2.waitKey(5000) # 5 Secs
    # cv2.destroyWindow('show_rotated')

    # draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow("image_init", image_init)
    cv2.imshow("show_rotated", rotated)
    cv2.waitKey(5000) # 5 Secs
    cv2.destroyWindow('show_rotated')
    cv2.destroyWindow('image_init')
    cv2.imwrite("./data_dir/output_dir/img_skew/rotated_"+str(img_name)+"_.png", rotated)


def boundary_draw(img_init,img_name):
    """
    PARAM : bound_pixels_count  
    defines Thickness of the BLACK or WHITE ( or any other COLOR )Border Padding

    """
    bound_pixels_count = 80 

    img_init[:bound_pixels_count, :] = 0
    img_init[-bound_pixels_count:, :] = 0
    img_init[:, :bound_pixels_count] = 0
    img_init[:, -bound_pixels_count:] = 0
    
    cv2.imwrite("./data_dir/output_dir/img_skew/bound_bb_"+str(img_name)+"_.png", img_init)
    return img_init


def get_warped_img(img_init,img_name):
    """
    perspective transform 
    warp perspective  

    TODO -- https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    TODO -- https://stackoverflow.com/questions/42262198/4-point-persective-transform-failure
    TODO -- https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    https://stackoverflow.com/questions/42262198/4-point-persective-transform-failure
    https://stackoverflow.com/users/2393191/micka

    """
    height_init = image_init.shape[0] #print("-[INFO]--IMAGE_INIT__Height , Width--->",image_init.shape)
    width_init = image_init.shape[1]
    count_channels = image_init.shape[2]
    print("---height_init,width_init,count_channels--->",height_init,width_init,count_channels)
    
    # image_points = "[(100,150),(200,250),(250,280),(300,350)]"
    # image_points = "[(73, 239), (356, 117), (475, 265), (187, 443)]"
    image_points = "[(0, 0), ("+str(width_init)+",0), ("+str(width_init)+", "+str(height_init)+"), (0,"+str(width_init)+")]"

    print("----get_warped_img-------TYPE-image_points-aaa-",type(image_points))
    print("----get_warped_img-------TYPE-image_points-aaa-",image_points)

    ls_rect_coords = get_order_points(img_init,image_points)
    (top_L, top_R, bot_R, bot_L) = ls_rect_coords     #(top_left, top_right, bottom_right, bottom_left) = ls_rect_coords
    #[(0, 0), (width, 0), (0, height), (width, height)] ## THIS IS ==  bot_L >> bot_R

    # def get_euler_distance(pt1, pt2):
    #     return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5

    # calc the --  Euclidean Distance -- between which points ?
    # 
    print("---bot_R , bot_L---",bot_R ,bot_L)
    print("---top_R , top_L---",top_R ,top_L)

    print("---bot_R[0] , bot_L[0]----",bot_R[0] ,bot_L[0])
    print("---bot_R[1] , bot_L[1]----",bot_R[1] ,bot_L[1])
    #
    widthA = np.sqrt(((bot_R[0] - bot_L[0]) ** 2) + ((bot_R[1] - bot_L[1]) ** 2)) 
    widthB = np.sqrt(((top_R[0] - top_L[0]) ** 2) + ((top_R[1] - top_L[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    print("--maxWidth---",maxWidth)

    heightA = np.sqrt(((top_R[0] - bot_R[0]) ** 2) + ((top_R[1] - bot_R[1]) ** 2))
    heightB = np.sqrt(((top_L[0] - bot_L[0]) ** 2) + ((top_L[1] - bot_L[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    print("--maxHeight---",maxHeight)


    destination = np.array([[0, 0], #top_L
                [maxWidth - 1, 0],  #top_R
                [maxWidth - 1, maxHeight - 1], #bot_R
                [0, maxHeight - 1]], #bot_L
                dtype="float32") 

    matrix_persp = cv2.getPerspectiveTransform(ls_rect_coords, destination) # perspective transform 
    print("----get_warped_img----matrix_persp-",type(matrix_persp)) #<class 'numpy.ndarray'>
    #print("----get_warped_img----matrix_persp-",matrix_persp)
    print("-[INFO]--get_warped_img----matrix_persp-->",matrix_persp.shape) ##(3, 3)


    warped_img = cv2.warpPerspective(img_init, matrix_persp, (maxWidth, maxHeight)) # warp perspective  
    cv2.imwrite("./data_dir/output_dir/img_skew/warped_img_"+str(img_name)+"_.png", warped_img)
    
    return warped_img


def get_order_points(img_init,image_points):
    """
    """
    # ls_rect_coords = top_L , top_R , bot_R , bot_L
    ls_rect_coords = np.zeros((4, 2), dtype = "float32")
    print("---1_ls_rect_coords--",ls_rect_coords)

    # get Sum of the points
    print("---image_points--",image_points) #TODO -- Why String ? 
    print("--TYPE-image_points-aaa-",type(image_points))
    image_points = np.array(eval(image_points), dtype = "float32")
    print("--TYPE-image_points-bbb-",type(image_points))
    print("---image_points-bbb-",image_points)


    sum_points = image_points.sum(axis = 1)
    print("--get_order_points-sum_points--",sum_points)

    ls_rect_coords[0] = image_points[np.argmin(sum_points)] # top_L == smallest sum
    ls_rect_coords[2] = image_points[np.argmax(sum_points)] # bot_R == largest sum
    print("--get_order_points-ls_rect_coords[0]--->",ls_rect_coords[0])
    print("--get_order_points-ls_rect_coords[2]--->",ls_rect_coords[2])

    # get difference between the points
    diff = np.diff(image_points, axis = 1)
    ls_rect_coords[1] = image_points[np.argmin(diff)] #top_R == SMALLEST DIFF 
    ls_rect_coords[3] = image_points[np.argmax(diff)] #bot_L == LARGEST DIFF 

    print("---3_ls_rect_coords---type(ls_rect_coords)--->",type(ls_rect_coords))
    print("---3_ls_rect_coords--",ls_rect_coords)

    img_points_ = img_init.copy()  
    for (x, y) in ls_rect_coords.astype("int32"):
        #print("--x,y---",(x,y))
        cv2.circle(img_points_, (x, y), 5, (0,0,255), -1)
    cv2.imwrite("./data_dir/output_dir/img_skew/img_points_"+str(img_name)+"_.png", img_points_)

    return ls_rect_coords

def init_img_transforms(image_init,img_name):
    """
    initial image pre-process
    # load our input image, convert it to grayscale, and blur it slightly
    
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    """
    
    gray = cv2.cvtColor(image_init, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100) #edge detection
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cv2.imwrite("./data_dir/output_dir/img_skew/img_edged_"+str(img_name)+"_.png", edged)
    return edged

def get_contours(image_init,img_edged,img_name):
    """
    Contours extraction 
    # TODO - uses - imutils 

    Wrapper Func for -- get_order_points_1()
    """
    # find contours in the edge map
    img_edged_1 = img_edged.copy()
    cnts = cv2.findContours(img_edged_1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the bounding box
    # point colors
    (cnts, _) = contours.sort_contours(cnts)
    colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
    # loop over the contours individually
    for (iter_k, c) in enumerate(cnts):
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue
        # compute the rotated bounding box of the contour, then draw the contours
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        cv2.drawContours(image_init, [box], -1, (0, 255, 0), 2)
        # show the original coordinates
        print("Object #{}:".format(iter_k + 1))
        print(box) ## BBOX Coordinates -- printed into the terminal Log Files
        print(" ---box--Above ---  "*90)
        # TODO -- get_cropped_mask( -->
        # for the Number Plate images - get only CNTRS that are a CERTAIN DISTANCE from IMAGE TOP and BOTTOM  
        cv2.imwrite("./data_dir/output_dir/img_skew/img_contours_"+str(img_name)+"_.png", image_init)
        get_order_points_1(image_init,img_name,box,iter_k)


def get_order_points_1(image_init,img_name,box,iter_k):
    """
    How diff from -- ls_rect_coords = get_order_points(img_init,image_points)

    #Ordering coordinates clockwise with Python and OpenCV
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box

    """
    colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

    #rect = order_points_old(box) ## OLd -- Deprecated 
    # Dont check to see if the new method should be used for ordering the coordinates
    # Use New here 
    rect = perspective.order_points(box)
    # show the re-ordered coordinates
    print(rect.astype("int"))
    print(" ---RECT--Above ---  "*90)
    """
    TODO -- for certain OBJECTS in the Images 
    The 2 Vals are DIFF - 
    The ndArray---> box 
    is DIFF from 
    The ndArray---> rect
    Identify such OBJ's and print somewhere asto why this DIFF 
    """
    # loop over the original points and draw them
    for ((x, y), color) in zip(rect, colors):
        cv2.circle(image_init, (int(x), int(y)), 5, color, -1)
    # draw the object num at the top-left corner
    cv2.putText(image_init, "Object #{}".format(iter_k + 1),
        (int(rect[0][0] - 15), int(rect[0][1] - 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # # show the image
    # cv2.imshow("Image", image_init)
    # cv2.waitKey(0)
    cv2.imwrite("./data_dir/output_dir/img_skew/img_cntr_pts_"+str(img_name)+"_.png", image_init)



def get_img_blur():
    """
    # TODO -- get_img_blur( # get the VALUE of Image BLUR -- 
    # Compare against an IMAGE_BLUR_THRESH value
    # categorize INIT_IMAGE - as too_blurred ( reject ) or ok_blurred ( can be used in data)
    """
    pass

def get_img_hsv():
    """
    # TODO -- get_img_blur( # get the VALUE of Image BLUR -- 
    # Compare against an IMAGE_BLUR_THRESH value
    # categorize INIT_IMAGE - as too_blurred ( reject ) or ok_blurred ( can be used in data)
    """
    pass

def plot_hsv_histograms():
    """
    # TODO -- get_img_blur( # get the VALUE of Image BLUR -- 
    """
    import matplotlib.gridspec as plt_gridspec  #print("---TYPE----",type(plt_gridspec))
    img_hsv_h = img_hsv[:,:,0]
    img_hsv_s = img_hsv[:,:,1]
    img_hsv_v = img_hsv[:,:,2]
    pass

def get_color_hist(image_init,crop_coord,img_name):
    """
    # TODO -- get_img_blur( # get the VALUE of Image BLUR -- 
    """
    hist = cv2.calcHist([image_init],[i],None,[256],[0,256])
    pass


def get_cropped_mask(image_init,crop_coord,img_name):
    """
    # TODO -- get_img_blur( # get the VALUE of Image BLUR -- 
    """
    #init a mask 
    mask_1 = np.zeros(image_init.shape[:2], np.uint8)
    pass









def get_clahe(img_clahe,
    clip_lim=3, #thresh>> contrast limiting - RANGE-->> 2 to 5 (5 will max local contrast --> maximize noise) 
    t_grid_size=8, #Split input image -->> K x K tiles , then apply Hist-Equal to each tile
    eros_iter=1,
    dil_iter=2,
    ):
    """
    CLAHE -->> Contrast Limited Adaptive Histogram Equalization - Increase Contrast of INIT_IAMGES
    Basic INIT Option >> cv2.equalizeHist
    Better Option >> cv2.createCLAHE
    """
    #img_num_pl = cv2.resize(img_num_plate, (800, 200))
    img_num_pl = cv2.fastNlMeansDenoisingColored(img_clahe, None, 10, 10, 7, 15)
    img_num_pl_gry = cv2.cvtColor(img_num_pl, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=(t_grid_size,t_grid_size))
    img_num_pl_gry = clahe.apply(img_num_pl_gry)
    img_num_pl_bin = cv2.threshold(img_num_pl_gry,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # img_num_pl_er = cv2.erode(img_num_pl_bin, (3, 3), iterations=eros_iter)
    # img_num_pl_dil = cv2.dilate(img_num_pl_er, (3, 3), iterations=dil_iter)

    return img_num_pl_bin


def get_contours_main(img_clahe_1,img_contrs,img_name,dict_colors):
    """
    TODO - experiment with various possible methods of Contour Extraction 
    RETR_EXTERNAL -- all_contours_RETR_EXTERNAL >> only PARENT Contours
    RETR_CCOMP
    RETR_LIST -- No hierarchy - flat LIST of all CNTRS
    
    TODO - experiment with Other OPTIONS for PARAMS --
    cv2.CHAIN_APPROX_SIMPLE (DONE)
    cv2.CHAIN_APPROX_NONE (TODO)

    """
    img_contrs_1 = img_contrs.copy()
    img_contrs_2 = img_contrs.copy()
    yellow = dict_colors["color_yellow"] 
    green = dict_colors["color_green"] 
    white = dict_colors["color_white"] 
    red = dict_colors["color_red"] 
    #invert  image
    #img_inverse_1 = cv2.bitwise_not(img_contrs, img_contrs)
    edged = cv2.Canny(img_clahe_1,100,200)
    cv2.imwrite("./data_dir/output_dir/img_skew/img_cntr_clahe_edged_"+str(img_name)+"_.png", edged)
    
    #TODO -- argparse >> Contour Retrieval Modes
    contrs_external, hierarchy_external = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # all_contours_RETR_EXTERNAL >> only PARENT Contours
    contrs_ccomp, hierarchy_ccomp = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # all_contours_RETR_CCOMP
    contours_list, hierarchy_list = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # all_contours_RETR_LIST >> No hierarchy - flat LIST of all CNTRS
    print("----type(contours_list----",type(contours_list)) #    print("----contours_list----",contours_list)
    print("----type(contrs_external----",type(contrs_external)) #    print("----contrs_external----",contrs_external)
    
    # dim_contrs = np.ndim(contrs_external)
    # print("----contrs_external--dim_contrs--",dim_contrs)
    # if dim_contrs >=1:
    #     print("-contrs_external[0]--",type(contrs_external[0]))
    #     print("-contrs_external[1]--",type(contrs_external[1]))
    # else:
    #     pass
    
    #invert  image again
    #img_inverse_2 = cv2.bitwise_not(img_inverse_1, img_inverse_1)
    #img_inverse_2 = cv2.cvtColor(img_clahe_1, cv2.COLOR_GRAY2RGB) #TODO - check
    
    cont_image_1 = cv2.drawContours(img_contrs, contrs_external,-1,yellow,1)
    cont_image_2 = cv2.drawContours(img_contrs_1, contrs_ccomp,-1,green,1)
    cont_image_3 = cv2.drawContours(img_contrs_2, contours_list,-1,red,1)
    
    cv2.imwrite("./data_dir/output_dir/img_skew/img_cntr_EXT_"+str(img_name)+"_.png", cont_image_1)
    cv2.imwrite("./data_dir/output_dir/img_skew/img_cntr_CCOMP_"+str(img_name)+"_.png", cont_image_2)
    cv2.imwrite("./data_dir/output_dir/img_skew/img_cntr_CON_LS_"+str(img_name)+"_.png", cont_image_3)

    return cont_image_1 , contrs_external, cont_image_2, contrs_ccomp ,cont_image_3, contours_list


def get_corners_1(img_corners,contours_list,img_name,dict_colors):
    """
    # TODO -- Whats VALUE for the -- key=cv2.contourArea
    # TODO -- SO -- https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
    """
 
    contours_all_sorted = sorted(contours_list, key=cv2.contourArea)
    # TODO -- Whats VALUE for the -- key=cv2.contourArea
    print("--contours_all_sorted--type(contours_all_sorted--\n",type(contours_all_sorted))
    #print("--contours_all_sorted--> contours_all_sorted--> \n",contours_all_sorted) # LIST of ndArrays -- Dimensions etc ? 

    box = contours_all_sorted[-2] 
    """
    # TODO - Now we know from experiment that this CONTOUR BBOX 
    Which is the 2nd LARGEST CONTOUR's BBOX 
    Has the DESIRED OBJECT for now -- the NUMBER PLATE 
    But how to Rule out the - NUMBER PLATE -- being within the 3rd LARGEST CONTOUR's BBOX , which is the box1 = contours_all_sorted[-3] below 
    """
    
    print("--contours_all_sorted--type(box--",type(box)) #<class 'numpy.ndarray'>
    box1 = contours_all_sorted[-3] 
    print("--contours_all_sorted--type(box1---",type(box1)) #<class 'numpy.ndarray'>

    def get_ele_1(input_x):
        # print("----input_x----ele_1",input_x)
        # print("----x----ele_1",input_x[1])
        return input_x[1]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in box]), key=get_ele_1)
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in box]), key=get_ele_1)
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in box]), key=get_ele_1)
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in box]), key=get_ele_1)

    bottom_right1, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in box1]), key=get_ele_1)
    top_left1, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in box1]), key=get_ele_1)
    bottom_left1, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in box1]), key=get_ele_1)
    top_right1, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in box1]), key=get_ele_1)
    
    bottom_right = box[bottom_right][0]
    top_left = box[top_left][0]
    bottom_left = box[bottom_left][0]
    top_right = box[top_right][0]

    bottom_right1 = box1[bottom_right1][0]
    top_left1 = box1[top_left1][0]
    bottom_left1 = box1[bottom_left1][0]
    top_right1 = box1[top_right1][0]

    corners = (top_left, top_right, bottom_left, bottom_right)
    corners1 = (top_left1, top_right1, bottom_left1, bottom_right1)

    circle_radius = 4
    for corner in corners:
        print("-aa---corner---",corner) #[109 215]
        cornr_image = cv2.circle(img_corners, tuple(corner), circle_radius,dict_colors["color_red"], -1)
        cv2.imwrite("./data_dir/output_dir/img_skew/img_cornr__"+str(img_name)+"_.png", cornr_image)

    for corner1 in corners1:
        print("-aa---corner1---",corner1)
        cornr_image1 = cv2.circle(img_corners, tuple(corner1), circle_radius,dict_colors["color_green"], -1)
        cv2.imwrite("./data_dir/output_dir/img_skew/img_cornr_1_"+str(img_name)+"_.png", cornr_image1)

    return corners



def pre_process_1(image_init,img_name):
    """
    pre_process_1

    """
    img_clahe = image_init.copy()
    img_contrs = image_init.copy()
    img_corners = image_init.copy()
    # hardCoded for now 
    clip_lim=3
    t_grid_size=8
    eros_iter=1
    dil_iter=2
    img_clahe_1 = get_clahe(img_clahe,clip_lim,t_grid_size,eros_iter,dil_iter)
    #cv2.imwrite(
    cont_image_1,contrs_external,cont_image_2,contrs_ccomp,cont_image_3,contours_list = get_contours_main(img_clahe_1,img_contrs,img_name,dict_colors)
    corners = get_corners_1(img_corners,contours_list,img_name,dict_colors)


if __name__ == "__main__":
    # path_img1 = "./data_dir/input_dir/img_skw_1.png"
    # path_img2 = "./data_dir/input_dir/img_skw_2.png"
    # path_img3 = "./data_dir/input_dir/img_skw_3.png"

    # path_img1 = "./data_dir/input_dir/warped_img_img_skw_1_.png"
    # path_img2 = "./data_dir/input_dir/warped_img_img_skw_2_.png"
    # path_img3 = "./data_dir/input_dir/warped_img_img_skw_3_.png"

    path_img2 = "./data_dir/input_dir/mh_0.png"
    path_img3 = "./data_dir/input_dir/mh_1.png"

    #ls_imgs = [path_img1,path_img2,path_img3]
    ls_imgs = [path_img2,path_img3]

    for iter_img in range(len(ls_imgs)):
        img_name = str(ls_imgs[iter_img]).rsplit("/",1)[1]
        img_name = str(img_name).rsplit(".png",1)[0]
        print("-[INFO]--INIT_IMAGE_NAME--->>",img_name)
        image_init = cv2.imread(ls_imgs[iter_img])
        # correct_skew(image_init,img_name)  #Image Angle Rotation 
        # img_init = boundary_draw(image_init,img_name) 
        # img_edged = init_img_transforms(image_init,img_name)
        # get_contours(image_init,img_edged,img_name) 
        #get_warped_img(image_init,img_name)

        #MAIN 
        pre_process_1(image_init,img_name)

        # get_warped_image_1()
        # get_corners()
        
