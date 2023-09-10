

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

    #Crop ROI of BOX -- or dont crop just search within ROI 
    # Within this ROI of BOX
    # get CONTRS 
    # get RATIO of CONTRS -- height with this BOX ROI 
    boundingRect(

    
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

