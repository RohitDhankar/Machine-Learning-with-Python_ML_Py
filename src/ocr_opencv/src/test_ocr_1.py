
## Source -- https://pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
## https://pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/

from PIL import Image
import pytesseract
import argparse , cv2 , os 
import numpy as np

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
    
    # grab the (x, y) coordinates of all pixel values that     # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all     # coordinates
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
    angle = cv2.minAreaRect(coords)[-1]
    print("--Coordinates--Angle-",angle)

    # the `cv2.minAreaRect` function returns values in the     # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we     # need to add 90 degrees to the angle
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




if __name__ == "__main__":
    path_img1 = "./data_dir/input_dir/img_skw_1.png"
    path_img2 = "./data_dir/input_dir/img_skw_2.png"
    path_img3 = "./data_dir/input_dir/img_skw_3.png"

    ls_imgs = [path_img1,path_img2,path_img3]

    for iter_img in range(len(ls_imgs)):
        img_name = str(ls_imgs[iter_img]).rsplit("/",1)[1]
        img_name = str(img_name).rsplit(".png",1)[0]
        print("--name ---",img_name)
        image = cv2.imread(ls_imgs[iter_img])
        correct_skew(image,img_name)     