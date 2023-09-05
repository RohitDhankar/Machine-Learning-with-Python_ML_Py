
## Source -- https://pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
## https://pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/

from PIL import Image
import pytesseract
import argparse , cv2 , os 

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



def correct_skew(image):
    """
    https://arxiv.org/pdf/1109.3317.pdf
    https://arxiv.org/pdf/1801.00824.pdf
    https://arxiv.org/pdf/2305.14672.pdf

    """
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    img_thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imwrite("./data_dir/output_dir/img_skew/skw_1.png", img_thresh)


if __name__ == "__main__":
    path_img = "./data_dir/input_dir/img_ocr_7.png"
    image = cv2.imread(path_img)
    correct_skew(image)     