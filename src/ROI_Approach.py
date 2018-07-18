# approach using cape cod video with region of interest
import cv2
import numpy as np


video = cv2.VideoCapture('cape_cod2.mp4')
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', 1000 , 700)

# create mask for ocean
# maybe use sand colors?
ocean_lower = np.array([100,150,0],np.uint8)
ocean_upper = np.array([140,255,255],np.uint8)


# kernel size for the gaussian blur processing
kernel_size = 25;

# parameters for region of interest
l_left = [820, 0]
l_right = [1200, 0]
t_left = [820, 1012]
t_right = [1200, 1012]
vertices = [np.array([l_left, t_left, t_right, l_right], dtype=np.int32)]


# Only keep the region of the image defined by the polygon
# formed from 'vertices'. The rest of the image is set to black
def region_of_interest(img, vertices):
    # defining a blank mask
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



# start video
while (video.isOpened()):
    ret, frame = video.read()
    imshape = frame.shape

    # convert to HSV color space
    hsv = cv2.cvtColor(region_of_interest(frame, vertices), cv2.COLOR_BGR2HSV)
    # convert to gray scale
    gray = cv2.cvtColor(region_of_interest(frame, vertices), cv2.COLOR_BGR2GRAY)
    # apply gaussian blur
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # filter image with mask
    mask_ocean = cv2.inRange(hsv, ocean_lower, ocean_upper)
    mask_white = cv2.inRange(gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_ocean)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)

    # gaussian blur
    gauss_gray = cv2.GaussianBlur(mask_yw_image, (kernel_size, kernel_size), 0)

    ret, thresh = cv2.threshold(blur_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # canny edge detection
    low_threshold = 50
    high_threshold = 150
    canny_edges = cv2.Canny(gray, low_threshold, high_threshold)

    # get contour location of the boundaries
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # draw line along the contour
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow('test', frame)



    # end video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()

