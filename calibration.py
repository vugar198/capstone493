import cv2
import numpy as np

def nothing(x):
    pass

def resize(image):
    # Get screen resolution (example: 1920x1080)
    screen_width = 1920
    screen_height = 1080

    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    # Scale the image to fit within the screen
    scale_width = screen_width / original_width
    scale_height = screen_height / original_height
    scale = min(scale_width, scale_height)  # Use the smaller scaling factor

    # Compute the new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    new_dimensions = (new_width, new_height)

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return resized_image


# Load the image
image = cv2.imread('./images/IMG_0.jpg')

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window with trackbars
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Hue Min', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('Hue Max', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('Sat Min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Sat Max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Val Min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Val Max', 'Trackbars', 255, 255, nothing)

while True:
    # Get trackbar positions
    h_min = cv2.getTrackbarPos('Hue Min', 'Trackbars')
    h_max = cv2.getTrackbarPos('Hue Max', 'Trackbars')
    s_min = cv2.getTrackbarPos('Sat Min', 'Trackbars')
    s_max = cv2.getTrackbarPos('Sat Max', 'Trackbars')
    v_min = cv2.getTrackbarPos('Val Min', 'Trackbars')
    v_max = cv2.getTrackbarPos('Val Max', 'Trackbars')

    # Set HSV range based on trackbar values
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # Create a mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply the mask on the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    mask = resize(mask)
    result = resize(result)
    # Show the mask and the result
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()