
import cv2

image = cv2.imread('998.jpg')
print(image.shape)
cv2.rectangle(image, (30, 130), (662, 544), (0, 0, 255), 2)
cv2.rectangle(image, (22, 119), (662, 544), (0, 0, 255), 2)  

cv2.imwrite('9980.jpg', image)