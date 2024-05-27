
import cv2


img = cv2.imread('./train\\100131_25420000032_8708_342900000513.tif',-1)
print(img.shape)
img = img / 255
print(img)
cv2.imshow('test_image',img)
cv2.waitKey(0)