import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, ndimage
from skimage import io, color
from skimage import exposure

file_image	= 'cau.jpg'

im_color 	= io.imread(file_image)
im_gray  	= color.rgb2gray(im_color)

Derivative_mask_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
Derivative_mask_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
Smooth_kernel = np.array([[.11, .11, .11], [.11, .11, .11], [.11, .11, .11]])
MySharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
my_Sobel_edge = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


im_conv_Smooth        = signal.convolve2d(im_gray, Smooth_kernel, boundary='symm', mode='same')
gx = ndimage.convolve(im_gray, Derivative_mask_x)
gy = ndimage.convolve(im_gray, Derivative_mask_y)
abs = np.hypot(gx, gy)
dir = gy/gx
my = ndimage.convolve(im_gray, my_Sobel_edge)

p1 = plt.subplot(3,3,1)
p1.set_title('color image')
plt.imshow(im_color)
plt.axis('off')

p2 = plt.subplot(3,3,2)
p2.set_title('gray image')
plt.imshow(im_gray, cmap='gray')
plt.axis('off')

p3 = plt.subplot(3,3,3)
p3.set_title(' grad x')
plt.imshow(gx , cmap='gray')
plt.axis('off')

p4 = plt.subplot(3,3,4)
p4.set_title('grad y')
plt.imshow(gy, cmap='gray')
plt.axis('off')

p4 = plt.subplot(3,3,5)
p4.set_title('absolution')
plt.imshow(abs, cmap='gray')
plt.axis('off')

p4 = plt.subplot(3,3,6)
p4.set_title('direction')
plt.imshow(dir, cmap='gray')
plt.axis('off')


p4 = plt.subplot(3,3,7)
p4.set_title('smooth')
plt.imshow(im_conv_Smooth, cmap='gray')
plt.axis('off')

p4 = plt.subplot(3,3,8)
p4.set_title('my sobel edge mask')
plt.imshow(my, cmap='gray')
plt.axis('off')
plt.show()