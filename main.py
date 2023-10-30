import pyautogui as pag
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from circleDetector import canny_edge_detector
from collections import defaultdict

# pag = x=-1192, y=401
# paint = 728, 39

# top left pag = x=-1920, y=363
# bottom right pag= -1 1442
# paint  0,0   1920,1080

#while True:
#    pagX, pagY = pag.position()
#    paintX=pagX+1920
#    paintY=pagY-363
#    print(f'x={paintX}, y={paintY}')
#    time.sleep(0.1)

#while True:
#    print(pag.position())
#    time.sleep(0.1)

startTime = time.time()

#im = pag.screenshot(region=(1084,374,1464-1084,1124-374))
#im.save(r"firstInput.png")

# uncomment those on Desktop ^ for testing on my laptop im gonna use a saved oldScreenshot


#imcv = np.array(im)[:, :, ::-1]
#imcv.imshow()
print(f'screenshot taken in {time.time()-startTime} seconds')

# Dark Felt Color = 2,78,70
# Light Felt Color = 22,149,128
#img = Image.open(r"firstInput.png")
img = Image.open(r"oldScreenshot.png")
pixdata = img.load()
for y in range(img.size[1]):
    for x in range(img.size[0]):
         r, g, b = img.getpixel((x, y))
         if (0<r<55) and (70<g<170) and (60<b<150):
              pixdata[x, y] = (255, 105, 180)
img.save('chromaInput.png')
print(f'chroma keyed in {time.time()-startTime} seconds')




input_image=Image.open(r"chromaInput.png")
output_image = Image.new("RGB", input_image.size)
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)
# Find circles
rmin = 10
rmax = 13
steps = 200
threshold = 0.4


points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

print(f'points appended in {time.time()-startTime} seconds')

# Function to check if two circles intersect at all
def do_circles_intersect(x1, y1, r1, x2, y2, r2):
    distance_between_centers = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance_between_centers < r1 + r2

acc = defaultdict(int)
for x, y in canny_edge_detector(input_image):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1

circles = []
circle_centers = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and not any(do_circles_intersect(x,y,r,xc,yc, rc) for xc, yc, rc in circles):
        #print(v / steps, x, y, r)
        circles.append((x, y, r))
        circle_centers.append((x,y))
print(f'circles found in {time.time()-startTime} seconds')

imcv = np.array(output_image)[:, :, ::-1]
gray = cv2.cvtColor(imcv, cv2.COLOR_BGR2GRAY)
img_circle = output_image.copy()
avg_colors =[]
for x, y, r in circles:
    mask = np.zeros_like(gray)
    cv2.circle(img_circle, (x, y), r, (0, 0, 255), 2)
    cv2.circle(mask, (x, y), r, 255, -1)
    avg_colors.append(cv2.mean(img, mask=mask)[:3])
    print("average circle color:", avg_colors)

for x, y, r in circles:
    r = 11 # override since we know radius
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(0,0,255,0))
print(f'cirlces drawn in {time.time()-startTime} seconds')


# Save output image
output_image.save("result.png")

# alternative methods for better calculations down the road. Take the centers of the balls, and paste over the pixels from the original screenshot within a set radius, since we KNOW that all the balls have the same radius.
# do a whole second pass on the circle detection by re-chroma-ing usiong the first circle pass to un chroma an area around each ball. ?
# when drawing the circles, fill in with a certain color, making the stripes different if we can.
# use cv2's built-in HoughCircles function