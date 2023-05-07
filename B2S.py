import matplotlib.pyplot as plt
import cv2
import math
import os # For playing the converted audio

from PIL import Image
from gtts import gTTS # module for text to speech conversion

# model
charToArray = {
    " " : [[0,0],[0,0],[0,0]],
    "a" : [
            [1,0],
            [0,0],
            [0,0]
        ],
    "b" : [
            [1,0],
            [1,0],
            [0,0]
        ],
    "c" : [
            [1,1],
            [0,0],
            [0,0]
        ],
    "d" : [
            [1,1],
            [0,1],
            [0,0]
        ],
    "e" : [
            [1,0],
            [0,1],
            [0,0]
        ],
    "f" : [
            [1,1],
            [1,0],
            [0,0]
        ],
    "g" : [
            [1,1],
            [1,1],
            [0,0]
        ],
    "h" : [
            [1,0],
            [1,1],
            [0,0]
        ],
    "i" : [
            [0,1],
            [1,0],
            [0,0]
        ],
    "j" : [
            [0,1],
            [1,1],
            [0,0]
        ],
    "k" : [
            [1,0],
            [0,0],
            [1,0]
        ],
    "l" : [
            [1,0],
            [1,0],
            [1,0]
        ],
    "m" : [
            [1,1],
            [0,0],
            [1,0]
        ],
    "n" : [
            [1,1],
            [0,1],
            [1,0]
        ],
    "o" : [
            [1,0],
            [0,1],
            [1,0]
        ],
    "p" : [
            [1,1],
            [1,0],
            [1,0]
        ],
    "q" : [
            [1,1],
            [1,1],
            [1,0]
        ],
    "r" : [
            [1,0],
            [1,1],
            [1,0]
        ],
    "s" : [
            [0,1],
            [1,0],
            [1,0]
        ],
    "t" : [
            [0,1],
            [1,1],
            [1,0]
        ],
    "u" : [
            [1,0],
            [0,0],
            [1,1]
        ],
    "v" : [
            [1,0],
            [1,0],
            [1,1]
        ],
    "w" : [
            [0,1],
            [1,1],
            [0,1]
        ],
    "x" : [
            [1,1],
            [0,0],
            [1,1]
        ],
    "y" : [
            [1,1],
            [0,1],
            [1,1]
        ],
    "z" : [
            [1,0],
            [0,1],
            [1,1]
        ],
}

# contour
url = r'C:\Users\pimju\PycharmProjects\B2S\Braille Dataset\Hello\whatisdis.jpg'
img = cv2.imread(url)
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, bw = cv2.threshold(gray,90,255,cv2.THRESH_BINARY_INV)

# contours and hierachy
contours, hir = cv2.findContours(bw,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)

img = cv2.cvtColor(bw,cv2.COLOR_GRAY2RGB)
cv2.drawContours(img,contours,-1,(255,0,0),1)

for i in range(len(contours)):
  cnt = contours[i]
  x,y,w,h = cv2.boundingRect(cnt)
  # print(x, y) #print coordinate

  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# print((x+w)-x,(y+h)-y)

image = Image.open(url)

amount = image.width/(((x+w)-x)*4)
# print(math.floor(amount))

# Calculate the number of columns
num_columns = int(math.floor(amount))

# Get the coordinates of the left-top and right-bottom corners
left, top, right, bottom = image.getbbox()

# Calculate the size of each cell
cell_width = (image.width / num_columns)
cell_height = int(cell_width * image.height / image.width)

plt.imshow(img)
plt.show()

# Crop the image into separate columns and save them as separate files
char = []
for col in range(num_columns):
    x0 = col * cell_width
    y0 = 0
    x1cel = x0 + cell_width
    y1cel = image.height
    column_image = image.crop((x0, y0, x1cel, y1cel))
    column_image.save(f"column_{col+1}.jpg")

    url2 = f"column_{col+1}.jpg"
    img = cv2.imread(url2)

    # plt.imshow(img)
    # plt.show()

    box_width = int((x1cel-x0)/2)
    box_height = int(y1cel/3)

    boxes = []
    # Loop through each box
    for i in range(3):
        r = []
        for j in range(2):
            # Define the boundaries of the current box
            x1 = int(j * box_width)
            y1 = int(i * box_height)
            x2 = int((j + 1) * box_width)
            y2 = int((i + 1) * box_height)

            # Draw box on image (optional)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop the current box from the image
            box = img[y1:y2, x1:x2]

            # Convert the box to grayscale and apply thresholding to make the dot more visible
            box_gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            _, box_thresh = cv2.threshold(box_gray, 127, 255, cv2.THRESH_BINARY_INV)

            # Find contours in the box
            contours, _ = cv2.findContours(box_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If at least one contour is found, there is a dot in the box
            if len(contours) > 0:
                # Get the coordinates of the center of the dot
                M = cv2.moments(contours[0])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Print the coordinates of the dot relative to the entire image
                dot_x = x1 + cx
                dot_y = y1 + cy
                # print(f"box ({i}, {j}) at ({dot_x}, {dot_y})")

                r.append(1)
            else:
                r.append(0)
        boxes.append(r)

    # plt.imshow(img)
    # plt.show()

    # gat the alphabet
    for key, value in charToArray.items():
        if value == boxes:
            print(key)
            char.append(key)

print(char)
SPEECH = ''.join(map(str, char))
print(SPEECH)

# The text that you want to convert to audio
mytext = SPEECH

# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)

# Saving the converted audio in a mp3 file named
# welcome
myobj.save("welcome.mp3")

# Playing the converted file
os.system("welcome.mp3")

# Wait for close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
