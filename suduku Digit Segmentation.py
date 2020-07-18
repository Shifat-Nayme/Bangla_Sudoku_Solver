import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
count = 0
import math

final_word = []
final_image = []
 
def alpha_bat(image,size,start,end):
    global final_image
    h,w = image.shape
    img_array = np.zeros((h,size+10), dtype="uint8")
    x = 0
    y = 0
    start = start - 5
    end = end + 5

    #print(x)
    print(img_array.shape)
    for i in range(h):
        y = 0
        for j in range(w):
            if(j>=start and j<end):
                img_array[x][y] = image[i][j]
                y = y + 1

        x = x + 1

    final_image.append(img_array)

def word_image_array(image,size,letter,start,end):
    global final_word
    h,w = image.shape
    img_array =np.zeros((size,w), dtype="uint8")
    x = 0
    y = 0
    for i in range(h):
        if(i>=start and i<end):
            y = 0
            for j in range(w):
                img_array[x][y] = image[i][j]
                y = y + 1
            
            x = x + 1

    img = np.stack((img_array,) * 3,-1)
    img = img.astype(np.uint8)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.resize(grayed, dsize =(w,100), interpolation = cv2.INTER_AREA)
    final_word.append(grayed)

   


def word_lenght_img(image,size,start,end):
    h,w = image.shape
    img_array = np.zeros((h,size), dtype="uint8")
    x = 0
    y = 0
  
    for i in range(h):
        y = 0
        for j in range(w):
            if(j>=start and j<end):
                img_array[x][y] = image[i][j]
                y = y + 1
            
        x = x + 1
    h2,w2 = img_array.shape
    #plt.imshow(img_array)
    #plt.show()
    cout = 0
    c = 0
    gap = []

    gap.append(0)
    for i in range(h2):
        cout = 0
        for j in range(w2):
            if(img_array[i][j] == 0):
                cout = cout +1
                if cout == w2-5:
                    c = c + 1
                    gap.append(i)

    gap.append(h2)
    word_size = []
    word_start =[]
    word_end = []
    for i in range(len(gap)-1):
        if(gap[i+1] - gap[i]>5):
            word_size.append(gap[i+1] - gap[i])
            word_start.append(gap[i])
            word_end.append(gap[i+1])
    
    for i in range(len(word_size)):
        word_image_array(img_array,word_size[i],size,word_start[i],word_end[i])
       

def line_image(image,size,letter,start,end):
    h,w = image.shape
    img_array =np.zeros((size,w), dtype="uint8")
    x = 0
    y = 0

    for i in range(h):
        if(i>=start and i<end):
            y = 0
            for j in range(w):
                img_array[x][y] = image[i][j]
                y = y + 1
            
            x = x + 1
    
    return img_array


def word_image(image):
    word_gap = []
    c = 0
    cout = 0
    sum = []
    null_list = []
    h,w = image.shape
    copy = image
    sum.append(copy.sum(axis = 0))
    a = np.asarray(sum)
    m,n = a.shape
    #print(m,n)
    gapp = []
    for i in range(m):
        for j in range(n-1):
            if(a[i][j]<300):
                null_list.append(a[i][j])
                gapp.append(j)
    
    gapp.append(w)
    word_size = []
    word_start =[]
    word_end = []
    for i in range(len(gapp)-1):
        if(gapp[i+1] - gapp[i]>5):
            word_size.append(gapp[i+1] - gapp[i])
            word_start.append(gapp[i])
            word_end.append(gapp[i+1])
    
    for i in range(len(word_size)):
        word_lenght_img(image,word_size[i],word_start[i],word_end[i])


def word_part(image):
    h,w = image.shape
    f = h/2
    f=math.floor(f)
    sizee = (h-6)-(h-f-8)
    arr =np.zeros((sizee,w), dtype="uint8")
    x = 0
    y = 0
    for i in range(h-f-8,h-6):
        y = 0
        for j in range(w):
                if(image[i][j]>1):
                   arr[x][y] = 1
                else:
                   arr[x][y] = 0
                y = y+1

        x = x +1

    sum =[]
    sum.append(arr.sum(axis = 0))
    a = np.asarray(sum)
    m,n = a.shape
    gapa = []
    gapa.append(0)
    for i in range(m):
        for j in range(n-1):
            if(a[i][j]==0):
                gapa.append(j)

    gapa.append(w)
    #print(gapa)
    alpha_size = []
    alpha_start =[]
    alpha_end = []
    for i in range(len(gapa)-1):
        if(gapa[i+1] - gapa[i]>5):
            alpha_size.append(gapa[i+1] - gapa[i])
            alpha_start.append(gapa[i])
            alpha_end.append(gapa[i+1])
    

    for i in range(len(alpha_size)):
        alpha_bat(image,alpha_size[i],alpha_start[i],alpha_end[i])



import cv2


path = "C:\\Users\\Fuad\\Desktop\\New folder\\su.jpg"
img = cv2.imread(path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

main_image = gray
ret, thresh = cv2.threshold(gray, 150, 150, 
                            cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU) 

gray=thresh

"""img = thresh
im = Image.fromarray(img)
im.save("C:\\Users\\USER\\Desktop\\su.jpg")'''
   
kernel = np.ones((3, 3), np.uint8) 
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                            kernel, iterations = 2) 
  
# Background area using Dialation 
bg = cv2.dilate(closing, kernel, iterations = 1) 
  
# Finding foreground area 
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
ret, fg = cv2.threshold(dist_transform, 0.02* dist_transform.max(), 255, 0)
copy = fg.copy()
"""
height = gray.shape[0]
width = gray.shape[1]

print("\n Resizing Image........")
image = cv2.resize(gray, dsize =(1000, int(1000*height/width)), interpolation = cv2.INTER_AREA)
print("Noise Removal From Image.........")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

h,w = image.shape


cout = 0
c = 0
gap = []

gap.append(0)
for i in range(h):
    cout = 0
    for j in range(w):
        if(image[i][j] ==0):
            cout = cout +1
            if cout >=950:
                c = c + 1
                gap.append(i)

gap.append(h)

#print(gap)
letter_size = []
line_start =[]
line_end = []
for i in range(len(gap)-1):
    if(gap[i+1] - gap[i]>10):
        letter_size.append(gap[i+1] - gap[i])
        line_start.append(gap[i])
        line_end.append(gap[i+1])


for i in range(len(line_start)):
    print(line_start[i])
    print(line_end[i])

copy_image = np.array(image)

crop_image = []
for i in range(len(letter_size)):
    crop = line_image(copy_image,letter_size[i],letter_size,line_start[i],line_end[i])
    img = np.stack((crop,) * 3,-1)
    img = img.astype(np.uint8)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(grayed)
    #plt.show()
    grayed = cv2.resize(grayed, dsize =(1000, 150), interpolation = cv2.INTER_AREA)
    crop_image.append(grayed)


for i in range(len(crop_image)):
    word_image(crop_image[i])

for i in range(len(final_word)):
    word_part(final_word[i])

model_image = []

'''outpath ="C:\\Users\\Fuad\\Desktop\\MathNet-Code-master\\"
idx =0
print(len(model_image))
for j in range(len(model_image)):
    img = model_image[j]
    #plt.imshow(img)
    #plt.show()
    #img = cv2.resize(img,dsize =(28,28), interpolation = cv2.INTER_AREA)
    cv2.imwrite(outpath + str(idx) + '.jpg', img)
    idx = idx + 1'''



print(len(final_image))
for j in range(len(final_image)):
    #plt.imshow(final_image[j])
    #plt.show()
    gray=final_image[j]
    gray = cv2.resize(gray,dsize =(64,64), interpolation = cv2.INTER_AREA)
    h,w = gray.shape
    image = gray
    #plt.imshow(gray)
    #plt.show()
    ret, thresh = cv2.threshold(gray, 150, 150, 
                            cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU) 
    image =thresh
    #print(h)
    #print(w)
    for i in range (h):
        for j in range (w):
            if(j>55):
                image[i][j]=0
            if(i<7):
                image[i][j]=0
            if(j<8):
                image[i][j]=0

    model_image.append(image)

for j in range(len(model_image)):
    plt.imshow(model_image[j])
    plt.show()

    
