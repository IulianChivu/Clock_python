import numpy as np
from matplotlib import pyplot as plt
from skimage import io, color

from skimage.filters import threshold_otsu

import cv2
import math

def preprocess(img):
    img_pro = np.copy(img)
    
    # conversie nivele de gri si gama 0-255, uint 8
    try:
        img_pro = color.rgb2gray(img_pro)
        img_pro = np.array(img_pro *255, dtype = 'uint8') #conversie uint8
        
    except Exception:
        pass
    
    # binarizare
    thr = threshold_otsu(img_pro)
    img_pro[img_pro >= thr] = 255
    img_pro[img_pro < thr] = 0
    
    # invert image color
    img_pro = ~img_pro
    
    # create a mask that selects only the center of the image
    h, w = img_pro.shape
    
    center = (int(w/2), int(h/2))
    # use the smallest distance between the center and image walls  and devide that by 3
    radius = int (min(center[0], center[1], w-center[0], h-center[1]) / 1.5)

    mask = np.ones((h, w), dtype = "uint8") 
    cv2.circle(mask, center, radius, 255, -1)
    
    mask[mask != 255] = 0
    
    fig = plt.figure(figsize = (20, 20))
    fig.add_subplot(1, 3, 1), plt.imshow(img_pro, cmap='gray'), plt.title("Imaginea initiala")
    fig.add_subplot(1, 3, 2), plt.imshow(mask, cmap='gray'), plt.title("Masca")
    #bitwise and with mask
    img_pro = cv2.bitwise_and(img_pro, mask)
    
    fig.add_subplot(1, 3, 3), plt.imshow(img_pro, cmap='gray'), plt.title("Imaginea rezultata")
    
    
    fig = plt.figure(figsize = (20, 20))
    fig.add_subplot(1, 2, 1), plt.imshow(img_pro, cmap='gray'), plt.title("Imaginea initiala")
    
    #erodare imagine
    kernel = np.ones((2,2), np.uint8)
    #kernel = np.ones((6,6), np.uint8)
    #kernel[0, 0] = 0
    #kernel[2, 2] = 0
    #kernel[2, 0] = 0
    #kernel[0, 2] = 0
    img_pro = cv2.erode(img_pro, kernel, iterations=1)
    
    fig.add_subplot(1, 2, 2), plt.imshow(img_pro, cmap='gray'), plt.title("Imaginea erodata")
    
    return img_pro

def hough_transform(img):
    # Copy edges to the images that will display the results in RGB
    cdstP = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    #
    linesP = cv2.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
            
    return cdstP, linesP

def compute_dist(coordonate_linii):
    lungimi = []
    if coordonate_linii is not None:
        for i in range(0, len(coordonate_linii)):
            l = coordonate_linii[i][0]

            lungime = math.sqrt ((l[2] - l[0])**2 + (l[3] - l[1])**2)
            lungimi.append(lungime)

    max_value = max(lungimi)
    index_minutar = lungimi.index(max_value)
    
    min_value = min(lungimi)
    index_orar = lungimi.index(min_value)
    
    return lungimi, index_minutar, index_orar

def plot_axa_referinta(plot_linii):
    h, w, c = plot_linii.shape
    #center = (int(w/2), int(h/2))
    cv2.line(plot_linii, (int(w/2), int(h/2)), (int(w/2), 0), (0,0,255), 3, cv2.LINE_AA)

    plt.figure(), plt.imshow(plot_linii), plt.title("Linii detectate == rosu ; axa referinta == albastru")
    
def calcul_unghi(img, coordonate_linie) :
    #dimensiuni imagine
    h, w = img.shape
    #stabilire capat linie
    d1 = np.zeros([2])
    #calcul dintre centru si coordonate
    d1[0] = abs(w//2 - coordonate_linie[0])
    d1[1] = abs(h//2 - coordonate_linie[1])
    d11 = d1[0]+d1[1]
    d2 = np.zeros(2)
    d2[0] = abs(w//2 - coordonate_linie[2])
    d2[1] = abs(h//2 - coordonate_linie[3])
    d22 = d2[0]+d2[1]

    if d11 > d22 :
      x = coordonate_linie[0]
      y = coordonate_linie[1]
    else:
      x = coordonate_linie[2]
      y = coordonate_linie[3]
   
    #calcul valori lungimii laturi triunghi
    
    a= math.sqrt((w/2-x)**2+(0-y)**2)
    
    b = h/2
    
    c = math.sqrt((w/2-x)**2+(h/2-y)**2)
    
    #determinare unghi cu teorema cosinus
    # a^2 = b^2+c^2 - 2bc*cosA
    
    cosA = (b**2 + c**2 - a**2) / (2*b*c)
    
    #calcul unghi A in radiani
    A = math.acos(cosA)

    # calcul unghi A in grade
    unghiA = A*180/math.pi
    
    # determinare semicerc linie (stanga/dreapta)
    
    if x < w/2 :
        unghiA = 360 - unghiA
    
    return unghiA   

def calcul_ceas (unghi_ora, unghi_minut):
    
    # o ora este echivalatenta cu 360 grade / 12 ore = 30 de grade
    ora = unghi_ora//30
    # un minut este echivalent cu 360 grade / 60 minute = 6 grade
    minut = round(unghi_minut/6)
    return ora, minut


#main
if __name__=="__main__":

    # incarca imagine
    img = io.imread('clock_1.jpg')
    
    # afisare imagine originala
    plt.figure(), plt.imshow(img), plt.title("Imaginea originala")
    
    # preprocesare imagine
    img = preprocess(img)
    
    # afisare imagine preprocesata
    plt.figure(), plt.imshow(img, cmap='gray'), plt.title("Imaginea preprocesata")
    
    #apelare transformata hough
    plot_linii, coordonate_linii = hough_transform(img)
    
    #afisare linii detectate
    plt.figure(), plt.imshow(plot_linii), plt.title("Linii detectate (rosu)")
    print("Linii detectate == " + str(coordonate_linii.shape[0]))
    
    
    #determinare distante si clasificare minutar / orar
    lungimi, min_i, ora_i = compute_dist(coordonate_linii)
    
    print("Minutar: dimensiune == " + str(int(lungimi[min_i])) + " coordonate == (" + str(coordonate_linii[min_i][0][0]) + ", " + str(coordonate_linii[min_i][0][1]) + ") ("  + str(coordonate_linii[min_i][0][2]) + ", "  + str(coordonate_linii[min_i][0][3]) + ") "  )
    print("Orar: dimensiune == " + str(int(lungimi[ora_i])) + " coordonate == (" + str(coordonate_linii[ora_i][0][0]) + ", " + str(coordonate_linii[ora_i][0][1]) + ") ("  + str(coordonate_linii[ora_i][0][2]) + ", "  + str(coordonate_linii[ora_i][0][3]) + ") "  )
    
    #plot axa referinta
    plot_axa_referinta(plot_linii)
    
    #determinare unghiuri
    unghi_minut = calcul_unghi(img, coordonate_linii[min_i][0])
    unghi_ora = calcul_unghi(img, coordonate_linii[ora_i][0])
    
    #determinare ceas
    ora, minut = calcul_ceas(unghi_ora, unghi_minut) 
    
    print("Ceasul este: ", int(ora)," :", int(minut))