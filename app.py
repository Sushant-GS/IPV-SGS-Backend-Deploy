from flask import Flask
import numpy as np
import cv2
import os

app = Flask(__name__)

def median_surrounding_inpaint(image, mask, loc):
    H = image.shape[0]
    W = image.shape[1]
    i, j = loc
    arrR = []
    arrG = []
    arrB = []
    for ii in range(i-1, i+2, 1):
        for jj in range(j-1, j+2, 1):
            if ii > 0 and jj > 0 and ii < H-1 and jj < W-1 and mask[ii, jj] == 0:
                arrR.append(image[ii, jj, 0])
                arrG.append(image[ii, jj, 1])
                arrB.append(image[ii, jj, 2])
    arrR = np.sort(np.array(arrR))
    arrG = np.sort(np.array(arrG))
    arrB = np.sort(np.array(arrB))
    image[i, j, 0] = arrR[len(arrR) // 2]
    image[i, j, 1] = arrG[len(arrG) // 2]
    image[i, j, 2] = arrB[len(arrB) // 2]
    mask[i, j] = 0

def advanced_inpaint(image, mask):
    state = 0
    lock = False
    mask_found = True

    H = image.shape[0]
    W = image.shape[1]

    while mask_found:
        mask_found = False
        lock = False
        HS, HE, WS, WE, sth, stw = 0, H, 0, W, 0, 0
        if state == 0:
            HS, WS, HE, WE, sth, stw = 0, 0, H, W, 1, 1
        elif state == 1:
            HS, WS, HE, WE, sth, stw = W-1, 0, 0, H, -1, 1
        elif state == 2:
            HS, WS, HE, WE, sth, stw = H-1, 0, 0, W, -1, 1
        elif state == 3:
            HS, WS, HE, WE, sth, stw = 0, 0, W, H, 1, 1


        for i in range(HS, HE, sth):
            for j in range(WS, WE, stw):
                if mask[i, j] > 0:
                    mask_found = True
                    lock = True
                    median_surrounding_inpaint(image, mask, (i, j))
            if lock:
                state = (state + 1) % 4
                break
    return image

@app.route('/')
def test():
    return 'Test Successful ! GO to uploads endpoint to save the processed image after applying the mask'

@app.route('/uploads')
def test_inpaint():
    image_path = os.path.join('uploads', 'beach.jpg')  
    mask_path = os.path.join('uploads', 'mask.jpg')  

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return "Image or Mask not found.", 404

    result = advanced_inpaint(image, mask)

    # Save the result
    output_path = os.path.join('uploads', 'outputbeachMask.png')
    cv2.imwrite(output_path, result)

    return "Image processed and saved successfully!"

if __name__ == '__main__':
    app.run(debug=False,port=8000)
