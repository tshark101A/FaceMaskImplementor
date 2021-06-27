# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:04:38 2021

@author: Lenovo
"""
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle

def encode():
    
    path = 'singleShot'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)
    
    
    def findEncodings(images):
        encodeList = []
    
    
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
    
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
    
    file_name = "encodings.pkl"
    
    open_file = open(file_name, "wb")
    pickle.dump(encodeListKnown, open_file)
    open_file.close()