# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:21:27 2021

@author: Lenovo
"""

import face_and_mask_recognizer
import Register_Student
import encoded_images
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print("**************************Welcome to Face Mask Disciple Implementor******************************")
temp=1
while(temp):
    print("*************************************************************************************************")
    Choice=int(input("Enter 1 to add or remove a Student + encode newly added Images/2 to start the recognition/3 to exit "))
    if(Choice==1):
        Register_Student.register()
        encoded_images.encode()
    elif(Choice==2):
        face_and_mask_recognizer.call()
    else:
        temp=0
    