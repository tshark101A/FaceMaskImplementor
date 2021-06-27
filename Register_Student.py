"""
Created on Wed May 12 15:36:53 2021

@author: Lenovo
"""

import cv2
import numpy as np
import os
from csv import writer
import pandas as pd

def register():
    t=1
    lst=[]
    directory = r'singleShot'
    while(t):
     print("Enter 1 to register a new student / 2 to remove an existing student/ 3 to exit: ")
     Course_list={1:"B.Tech",2:"B.Agriculture",3:"B.Pharma",4:"B.Com",5:"B.A",6:"BCA",7:"B.Sc",8:"M.Tech",9:"M.A",10:"M.Com",11:"MBA",12:"MCA",13:"Others"}
     Choice=int(input())
     if(Choice==1):
        Name=input("Enter the name of the student: ")
        lst.append(Name)
        Roll=int(input("Enter the University Roll No of the student: "))
        lst.append(Roll)
        E_mail= input("Enter the E-mail of the student: ")
        lst.append(E_mail)
        Phone=int(input("Enter the phone number of the student :"))
        lst.append(Phone)
        Tel_id=int(input("Enter the telegram id of the student :"))
        lst.append(Tel_id)
        print(Course_list)
        Course=int(input("Input your Course no: "))
        Crse=Course_list[Course]
        lst.append(Crse)
        Semester=input("Enter the Semester of the Student: ")    
        lst.append(Semester)
        Section=input("Enter the Section of the Student: ")    
        lst.append(Section)
        lst.append(0)
        data=pd.read_csv("Details.csv")
        a_series = pd.Series(lst, index = data.columns)
        data=data.append(a_series,ignore_index=True)
        data.to_csv("Details.csv",index=False)
        print("Data written successfully!")
        lst=[]
        print("Look into the camera for your picture: ")
        k=1
        while(k):
            cam = cv2.VideoCapture(0) 
            ret,frame=cam.read()
            print(ret)
            cam.release()
            cv2.imshow("my image", frame)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()
            k=int(input("Enter 1 to retske the image/ 0 if the image is fine: "))
        filename = 'singleShot\\'+str(Name)+'_'+str(Roll)+".jpg"
        cv2.imwrite(filename, frame)
        print("Resgistration Successful")
       
    
     elif(Choice==2):
          data= pd.read_csv("Details.csv")
          print(data)
          Name=input("Enter the name of the Student to be deleted: ")
          Roll=int(input("Enter the University Roll No of the student: "))
          data = data.loc[data["Roll_No"] != Roll]
          print(data)
          data.to_csv('Details.csv',index=False)
          print(data)
          os.remove('singleShot\\' + str(Name)+'_'+str(Roll)+'.jpg')
     else:
         t=0
    
    