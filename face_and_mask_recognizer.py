from PIL import Image
import face_recognition
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime 
import os
import email_to
import pandas as pd
import telegram
import pickle
import credentials

my_token = '1653433444:AAHkjQ0Z0YGLbbhjo0kRXl5v1blxPENOkRQ'
namelist=[]
dat=datetime.now()
date= dat.strftime('%m-%d-%Y')
time=dat.strftime('%H:%M:%S')

def email(name,roll):
    data=pd.read_csv("Details.csv")
    mail_address = data.loc[(data['Roll_No'] == int(roll))]['E-mail']
    server = email_to.EmailServer('smtp.gmail.com', 587, credentials.email_id,credentials.password)
    server.quick_email(mail_address[0], 'Test',
                   ['# Wear Mask and help prevent Covid-19', 'Dear '+name+' you are requested to wear your mask as soon as possible'],
                   style='h1 {color: blue}')
    return

   


def send_telegram(name,roll,token=my_token):
    msg="Dear "+ name+ " you are requested to wear your mask as soon as possible."
    data=pd.read_csv("Details.csv")
    chat_id=data.loc[(data['Roll_No'] == int(roll))]['Tel_id']
    bot=telegram.Bot(token=token)
    bot.sendMessage(chat_id=str(chat_id[0]), text=msg)
    return


def send_message(name):
         df=pd.read_csv("Details.csv")
         df1=pd.read_csv("log.csv")
         splt=name.split('_')
         dt = datetime.now()
         time= dt.strftime('%H:%M:%S')
         date= dt.strftime('%m-%d-%Y')
         temp=df.loc[df['Roll_No'] == int(splt[1])]
         temp.drop("Times_notified",inplace=True,axis=1)
         temp["Date"]=date
         temp["Time"]=time
         df1=pd.concat([df1,temp])
         print(df1)
         df1.to_csv("log.csv",index=False)
         df.loc[df["Roll_No"]==int(splt[1]),["Times_notified"]]+=1
         df.to_csv("Details.csv",index=False) 
         if(name not in namelist):
                       email(splt[0],splt[1])
                       namelist.append(name)
         send_telegram(splt[0],splt[1],my_token)
         return

    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_extractor(img):
    
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces == ():
        return None
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]
        
    return cropped_face


    
file_name = "encodings.pkl"
open_file = open(file_name, "rb")
encodeListKnown = pickle.load(open_file)
open_file.close()

def call():
    
    model2= load_model('facefeatures_new_model_2.h5')

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
    video_capture = cv2.VideoCapture(0)
    count=1
    prev_n=""
    while True:
        _, frame = video_capture.read()
        face=face_extractor(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (224, 224))
            im = Image.fromarray(face, 'RGB')
         
            img_array = np.array(im)
        
                   
            img_array = np.expand_dims(img_array, axis=0)
            pred2= model2.predict(img_array)
            name="Mask On"
            print(pred2[0][0])
            if(pred2[0][0]>0.5):
               cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            else:
               imge = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
               imge = cv2.cvtColor(imge, cv2.COLOR_BGR2RGB)
               facesCurFrame = face_recognition.face_locations(imge)
               encodesCurFrame = face_recognition.face_encodings(imge, facesCurFrame)

               for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                         name = classNames[matchIndex].upper()
                     
                         y1, x2, y2, x1 = faceLoc
                         y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                         cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                         cv2.putText(frame, name + " No Mask", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                         if(prev_n==name and name!='Unknown'):
                              count=count+1
                         if(count==10):
                              send_message(name)
                              count=1
                         prev_n=name  
                
                    else:
                        name="Unknown"
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, name + " No Mask", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        
                          
        else:
            cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    video_capture.release()
    cv2.destroyAllWindows()
   
