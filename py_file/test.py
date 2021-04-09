
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from imutils.video import VideoStream
#import dlib
import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import PySimpleGUI as sg
#import serial 
#ard =serial.Serial('/dev/ttyUSB0',9600)

#import pyttsx3
import datetime
print("finish load lib")

#cap = cv2.VideoCapture(0)
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font =cv2.FONT_HERSHEY_SIMPLEX
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
points = []
"""
engine = pyttsx3.init()
rate = engine.getProperty('rate')   
engine.setProperty('rate', 150)   
voices=engine.getProperty("voices")
engine.setProperty('voice',voices[1].id)
def say(a):
	engine.say(a)
	engine.runAndWait()
"""
txt = True 
temp = ''
pathmain ="C:/Users/tranlequybao/Desktop/testFace/"
#pathtrain = "python C:/Users/tranlequybao/Desktop/testFace/src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25"

pathImage ="E:/win14012021/Desktop180321/testFace/Dataset/FaceData/processed/"

sg.theme("TanBlue")
menu_def = [['&File', ['&Open', '&Save', '&Properties', 'E&xit']],
                ['&Edit', ['&Paste', ['Special', 'Normal', ], 'Undo'], ],
                ['&Toolbar', ['---', 'Command &1', 'Command &2',
                              '---', 'Command &3', 'Command &4']],
                ['&Help',['&About','&Creater']], ]
layout = [[sg.Menu(menu_def)],
          [sg.Text('POWAKE - DEMO', size=(50, 1), justification='center', font='Helvetica 20')],
          [sg.B("Start",key='Start')],
          [sg.B("loadImage",key='loadImage')],
          [sg.B("train",key='train')],
          [sg.Image(filename='', key='image',size = ( 765 , 400 ), background_color = 'blue')],]
window = sg.Window("testGUI",layout)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)


    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)


    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)


    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(255, 255,255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.7, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
    print("start func main")

    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "py_file/align")
            people_detected = set()
            person_detected = collections.Counter()
            print("start open cam")

            cap  = VideoStream(src=0).start()
            #cap = cv2.VideoCapture(0)
            txt = True
            

            while (True):




                frame = cap.read()
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #faces = detector(gray) 
                
                #frame = imutils.resize(frame, width=1024,height= 768)
                #detections = face_detector.detect_faces(frame)                
                #frame = cv2.flip(frame, 1)

                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                #try:
                    #if faces_found > 1:
                        #cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    #1, (255, 255, 255), thickness=1, lineType=2)
                try: 
                        #faces_found >0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        #print(bb[i][3]-bb[i][1])
                        print(frame.shape[0])
                        print((bb[i][3]-bb[i][1])/frame.shape[0])
                        if (bb[i][3]-bb[i][1])/frame.shape[0]>0.05:
                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)

                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                 
                            if best_class_probabilities > 0.95:
                                draw_border(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2,5,10)
                                #cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                name = class_names[best_class_indices[0]]
                                #cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                #1, (255, 255, 255), thickness=1, lineType=2)
                                draw_text(frame,str(name), pos =(text_x, text_y),text_color=(255, 255, 255),text_color_bg=(0, 0, 0))
                                cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 37),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 0, 255), thickness=1, lineType=2)
                                
                                #say("hello"+str(name))
                                try:
                                    print('vaof ghi')
                                    if txt:

                                        file = open("/home/tranlequybao/Desktop/face_demo/data.txt", "a")
                                        time = datetime.datetime.now()
                                        file.write('Tên :'+ name + '    Vào lúc :'+ str(time)+'   \n')
                                        file.close() 
                                        txt = False
                                        temp =name 
                                        print("ghi ok")
                                        print(txt+'kkkkkkkkkkkkkkkkkkkkkkkkkkkk')
                                        
                                             
                                            
                                    else:
                                        pass
                                except:
                                    pass
                                if str(temp) !=str(name) :
                                    txt =True
                                    print(" set txt True")
                                


                                person_detected[best_name] += 1
                                #for file in os.listdir(pathImage):
                                    #if str(file) == str(name):
                            #if name !=name:
                                #ard.write('s'.encode())

                                

                                
                            else:
                                """
                                    for face in faces:
                                        x1 = face.left()
                                        y1 = face.top()
                                        x2 = face.right()
                                        y2 = face.bottom()
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        landmarks = predictor(gray, face)
                                        for n in range(0, 81):
                                            x = landmarks.part(n).x
                                            y = landmarks.part(n).y
                                            points.append((x,y))
                                            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                                """
                                #cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 155), 2)
                                #cv2.putText(frame, "Unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                               # 1, (0, 0, 255), thickness=1, lineType=2)
                                name = "Unknown"
                                draw_border(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2,5,10)
                                """
                                    for res in detections:
                                        x, y, w, h = res["box"]
                                        x1,y1 = abs(x),abs(y)
                                        x2,y2 = x1+w,y1+h
                                        confidence = res["confidence"]
                                        keypoints = res["keypoints"].values()
                                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                                        cv2.putText(frame,f"c:{confidence:.1f}",(x1,y1),cv2.FONT_ITALIC,1,(0,255,255),1)
                                        cv2.putText(frame,str(fps),(30,30),cv2.FONT_ITALIC,1,(0,255,255),4)
                                        for point in keypoints:
    	                                    cv2.circle(frame,point, 3, (255, 0, 0), -1)
    	                        """                                 
                            
                except:
                    pass
                cv2.imshow('Face Recognition', frame)
                #imgface = cv2.imread(pathImage+str(name)+"/"+str(name)+"1.jpg")
                #cv2.imshow("kkkkk",imgface)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


while True:
    event,values = window.read(timeout=1/1000)
    if event =="Exit" or event == sg.WIN_CLOSED:
    	break
    elif event == 'Start':
        window.Hide()
        main()
    elif event == "loadImage":
        
        name =sg.popup_get_text('Nhập tên đầy đủ của bạn!')
        i,j=0,0
        offset =30
        #offset = int(values['sl'])
        print(offset)
        if name == "":
            sg.Popup('Tên bạn đã để trống,vui lòng nhập lại tên!')
        else:
            try:
            #dialog.Show(False)
                path = "/home/tranlequybao/Desktop/face_demo/DataSet/inputImage/"+ name
                if os.path.exists(path):
                    print("folder is exists")
                else:
                    print("folder is created")
                    os.mkdir(path)
                for root,dirs,files in os.walk(path):
                    for file in files:
                        j += 1
                while True:
                    ret,img = cap.read()
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    
                    faces = haar_face_cascade.detectMultiScale(gray,1.2,5)

                    
                    for (x,y,w,h) in faces:
                        i +=1
                        cv2.imwrite(path+"/"+name+str(i+j)+".jpg",gray[y-offset:y+h+offset,x-offset:x+w+offset])
                        

                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
                        cv2.putText(img,"sample"+str(i),(x,y), font, 1,(0,0,255),4,cv2.LINE_AA)

                        cv2.imshow("tao co so du lieu",img)
                    cv2.waitKey(50)
                    if i>=100:
                        cv2.destroyAllWindows()
                        cap.release()
                        #winsound.PlaySound('pip.wav',1)
                        sg.Popup('Hoàn thành!')
                        break
            except:
                
                continue

    elif event =="imagecut":
        try:
            os.system(trainpath)
        except:
            sg.Popup('LỖI CMN RỒI')



    elif event=="train":
        try:
            os.system(pathtrain)
        except:
           sg.Popup('LỖI CMN RỒI')

