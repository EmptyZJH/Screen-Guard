import face_recognition
import cv2
import time
import numpy as np
from PIL import Image
from pylab import *


image=Image.open('C:/Users/EmptyZJH/Desktop/4.jpg')

me_image = face_recognition.load_image_file("C:/Users/EmptyZJH/Desktop/666.jpg")
me_face_encoding = face_recognition.face_encodings(me_image)[0]
known_face_encoding = [me_face_encoding, ]



def detect_self(img):
    
    # 缩小图片尺寸，加快速度
    rgb_frame = cv2.resize(img, (0, 0), fx=1, fy=1)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # BGR格式 TO RGB格式
    rgb_frame = img[:, :, ::-1]
 
    #获得图片中人脸的位置list
    face_locations = face_recognition.face_locations(rgb_frame)
    #若检测到人脸
    if face_locations:
        
        #获得检测到的人脸encoding list
        face_encodings = face_recognition.face_encodings(rgb_frame,face_locations)
        for face_encoding in face_encodings:
            if len(face_locations)==0:
                exit(0)
            # known_face_encodings 中是否存在face_encoding 并返回一个对应的True/False的list
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            if True in matches:
                #匹配到本人
                if len(face_locations)==1:
                    print("Welcome!")
                    return True
                else:
                    return False
            else:   
                
                #非本人
                return False
 
 
def main():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        #将图像与y轴对称
        frame = cv2.flip(frame,1)
 
        if detect_self(frame)==False:
                video_capture.release()
                cv2.destroyAllWindows()
                image.show()
                break
        else:
            exit(0)
        
    return 0

 
if __name__ == '__main__':
    main()
