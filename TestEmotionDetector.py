import cv2
import numpy as np
from keras.models import model_from_json
from flask import Flask,request,jsonify
from PIL import Image
from keras.preprocessing import image










# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# # load json and create model
# json_file = open('model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# # load weights into new model
# emotion_model.load_weights("model/emotion_model.h5")
# emotionSeries=[]


# start the webcam feed
# cap = cv2.VideoCapture(0)
# add="http://192.168.1.4:8080/video"
# cap.open(add)q

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/







# def video_prediction_emotion(file_path):
     

#      cap = cv2.VideoCapture(file_path)

#      if os.path.isfile(file_path):
#        os.remove(file_path)

#      while True:
#     # Find haar cascade to draw bounding box around face
#        ret, frame = cap.read()
#        frame = cv2.resize(frame, (1280, 720))
#        if not ret:
#         break
#        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # detect faces available on camera
#        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#     # take each face available on the camera and Preprocess it
#        for (x, y, w, h) in num_faces:
#         # cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#         # predict the emotions
#         emotion_prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(emotion_prediction))
#         # cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#        emotionSeries.append(emotion_dict[maxindex])
# #     cv2.imshow('Emotion Detection', frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
#      return jsonify(emotionSeries)
















# def image_predict_emotion(file_path):
  

#    img = Image.open(file_path)

#      # nparr = np.fromstring(file_path.read(), np.uint8)
#      # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    

#      img = cv2.resize(np.array(img), (1280, 720))

     
#      face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#      gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # detect faces available on camera
#      num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    

#     # take each face available on the camera and Preprocess it
#      for (x, y, w, h) in num_faces:
#             #  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

#         # الاربع سطور دوول مسوولين عن تحديد الرقم من الديكشناري
#         #وبالتالي تحديد اسم الكلاس

#              roi_gray_frame = gray_frame[y:y + h, x:x + w]
#              cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#         # predict the emotions
#              emotion_prediction = emotion_model.predict(cropped_img)      #important هاااااااااااااااااااااام
#              maxindex = int(np.argmax(emotion_prediction))

#             #  emotionSeries.append( emotion_dict[maxindex])


#     #          cv2.putText(img, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#     #  cv2.imshow('Emotion Detection', img)

#     #  cv2.waitKey(0)

   



#     #  img.release()
#     #  cv2.destroyAllWindows()
    

#      return f"{emotion_dict[maxindex]}"

   











app=Flask(__name__)

  

@app.route('/predictttttt',methods=["POST","GET"])

def insert():
 return {"msg":"hello"}
 # if request.method=="POST":

 #  if 'file' in request.files:
  
 #     f=request.files.get('file')


 #     if (str(file_path).split("."))[-1].lower() in ['png','jpg','svg','igmp','JPEG']:
 #       return image_predict_emotion(file_path=f)
     


  
     




   



   



if __name__=="__main__":
    app.run(debug=True)

