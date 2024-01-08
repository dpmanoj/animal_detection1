from detection import detection
import cv2
import json
import time

detector = detection()

video_path = "[ Path to video you want to detect]"
video = cv2.VideoCapture(video_path)

while video.isOpened():

    ret, frame = video.read()

    if ret:

        image_w, image_h, _ = frame.shape
        image = frame.reshape(1, image_w, image_h, 3)

        ## below code is required if your model is hosted in another server
        ## it does preprocessing for you before sending it throung http request 
        
        # image = json.dumps({
        #     "signature_name": "serving_default",
        #     "instances": image.tolist()
        # })

        boxes, scores, classe = detector.predict(image)
        detector.visual(frame, boxes, classe, scores)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()