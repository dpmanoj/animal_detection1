from detection import detection
import cv2
import json
import time

detector = detection()

# async def sendData(frame):
#     image_w, image_h, _ = frame.shape
#     image = frame.reshape(1, image_w, image_h, 3)
#     data = json.dumps({
#         "signature_name": "serving_default",
#         "instances": image.tolist()
#     })
#     await detector.predict(data)


video_path = "cow.mp4"
video = cv2.VideoCapture(video_path)

# video = cv2.VideoCapture("rtsp://anshan:A_shan11@192.168.225.50:554/Streaming/channels/301")

while video.isOpened():

    ret, frame = video.read()

    if ret:

        image_w, image_h, _ = frame.shape
        image = frame.reshape(1, image_w, image_h, 3)
        data = json.dumps({
            "signature_name": "serving_default",
            "instances": image.tolist()
        })

        # sendData(frame)
        # asyncio.run(detector.predict(data))
        boxes, scores, classe = detector.predict(data)
        # print(type(classe[0]))
        # new_id = detector.track(boxes)
        # detector.visual(frame, boxes, classe, scores, new_id)
        # detector.visual(frame, boxes, classe, scores)
        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


# time.sleep(10)
# When everything done, release the capture
video.release()
cv2.destroyAllWindows()