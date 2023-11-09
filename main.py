from detection import detection
import cv2

detector = detection()

video_path = "test/data/cow.mp4"
video = cv2.VideoCapture(video_path)

while video.isOpened():

    ret, frame = video.read()

    if ret:

        boxes, scores, classe = detector.predict(frame)
        new_id = detector.track(boxes)
        detector.visual(frame, boxes, classe, scores, new_id)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()