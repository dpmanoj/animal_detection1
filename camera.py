import cv2

cap = cv2.VideoCapture("rtsp://anshan:A_shan11@192.168.225.50:80/ISAPI/Streaming/channels/301")

while (cap.isOpened()):
    ret , frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


