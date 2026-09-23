import cv2

cap = cv2.VideoCapture("test_video.mp4")
print("Opened:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()