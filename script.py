import cv2, time

video = cv2.VideoCapture(0)
first_frame = None

time.sleep(5)
while True:
    check, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0) #easier to interprey

    if first_frame is None:
        first_frame = gray_frame
        continue

    df = cv2.absdiff(first_frame, gray_frame)

    # cv2.imshow('first', first_frame)

    cv2.imshow('Gray frame', gray_frame)
    cv2.imshow('Delta frame', df)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

    