import cv2, time

video = cv2.VideoCapture(0)
first_frame = None
video.read()
time.sleep(2) #so the camera can properly register first_frame

while True:
    check, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0) #easier to interpret

    if first_frame is None:
        first_frame = gray_frame
        print('Captured user background')
        continue

    df = cv2.absdiff(first_frame, gray_frame)
    tf = cv2.threshold(df, 30, 255, cv2.THRESH_BINARY)[1]
    tf = cv2.dilate(tf, None, iterations = 1)

    (cnts,_) = cv2.findContours(tf.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find d-outlines

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv2.imshow('first', first_frame)
    cv2.imshow('Gray frame', gray_frame)
    cv2.imshow('Delta frame', df)
    cv2.imshow('Threshold', tf)
    cv2.imshow('Color frame', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

    