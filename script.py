import cv2, time, pandas, numpy
from datetime import datetime

video = cv2.VideoCapture(0)
first_frame = None
status_list = [None, None]
change_list = []
data = pandas.DataFrame(columns = ['start', 'end'])

video.read()
time.sleep(2) #so the camera can properly register first_frame

while True:
    check, frame = video.read()
    status = 0
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
        
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        change_list.append(datetime.now())
    elif status_list[-1] == 0 and status_list[-2] == 1:
        change_list.append(datetime.now())


    # cv2.imshow('first', first_frame)
    cv2.imshow('Gray frame', gray_frame)
    cv2.imshow('Delta frame', df)
    cv2.imshow('Threshold', tf)
    cv2.imshow('Color frame', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1: 
            change_list.append(datetime.now())
        break

print(change_list)

for i in range(0, len(change_list), 2):
    data = data.append({'start': change_list[i], 'end':change_list[i + 1]}, ignore_index = True)

data.to_csv('motion-times.csv')

video.release()
cv2.destroyAllWindows()

    