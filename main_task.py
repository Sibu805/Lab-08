import cv2
import time

def image_processing():

    img = cv2.imread('variant-9.png')
    cv2.imshow('variant-9.png', img)
    temp = img.copy()
    for i in range(3): 
        temp = cv2.pyrDown(temp)  
        cv2.imshow(f'Pyramid Level {i+1}', temp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_processing():

    cap = cv2.VideoCapture(0)  
    down_points = (640, 480) 
    i = 0 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

        contours, hierarch = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours):
            c = max(contours, key=cv2.contourArea)  
            x, y, w, h = cv2.boundingRect(c) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

            if i % 5 == 0:  
                a = x + (w // 2)  
                b = y + (h // 2)  
                print(f'Tracked Coordinate: ({a}, {b})')

        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)  
        i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_processing()  
    video_processing()  
