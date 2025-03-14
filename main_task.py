'''код отображает пирамиду изображений
   код использует камеру для захвата поверхности с надписью и реализации алгоритма ее отслеживания.
   он выводит на консоль среднюю координату для текущего сеанса работы программы.'''

import cv2

def image_processing():

    layer = img.copy()
    gp = [layer]

    for i in range(4):
        layer = cv2.pyrDown(layer)
        gp.append(layer)
        cv2.imshow(str(i), layer)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_processing():

    obj_img = cv2.imread('variant-9.png', cv2.IMREAD_GRAYSCALE)
    obj_height, obj_width = obj_img.shape

    total_x, total_y, count = 0, 0, 0
    similarity_thres = 0.2  

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_frame, obj_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= similarity_thres:
            top_left = max_loc
            bottom_right = (top_left[0] + obj_width, top_left[1] + obj_height)

            roi = gray_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            _, binary_roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                adjusted_top_left = (top_left[0] + x, top_left[1] + y)
                adjusted_bottom_right = (adjusted_top_left[0] + w, adjusted_top_left[1] + h)
                
                #Вычислите центр ограничивающего прямоугольника
                center_x = adjusted_top_left[0] + w // 2
                center_y = adjusted_top_left[1] + h // 2
                total_x += center_x
                total_y += center_y
                count += 1

                #нарисуйте прямоугольник и обведите соответствующую область
                cv2.rectangle(frame, adjusted_top_left, adjusted_bottom_right, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            else:
                center_x = top_left[0] + obj_width // 2
                center_y = top_left[1] + obj_height // 2

                total_x += center_x
                total_y += center_y
                count += 1
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        else:
            cv2.putText(frame, "No match found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if count > 0:
        avg_x = total_x / count
        avg_y = total_y / count
        print(f"Average coordinates: ({round(avg_x, 2)}, {round(avg_y, 2)})")
    else:
        print("No valid matches found")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    img = cv2.imread('variant-9.png')
    image_processing()  
    video_processing()  
