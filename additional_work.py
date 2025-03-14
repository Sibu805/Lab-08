''' код использует камеру для захвата поверхности с надписью и реализации алгоритма ее отслеживания.
    он берет изображение мухи (fly 64.png) и накладывает его на программную рамку из шага 2 таким образом, чтобы центр мухи совпадал с центром метки.'''

import cv2

def video_processing():
    obj_img = cv2.imread('variant-9.png', cv2.IMREAD_GRAYSCALE)
    fly_img = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)

    obj_height, obj_width = obj_img.shape
    fly_height, fly_width, _ = fly_img.shape

    total_x, total_y, frame_count = 0, 0, 0
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
                center_x = adjusted_top_left[0] + w // 2
                center_y = adjusted_top_left[1] + h // 2
            else:
                center_x = top_left[0] + obj_width // 2
                center_y = top_left[1] + obj_height // 2
        else:
            cv2.putText(frame, "No match found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Object Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  

        total_x += center_x
        total_y += center_y
        frame_count += 1

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        fly_x = center_x - fly_width // 2
        fly_y = center_y - fly_height // 2

        if 0 <= fly_x < frame.shape[1] - fly_width and 0 <= fly_y < frame.shape[0] - fly_height:
            musk = fly_img[:, :, 3] / 255.0  
            for c in range(3):  
                frame[fly_y:fly_y + fly_height, fly_x:fly_x + fly_width, c] = (
                    fly_img[:, :, c] * musk + frame[fly_y:fly_y + fly_height, fly_x:fly_x + fly_width, c] * (1 - musk)
                )

        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if frame_count > 0:
        avg_x = total_x / frame_count
        avg_y = total_y / frame_count
        print(f"Average coordinates: ({round(avg_x, 2)}, {round(avg_y, 2)})")
    else:
        print("No valid matches found")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_processing()
