import cv2
import numpy as np
import imutils


vs = cv2.VideoCapture("videos/match.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    grabbed, frame = vs.read()
    if not grabbed:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_red = np.array([0, 31, 255])
    upper_red = np.array([176, 255, 255])

    lower_white = np.array([0, 0, 0])
    upper_white = np.array([0, 0, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((13, 13), np.uint8)
    thresh = cv2.threshold(
        res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= 1.5 * w and w > 15:
            player_bgr = frame[y:y + h, x:x + w]
            player_hsv = cv2.cvtColor(player_bgr, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
            res1 = cv2.bitwise_and(player_hsv, player_hsv, mask=mask1)
            res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
            nz_count = cv2.countNonZero(res1)

            mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
            res2 = cv2.bitwise_and(player_hsv, player_hsv, mask=mask2)
            res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
            res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
            nz_countred = cv2.countNonZero(res2)

            if nz_count >= 20 or nz_countred >= 20:
                label = "France" if nz_count >= 20 else "Belgium"
                cv2.putText(frame, label, (x - 2, y - 2), font, 0.8,
                            (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        if h > 1 and w > 1 and h < 30 and w < 30:
            ball_bgr = frame[y:y + h, x:x + w]
            ball_hsv = cv2.cvtColor(ball_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(ball_hsv, lower_white, upper_white)
            res = cv2.bitwise_and(ball_hsv, ball_hsv, mask=mask)
            res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            nz_count = cv2.countNonZero(res)
            if nz_count > 3:
                cv2.putText(frame, "ball", (x - 2, y - 2), font, 0.8,
                            (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow("detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
