import cv2
import numpy as np
from head_position import HeadPosition
import matplotlib.pyplot as plt
import math

cap = cv2.VideoCapture("sergio_sleeping.avi")

hp = HeadPosition()
classes = ["Left", "Right", "Front", "Back"]

# Track head information
head_position = [0, 0, 0, 0]
head_distance_movement = []
total_frames = 0
center_prev = (0, 0)

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    # Get head position
    ret, class_id, box, center = hp.get_head_position(frame)
    if ret:
        x, y, w, h = box
        cv2.putText(frame, classes[class_id], (x, y - 15), 0, 1.3, hp.colors[class_id], 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(frame, center, 5, hp.colors[class_id], 3)

        # Update head information
        head_position[class_id] += 1
        total_frames += 1

        # get center head movement
        x, y = center
        distance = math.hypot(x - center_prev[0], y - center_prev[1])
        head_distance_movement.append(distance)

        # Store current frame center
        center_prev = center



    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord("s"):
        cv2.imwrite("rectangle_head.jpg", frame)

# Plot sleeping information
head_position_hours = [x/3600 for x in head_position]
head_distance_hours = [x/3600 for x in range(len(head_distance_movement))]

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.bar(classes, head_position_hours)
ax2.plot(head_distance_hours, head_distance_movement)
plt.show()

cap.release()
cv2.destroyAllWindows()