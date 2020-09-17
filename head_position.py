import cv2
import numpy as np

class HeadPosition:

    def __init__(self):
        # Load Yolo
        self.net = cv2.dnn.readNet("dnn_files/yolov4-custom4objects_5000.weights", "dnn_files/yolov4-custom4objects.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = ["left", "right", "front", "back"]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        #self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.colors = np.float64([
            [0, 255, 0], [255, 0, 0], [100, 0, 100], [0, 100, 100]
        ])


        self.confidence = 0.5


    def get_head_position(self, img):
        # Loading image
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        center_boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    center_boxes.append((center_x, center_y))
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)


        # Apply Non-maximum Suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Loop through the boxes again after Non-maximum suppression
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                center_x, center_y = center_boxes[i]
                # If a box was find, return it already
                return True, class_ids[i], (x, y, w, h), (center_x, center_y)

        # if it completes without detecting anything, it returns None
        return False, None, None, None
