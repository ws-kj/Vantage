import cv2 as cv
import numpy as np
from dataclasses import dataclass

@dataclass
class Detection:
    box:   list[int]
    class_id:   int
    confidence: int

class Detector(object):
    def __init__(self, names, weights, 
        confidence=0.5, id_confidence=0.45, nms_confidence = 0.45):

        self.names = open(names).read().strip().split('\n')
        self.net = cv.dnn.readNet(weights)

        self.confidence = confidence
        self.id_confidence = id_confidence
        self.nms_confidence = nms_confidence
    
    def detect_frame(self, frame):
        blob = cv.dnn.blobFromImage(frame, 1/255, (640, 640), [0,0,0], 1, crop=False)
        self.net.setInput(blob)

        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        return self._proc_detections(frame.copy(), outputs)

    def _proc_detections(self, frame, outputs):
        class_ids = []
        confidences = []
        boxes = []

        rows = outputs[0].shape[1]
        image_height, image_width = frame.shape[:2]

        x_factor = image_width / 640
        y_factor = image_height / 640

        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]

            if confidence >=self.confidence:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)

                if(classes_scores[class_id] > self.id_confidence):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    (cx, cy, w, h) = row[:4]
                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_confidence)
        detections = []
        for i in indices:
            detections.append(Detection(boxes[i][:4], class_ids[i], confidences[i]))

            box = boxes[i]
            (left, top, width, height) = box[:4]

            cv.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 2)
            label = "{}:{:.2f}".format(self.get_name(class_ids[i]), confidences[i])
            draw_label(frame, label, left, top)

        return frame

    def get_name(self, class_id):
        if class_id >= len(self.names) or class_id < 0:
            return "unknown"
        return self.names[class_id]

def draw_label(im, label, x, y):
    text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    (dim, baseline) = text_size[:2]
    cv.rectangle(im, (x,y), (x+dim[0], y+dim[1]+baseline), (0,0,0), cv.FILLED)
    cv.putText(im, label, (x,y+dim[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)

if __name__ == '__main__':
#   names = open("../data/coco.names").read().strip().split('\n')
    frame = cv.imread("../data/tennis.jpg")

#   net = cv.dnn.readNet("../data/yolov5s.onnx")
#   detections = pre_process(frame, net)
#   img = post_process(frame.copy(), detections)

    model = Detector("../data/coco.names", "../data/yolov5s.onnx")
    img = model.detect_frame(frame)

    cv.imshow('Output', img)
    cv.waitKey(0)
    
