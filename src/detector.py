import cv2 as cv
import numpy as np
import time
from dataclasses import dataclass

IMAGE_SIZE = 640
N_CLASSES = 80

@dataclass
class Target:
    class_id: int
    conf:     float
    prox:     float
    time:     int
    cooldown: int

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
 
        self.targets  = [] #detection confs
        self.t_frames = [0] * N_CLASSES  #frame count of active detections
        self.t_active = [0] * N_CLASSES  #quick lookup for detections
        self.t_cds    = [0] * N_CLASSES #cooldown lookup
    
    def add_target(self, target, callback):
        if self.t_active[target.class_id] == 0:
            self.targets.append((target, callback))
            self.t_active[target.class_id] = 1

    def start_capture(self, cap):
        if len(self.targets) == 0 or not cap.isOpened():
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.detect_frame(frame)

    def detect_frame(self, frame):
        blob = cv.dnn.blobFromImage(frame, 1/255, (IMAGE_SIZE, IMAGE_SIZE), [0,0,0], 1, crop=False)
        self.net.setInput(blob)

        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        return self._proc_detections(frame.copy(), outputs)

    def _proc_detections(self, frame, outputs):
        class_ids = []
        confidences = []
        boxes = []

        rows = outputs[0].shape[1]
        image_height, image_width = frame.shape[:2]

        x_factor = image_width / IMAGE_SIZE
        y_factor = image_height / IMAGE_SIZE

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
        detected_ids = []
        for i in indices:
            cid = class_ids[i]
            detected_ids.append(cid)
            if self.t_active[cid] == 1:
                print("target detected: " + self.get_name(cid) + ", cid: " + str(cid))
                self.t_frames[cid] += 1
            detections.append(Detection(boxes[i][:4], class_ids[i], confidences[i]))

            box = boxes[i]
            (left, top, width, height) = box[:4]

            cv.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 2)
            label = "{}:{:.2f}".format(self.get_name(class_ids[i]), confidences[i])
            draw_label(frame, label, left, top)

        inactive = [i for i, x in enumerate(self.t_active) if x==1 ]
        for cid in inactive:
            if cid not in detected_ids:
                print(cid)
                self.t_frames[cid] = 0

        self.try_callbacks(detections)

        return frame

    def try_callbacks(self, detections):  
        current = [i for i, x in enumerate(self.t_active) if x == 1]
        for c in current:
            target, callback = [t for t in self.targets if t[0].class_id == c][0]
            if self.t_frames[c] >= target.time and time.time()-self.t_cds[c] >= target.cooldown:  
                callback([d for d in detections if d.class_id == c])
                self.t_cds[c] = time.time()
                
    def get_name(self, class_id):
        if class_id >= len(self.names) or class_id < 0:
            return "unknown"
        return self.names[class_id]

    def get_cid(self, name):
        try:
            return self.names.index(name)
        except ValueError:
            return 0

def draw_label(im, label, x, y):
    text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    (dim, baseline) = text_size[:2]
    cv.rectangle(im, (x,y), (x+dim[0], y+dim[1]+baseline), (0,0,0), cv.FILLED)
    cv.putText(im, label, (x,y+dim[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
