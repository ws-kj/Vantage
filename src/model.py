import cv2 as cv
import numpy as np
from dataclasses import dataclass

@dataclass
class ObjTarget:
    class_id:    int
    confidence:  float = 0.5
    count:       bool = False
    proximity:   bool = False
    passthrough: bool = False
    track:       bool = False

class ModelStream(object):
    yolo_names = None
    yolo = None
    yolo_out = None

    @staticmethod
    def init_model(ynames_p, yconfig_p, yweights_p):
        yolo_names = open(ynames_p).read().strip().split('\n')
        yolo = cv.dnn.readNetFromDarknet(yconfig_p, yweights_p)
        yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        yolo_out = [yolo.getLayerNames()[i[0]-1] 
            for i in yolo.getUnconnectedOutLayers()]

    @staticmethod
    def get_name(class_id):
        return ModelStream.yolo_names[class_id]

    def __init__(self, source, targets):
        self.source = source
        self.targets = targets

    def proc_frame(self, image=None):
        if image is not None:
            frame = image
        else:
            _, frame = self.source.read()
        frame = cv.resize(frame, (416, 416))

        blob = cv.dnn.blobFromImage(
            frame, 1.0, (416, 416), (0,0,0), swapRB=True, crop=False)

        ModelStream.yolo.setInput(blob)
        outs = ModelStream.yolo.forward(ModelStream.yolo_ol)

        boxes = []
        confs = []
        cids  = []

        w, h = frame.shape[:2]
        for out in outs:
            for detection in out:
                output = detection[5:]
                class_id = np.argmax(output)
                
                target = next((t for t in self.targets if t.class_id == class_id), None)
                if target == None:
                    break
                
                conf = output[class_id]
                if conf > target.confidence:
                    box = detection[:4] * np.array([w, h, w, h])
                    (box_cx, box_cy, box_w, box_h) = box.astype("int")
                    box_x = int(box_cx, (box_w / 2))
                    box_y = int(box_cy - (box_h / 2))
                    boxes.append([box_x, box_y, int(box_w), int(box_h)])
                    confs.append(float(conf))
                    cids.append(class_id)

        detection_boxes = []
        for fbox in cv.dnn.NMSBoxes(boxes, confs, 0.5, 0.4):
            i = fbox[0]
            detection_boxes.append((boxes[i][:4], confs[i], cids[i]))


                    

                    




