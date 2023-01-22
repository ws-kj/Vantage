import cv2 as cv
import numpy as np
from dataclasses import dataclass

@dataclass
class ObjTarget:
    class_id:    int
    confidence:  float = 0.5
    presence_f:  int = 5
    cooldown:    int = 0
    count:       bool = False
    proximity:   bool = False
    passthrough: bool = False
    track:       bool = False


class ModelStream(object):
    yolo_names = None
    yolo = None
    yolo_out = None

    @staticmethod
    def init_model(ynames, yconfig, yweights):
        ModelStream.yolo_names = open(ynames).read().strip().split('\n')
        ModelStream.yolo = cv.dnn.readNetFromDarknet(yconfig, yweights)
        ModelStream.yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        ln = ModelStream.yolo.getLayerNames()
        ModelStream.yolo_out = [ln[i - 1] 
            for i in ModelStream.yolo.getUnconnectedOutLayers()]

    @staticmethod
    def get_name(class_id):
        if class_id >= len(ModelStream.yolo_names) or class_id < 0:
            return "unknown"
        return ModelStream.yolo_names[class_id]

    @staticmethod
    def get_cid(name):
        try:
            return ModelStream.yolo_names.index(name)
        except ValueError as e:
            return -1

    def __init__(self, source, targets):
        self.source = source
        self.targets = targets
        self.current_detections = []

    def proc_frame(self, image=None, debug=False):
        if image is not None:
            frame = image
        else:
            _, frame = self.source.read()
        frame = cv.resize(frame, (416, 416))

        blob = cv.dnn.blobFromImage(
            frame, 1.0, (416, 416), (0,0,0), swapRB=True, crop=False)

        ModelStream.yolo.setInput(blob)
        outs = ModelStream.yolo.forward(ModelStream.yolo_out)

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
                    box_x = int(box_cx - (box_w / 2))
                    box_y = int(box_cy - (box_h / 2))
                    boxes.append([box_x, box_y, int(box_w), int(box_h)])
                    confs.append(float(conf))
                    cids.append(class_id)

        #detections = {}
        post_nms = cv.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        if len(post_nms) > 0:
            for i in post_nms.flatten():
                self.current_detections.append(
                    (cids[i], ModelStream.get_name(cids[i]), boxes[i][:4], confs[i])
                )

        if debug:
            pass



        


                    

                    




