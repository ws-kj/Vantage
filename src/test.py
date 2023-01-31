import cv2 as cv
import numpy as np

def pre_process(image, net):
    blob = cv.dnn.blobFromImage(image, 1/255, (640, 640), [0,0,0], 1, crop=False)
    net.setInput(blob)

    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs

def post_process(image, outputs):
    class_ids = []
    confidences = []
    boxes = []

    rows = outputs[0].shape[1]
    image_height, image_width = image.shape[:2]

    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        if confidence >= 0.5:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)

            if(classes_scores[class_id] > 0.45):
                confidences.append(confidence)
                class_ids.append(class_id)
                (cx, cy, w, h) = row[:4]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.45)
    for i in indices:
        box = boxes[i]
        (left, top, width, height) = box[:4]

        cv.rectangle(image, (left, top), (left+width, top+height), (0, 255, 0), 2)
        label = "{}:{:.2f}".format(names[class_ids[i]], confidences[i])
        draw_label(image, label, left, top)
    return image

def draw_label(im, label, x, y):
    text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    (dim, baseline) = text_size[:2]
    cv.rectangle(im, (x,y), (x+dim[0], y+dim[1]+baseline), (0,0,0), cv.FILLED)
    cv.putText(im, label, (x,y+dim[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)

if __name__ == '__main__':
    names = open("../data/coco.names").read().strip().split('\n')
    frame = cv.imread("../data/tennis.jpg")

    net = cv.dnn.readNet("../data/yolov5s.onnx")
    detections = pre_process(frame, net)
    img = post_process(frame.copy(), detections)

    cv.imshow('Output', img)
    cv.waitKey(0)
    
