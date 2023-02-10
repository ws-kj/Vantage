import cv2 as cv
from detector import Detector, Target

def test_callback(detections):
    print("[callback] detection class_id: ", detections[0].class_id)

def vantage_main(debug=False):
    frame = cv.imread("../data/tennis.jpg")
    model = Detector("../data/coco.names", "../data/yolov5s.onnx")

    model.add_target(Target(model.get_cid("tennis racket"), 0.5, 0.5, 1, 48), test_callback)

    img = model.detect_frame(frame)

    #cv.imshow('Output', img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

if __name__ == "__main__":
    vantage_main(True)