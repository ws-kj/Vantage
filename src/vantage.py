import cv2 as cv
from detector import Detector, Target

def person_callback():
    print("person")

def vantage_main(debug=False):
    frame = cv.imread("../data/tennis.jpg")
    model = Detector("../data/coco.names", "../data/yolov5s.onnx")

    model.add_target(Target(model.get_cid("person"), 0.5, 0.5, 1), person_callback)

    img = model.detect_frame(frame)

    #cv.imshow('Output', img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

if __name__ == "__main__":
    vantage_main(True)