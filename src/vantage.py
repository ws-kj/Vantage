import cv2 as cv
from detector import Detector

def vantage_main(debug=False):
    
    frame = cv.imread("../data/tennis.jpg")
    
    model = Detector("../data/coco.names", "../data/yolov5s.onnx")
    img = model.detect_frame(frame)

    cv.imshow('Output', img)
    cv.waitKey(0)

    cv.imshow("frame", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    vantage_main(True)