import cv2 as cv
from model import ObjTarget, ModelStream

def vantage_main(debug=False):
    ModelStream.init_model(
        "../data/yolov3names.txt", 
        "../data/yolov3.cfg", 
        "../data/yolov3.weights"
    )

    targets = [
        ObjTarget(ModelStream.get_cid("person"))
    ]
    model = ModelStream(None, targets)

    frame = cv.imread("../data/person.jpg")
    frame = cv.resize(frame, (416, 416))
    model.proc_frame(image=frame)

    for det in model.current_detections:
        cid = det[0]
        (x, y, w, h) = det[1:4]
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = "{} ({}): {:.4f}".format(det[0], cid, det[2])
        print(text)
        cv.putText(
            frame, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 255, 0), 1)

    cv.imshow("frame", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    vantage_main(True)