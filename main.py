from predict import Predict

if __name__ == '__main__':
    pred = Predict()
    img = pred.detect_image('C:/AllProgram/testimage/faceRecognition/four.jpg')
    img.show()
