import os
import cv2
from pandas import np, datetime
from tensorflow.keras.models import load_model
from base_camera import BaseCamera


class Camera(BaseCamera):
    # make the necessary initializations
    # this defines the size of the image we will keep in order to make use CNN model to classify. the bigger this is the greater the image
    size = 2
    # load CNN model for mask recognition
    model = load_model('../cnn_model.h5')
    # import classifier for face recognition
    classifier = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
    labels_dict = {0: 'with_mask', 1: 'without_mask'}
    color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
    label = ''
    frame_number = 0

    # initial local camera
    def __init__(self):
        super(Camera, self).__init__()

    @staticmethod
    def frames():
        # open camera and start recording
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        log = open(r"log.txt", "w")
        log.write("Video recoding started at " + str(datetime.now()) + "\n")
        log.write("--------------------------------------------------------------------------------\n")
        log.close()

        while True:
            # read current frame
            success, img = camera.read()

            log = open(r"log.txt", "a")
            Camera.frame_number += 1
            log.write("Frame " + str(Camera.frame_number) + ":\n")

            # resize image to speed up detection
            mini = cv2.resize(img, (img.shape[1] // Camera.size + 3, img.shape[0] // Camera.size + 3))

            # detect faces using the classifier imported above
            faces = Camera.classifier.detectMultiScale(mini)
            log.write("Faces found: " + str(len(faces)) + "\n")
            for f in faces:
                (x, y, w, h) = [v * Camera.size for v in f]  # Scale the shapesize backup
                # keep only the piece of image that contains the face
                face_img = img[y:y + h, x:x + w]

                # resize, normalize and remove colors from image in order to be able to use our trained model
                resized = cv2.resize(face_img, (300, 300))
                normalized = resized / 255.0
                grayscale = np.dot(normalized[..., :3], [0.2989, 0.5870, 0.1140])
                reshaped = np.reshape(grayscale, (1, 300, 300, 1))

                # make prediction/ classification
                result = Camera.model.predict(reshaped, batch_size=10)

                # decide result
                if result[0] > 0.5:
                    label = 1
                else:
                    label = 0

                # based on the prediction, add a green rectangle or red rectangle around individuals face with the corresponding label
                cv2.rectangle(img, (x, y), (x + w, y + h), Camera.color_dict[label], 2)
                cv2.rectangle(img, (x, y - 40), (x + w, y), Camera.color_dict[label], -1)
                cv2.putText(img, Camera.labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),2)

                log.write("Face " + str(f) + ": " + str(Camera.labels_dict[label]) + "\n")

            # take current datetime and insert it on live video and log
            dt = str(datetime.now())
            img = cv2.putText(img, dt, (5, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0), 4, cv2.LINE_8)
            log.write("Record Time: " + str(dt) + "\n")
            log.write("------------------------------------------------------\n")
            log.close()

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()