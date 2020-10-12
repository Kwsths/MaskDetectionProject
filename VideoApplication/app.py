from flask import Flask, render_template, Response
from camera import Camera

app = Flask(__name__)

@app.route('/')
def index():
    # start the home page
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        # here we specify we return a frame and each frame an image with its specific image
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    # use generator functions in order to continuously return frame to the screen
    # we will use multipart responses based on the fact that we want to have live video
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
