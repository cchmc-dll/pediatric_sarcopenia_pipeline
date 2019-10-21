import webbrowser
from threading import Timer

from flask import Flask, render_template, send_file, url_for


def make_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_template('hello.html')

    @app.route("/predictions/<int:subject_id>")
    def prediction_show(subject_id):
        return render_template('predictions/show.html', subject_id=subject_id)

    @app.route("/predictions/<int:subject_id>/image.png")
    def prediction_image(subject_id):
        return send_file(
            '/Users/jamescastiglione/git/jac241/Muscle_Segmentation/tmp/test_out/images/31/subject_31_IM-CT-24.png',
            mimetype='image/png'
        )

    return app


def main():
    app = make_app()
    app.run()
