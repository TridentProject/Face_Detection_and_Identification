from flask import Flask, render_template, request
import FaceDetectionModel
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(),"videos")


@app.route('/')
def index():
        return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
	if request.method == 'POST':

		video= request.files['video']
		
		filename = secure_filename(video.filename)
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(filename))
		video.save(file_path)
		

			
		database = FaceDetectionModel.setup_database()
		detected = FaceDetectionModel.run_face_recognition(database,file_path)
		
		return render_template("prediction.html",data = detected)
	else:
		return render_template("prediction.html",data = {'Empty':'File not found'})



if __name__ == "__main__":
	app.run(debug=True)