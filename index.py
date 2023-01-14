from flask import Flask, render_template, request
import cgi, os
import cgitb; cgitb.enable()
import matplotlib.image as mpimg
import numpy as np
from algo import predict, predictH,textcolor
app = Flask(__name__)
import os
@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/pred',methods = ['POST', 'GET'])
def analysis():
	if request.method == 'POST':
         image = request.files['myfile']
         save_path = os.path.join("static",image.filename)
         image.save(save_path)
         return render_template('analysis.html', image_name='static/' +image.filename,name=predict(save_path),titre="Descripteur avec texture")

	#UPLOAD_DIR="C:\\Users\\user\\Desktop\\hna\\tmp"
	#if request.method == 'POST':
	#	newfile = request.files.get('myfile')
	#	save_path = os.path.join(UPLOAD_DIR, newfile.filename)
	#	newfile.save(save_path)
       

	#return render_template('analysis.html',name=predict(save_path), image=save_path)

@app.route('/test',methods = ['POST', 'GET'])
def test():
	if request.method == 'POST':
         image = request.files['myfile']
         save_path = os.path.join("static",image.filename)
         image.save(save_path)
	return render_template('analysis.html',image_name='static/' +image.filename,name=predictH(save_path),titre="Descripteur avec couleur")

@app.route('/merge',methods = ['POST', 'GET'])
def merge():	
	if request.method == 'POST':
         image = request.files['myfile']
         save_path = os.path.join("static",image.filename)
         image.save(save_path)
	return render_template('analysis.html',image_name='static/' +image.filename,name=textcolor(save_path),titre="Descripteur avec merge")
if __name__=="__main__":
	app.run(debug=True,host='0.0.0.0', port=5000)