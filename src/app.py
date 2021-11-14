import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from svd import compress

app = Flask(__name__)

upload_folder = "static/uploads/"
download_folder = "static/downloads/"
if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)
if not os.path.exists(download_folder):
    os.mkdir(download_folder)
app.secret_key = b'mantap jiwa'
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['DOWNLOAD_FOLDER'] = download_folder


extensions = ['jpg', 'png', 'jpeg']

def check_extension(filename):
    return filename.split('.')[-1] in extensions
def get_extension(filename):
    return filename.split('.')[-1]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('File belum dimasukkan.')
            return redirect(request.url)
        elif check_extension(file.filename):
            filename = secure_filename(file.filename)
            num = request.form['number']
            if (num == '') :
                flash('Persentase kompresi belum dimasukkan.')
                return redirect(request.url)
            else:
                num = int(num)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                percentage, runtime = compress(filename, num)
                percentage = float("{:.2f}".format(percentage))
                runtime = float("{:.2f}".format(runtime))
                return render_template('index.html',filename=filename, percentage=percentage,tcomp=runtime)
        else :
            flash('Format file tidak diperbolehkan.')
            return redirect(request.url)

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run()