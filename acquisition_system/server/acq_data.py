from flask import Flask, request, url_for, redirect, render_template

app = Flask(__name__, template_folder='../web/html', static_folder="../web", static_url_path="")


@app.route('/')
def index():
    return render_template('info_view.html')


@app.route('/subject_info')
def info_page():
    return render_template('subject_info.html')


@app.route('/submit_info', methods=['POST'])
def subject_info():
    user = request.form
    print(user)
    return redirect(url_for('rest_info'))


@app.route('/rest_info')
def rest_info():
    return render_template('rest_info.html')


@app.route('/rest')
def rest():
    return render_template('rest.html')


@app.route('/video')
def video():
    return render_template('video.html')


@app.route('/subjective_evaluation')
def subjective_evaluation():
    return render_template('subjective_evaluation.html')


@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    user = request.form
    print(user)
    return redirect(url_for('end'))


@app.route('/end')
def end():
    return render_template('end.html')


if __name__ == '__main__':
    app.run(debug=True)
