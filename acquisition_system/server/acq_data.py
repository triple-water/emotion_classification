from flask import Flask, request, url_for, redirect, render_template

app = Flask(__name__, template_folder='../web/html', static_folder="../web", static_url_path="")
count = 0
video_list = ["demo1", "demo2"]


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


@app.route('/rest', methods=['POST','GET'])
def rest():
    global count
    video_name = video_list[count]
    return render_template('rest.html', video_name=video_name)


@app.route('/rest/<video_name>')
def video_name():
    # return render_template('rest.html')
    return redirect(url_for('video'))


@app.route('/video')
def video():
    print(request.args.get("v_name"))
    return render_template('video.html')


@app.route('/subjective_evaluation')
def subjective_evaluation():
    print("subjective_evaluation")
    return render_template('subjective_evaluation.html')


@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    user = request.form
    print(user)
    global count
    if count == len(video_list) - 1:
        return redirect(url_for('end'))
    elif count < len(video_list) - 1:
        count = count + 1
        return redirect(url_for('rest'))


@app.route('/end')
def end():
    global count
    count = 0
    return render_template('end.html')


if __name__ == '__main__':
    app.run(debug=True)
