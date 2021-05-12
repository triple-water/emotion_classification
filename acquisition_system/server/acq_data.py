from flask import Flask, request, url_for, redirect, render_template
from pathlib import Path
import acquisition_system.emotiv_api.my_marker as my_marker
import time, os
import datetime
import acquisition_system.tools.file_tools as file_tools

app = Flask(__name__, template_folder='../web/html', static_folder="../web", static_url_path="")
count = 0
v_name = ''
base_path = Path("../web/source/video")
hdf_path_list = base_path.glob("*.mp4")
video_list = [Path(video_name).stem for video_name in hdf_path_list]
save_path = "../acq_data"
data_path = ""
sub_info_file_name = "sub_info.txt"
subject_info_file_name = "subject_info.txt"
record_name = 'demo_marker' + str(datetime.datetime.now().strftime("%Y-%m-%d"))
record_description = 'gm_test'
export_folder = r'C:\Users\A\Desktop\test'

m = my_marker.Marker()


@app.route('/')
def index():
    return render_template('info_view.html')


@app.route('/subject_info')
def info_page():
    return render_template('subject_info.html')


@app.route('/submit_info', methods=['POST'])
def subject_info():
    sub_info_dict = request.form
    content = "age: {}\n" \
              "sex: {}\n" \
              "education: {}\n".format(sub_info_dict["age"], sub_info_dict["sex"],
                                       sub_info_dict["education"])
    global data_path
    data_path = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_tools.write_content_overlap(os.path.join(save_path, data_path), sub_info_file_name, content)
    return redirect(url_for('rest_info'))


@app.route('/rest_info')
def rest_info():
    m.start_record(record_name, record_description)
    return render_template('rest_info.html')


@app.route('/rest', methods=['POST', 'GET'])
def rest():
    global count

    if request.method == 'GET' or 'POST':
        m.inject_name_marker('start')
    video_name = video_list[count]
    return render_template('rest.html', video_name=video_name)


@app.route('/rest/<video_name>')
def video_name():
    # return render_template('rest.html')
    return redirect(url_for('video'))


@app.route('/video')
def video():
    global v_name
    v_name = request.args.get("v_name")
    m.inject_video_start(v_name)
    return render_template('video.html')


@app.route('/subjective_evaluation')
def subjective_evaluation():
    m.inject_sub_eval(v_name)
    print("subjective_evaluation")
    return render_template('subjective_evaluation.html')


@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    sub_eva_info = request.form
    sub_eva_content = "video name: {}\n" \
                      "emotion: {}".format(v_name, sub_eva_info["emotion"])
    file_tools.write_content_add(os.path.join(save_path, data_path), subject_info_file_name, sub_eva_content)
    global count
    if count == len(video_list) - 1:
        return redirect(url_for('end'))
    elif count < len(video_list) - 1:
        count = count + 1
        return redirect(url_for('rest'))


@app.route('/end')
def end():
    m.inject_end_marker(export_folder)
    # todo: marker: end
    global count
    count = 0
    return render_template('end.html')


if __name__ == '__main__':
    app.run(debug=True)
