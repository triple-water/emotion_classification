from flask import Flask, request, url_for, redirect, render_template
from pathlib import Path
from acquisition_system.emotiv_api.cortex import Cortex
import time
import datetime

app = Flask(__name__, template_folder='../web/html', static_folder="../web", static_url_path="")
count = 0
v_name = ''
base_path = Path("../web/source/video")
hdf_path_list = base_path.glob("*.mp4")
video_list = [Path(video_name).stem for video_name in hdf_path_list]

user = {
    "license": "30a7df02-37ba-462c-8b75-feedfee8c815",
    "client_id": "mm6xyIddErBCPW8lZFlAlG5k0f771wTzuvDT37fC",
    "client_secret": "AmGvdjxk17EgLOwYbmJosCch0JfXevUNDSc3LPne0gKbtKFbO89K5x5nphIQd6JJ5yewvHiXhhSdoHUZd7uceXZeHsdXdoZQlWUsMqEnUltome8bVKScgBRNZEQOpZ6D",
    "debit": 100
}
record_name = 'demo_marker' + str(datetime.datetime.now().strftime("%Y-%m-%d"))
record_description = 'gm_test'
export_folder = r'C:\Users\A\Desktop\test'
class Marker():
    def __init__(self):
        self.c = Cortex(user, debug_mode=True)
        self.c.do_prepare_steps()


    def inject_name_marker(self, name: str):
        marker_time = time.time() * 1000
        marker = {
            "label": name,
            "value": str(0),
            "port": "python-app",
            "time": marker_time
        }
        self.c.inject_marker_request(marker)

    def inject_end_marker(self,record_export_folder):
        marker_time = time.time() * 1000
        marker = {
            "label": 'end',
            "value": str(0),
            "port": "python-app",
            "time": marker_time
        }
        self.c.inject_marker_request(marker)
        self.c.stop_record()
        self.c.disconnect_headset()
        record_export_data_types = ['EEG', 'MOTION', 'PM', 'BP']
        record_export_format = 'CSV'
        record_export_version = 'V2'
        self.c.export_record(record_export_folder,
                             record_export_data_types,
                             record_export_format,
                             record_export_version,
                             [self.c.record_id])

    def inject_video_start(self, video_name):
        marker_time = time.time() * 1000
        marker = {
            "label": video_name + '_start',
            "value": video_name,
            "port": "python-app",
            "time": marker_time
        }
        self.c.inject_marker_request(marker)

    def inject_sub_eval(self, video_name):
        marker_time = time.time() * 1000
        marker = {
            "label": 'sub_eval_' + video_name + '_start',
            "value": video_name,
            "port": "python-app",
            "time": marker_time
        }
        self.c.inject_marker_request(marker)

    def start_record(self, name, description):
        self.c.create_record(name, description)


m = Marker()


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
    m.start_record(record_name,record_description)
    return render_template('rest_info.html')


@app.route('/rest', methods=['POST', 'GET'])
def rest():
    global count

    if request.method == 'GET'or'POST':
        m.inject_name_marker('start')
    # TODO: marker: start
    # TODO: first time: post start
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
    # todo: marker: video_name_start
    # print(request.args.get("v_name"))
    return render_template('video.html')


@app.route('/subjective_evaluation')
def subjective_evaluation():
    m.inject_sub_eval(v_name)
    # todo: marker: sub_eval_video_name_start
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
    m.inject_end_marker(export_folder)
    # todo: marker: end
    global count
    count = 0
    return render_template('end.html')


if __name__ == '__main__':
    app.run(debug=True)
