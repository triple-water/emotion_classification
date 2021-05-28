import time, os, datetime, threading
from queue import Queue
from pathlib import Path
import jinja2.exceptions as exceptions
from flask import Flask, request, url_for, redirect, render_template, abort, jsonify
import acquisition_system.tools.file_tools as file_tools
import acquisition_system.emotiv_api.my_marker as my_marker
import acquisition_system.tools.logs.log_auto as logs
import acquisition_system.tools.logs.log_tool as log_tool
# import malin2.midi_generator as midi_generator
import demo_malin.generate as generate
import classification.get_emotion as get_emotion
import numpy as np
from pylsl import StreamInlet, resolve_stream

# config
base_path = Path("../web/source/video")
save_path = "../acq_data"
save_path = os.path.join(save_path, time.strftime("%Y%m%d%H%M%S", time.localtime()))
# global parameters
app = Flask(__name__, template_folder='../web/html', static_folder="../web", static_url_path="")
count = 0
v_name = ''
video_list = [Path(video_name).stem for video_name in base_path.glob("*.mp4")]
sub_info_file_name = "sub_info.txt"
subject_info_file_name = "subject_info.txt"
record_name = 'test_marker' + str(datetime.datetime.now().strftime("%Y-%m-%d"))
record_description = 'test'
logger = log_tool.get_logger(log_file_name="acq_data")

m = my_marker.Marker()

emotion_class = None
eeg_queue = Queue(maxsize=0)
time_count = 0


# class Marker():
#     def inject_name_marker(self, name: str): pass
#
#     def inject_end_marker(self, record_export_folder): pass
#
#     def inject_video_start(self, video_name): pass
#
#     def inject_sub_eval(self, video_name): pass
#
#     def start_record(self, name, description): pass
#
#
# m = Marker()


@app.route('/')
@logs.log_record(logger)
def index():
    try:
        return render_template('info_view.html')
    except exceptions.TemplateNotFound as et:
        logger.error(et.message)
        abort(404)
    except Exception as e:
        logger.error(e)
        abort(500)


@app.route('/subject_info')
@logs.log_record(logger)
def info_page():
    try:
        return render_template('subject_info.html')
    except exceptions.TemplateNotFound as et:
        logger.error(et.message)
        abort(404)
    except Exception as e:
        logger.error(e)
        abort(500)


@app.route('/submit_info', methods=['POST'])
@logs.log_record(logger)
def subject_info():
    try:
        sub_info_dict = request.form
        content = "age: {}\n" \
                  "sex: {}\n" \
                  "education: {}\n".format(sub_info_dict["age"], sub_info_dict["sex"],
                                           sub_info_dict["education"])
        logger.info(count)
        file_tools.write_content_overlap(save_path, sub_info_file_name, content)
        return redirect(url_for('rest_info'))
    except exceptions.TemplateNotFound as et:
        logger.error(et.message)
        abort(404)
    except Exception as e:
        logger.error(e)
        abort(500)


@app.route('/rest_info')
@logs.log_record(logger)
def rest_info():
    try:
        m.start_record(record_name, record_description)
        return render_template('rest_info.html')
    except exceptions.TemplateNotFound as et:
        logger.error(et.message)
        abort(404)
    except Exception as e:
        logger.error(e)
        abort(500)


@app.route('/rest', methods=['POST', 'GET'])
@logs.log_record(logger)
def rest():
    try:
        global count
        if request.method == 'GET' or 'POST':
            m.inject_name_marker('start')
        video_name = video_list[count]
        t_product = threading.Thread(target=product)
        t_consume = threading.Thread(target=consume)
        t_product.daemon = True
        t_consume.daemon = True
        t_product.start()
        t_consume.start()
        eeg_queue.join()
        logger.info("video name: {}".format(v_name))
        return render_template('rest.html', video_name=video_name)
    except exceptions.TemplateNotFound:
        abort(404)
    except Exception as e:
        abort(500)


@app.route('/rest/<video_name>')
@logs.log_record(logger)
def video_name():
    try:
        return redirect(url_for('video'))
    except Exception as e:
        abort(500)


@app.route('/video')
# @logs.log_record(logger)
def video():
    try:
        global v_name
        v_name = request.args.get("v_name")
        m.inject_video_start(v_name)
        return render_template('video.html')
    except exceptions.TemplateNotFound as et:
        # logger.error(et.message)
        abort(404)
    except Exception as e:
        # logger.error(e)
        abort(500)


@app.route('/subjective_evaluation')
@logs.log_record(logger)
def subjective_evaluation():
    try:
        m.inject_sub_eval(v_name)
        return render_template('subjective_evaluation.html')
    except exceptions.TemplateNotFound as et:
        logger.error(et.message)
        abort(404)
    except Exception as e:
        logger.error(e)
        abort(500)


@app.route('/submit_evaluation', methods=['POST'])
@logs.log_record(logger)
def submit_evaluation():
    try:
        sub_eva_info = request.form
        sub_eva_content = "video name: {}\n" \
                          "emotion: {}".format(v_name, sub_eva_info["emotion"])
        logger.info(sub_eva_content)
        file_tools.write_content_add(save_path, subject_info_file_name, sub_eva_content)
        global count
        if count == len(video_list) - 1:
            return redirect(url_for('end'))
        elif count < len(video_list) - 1:
            count = count + 1
            return redirect(url_for('rest'))
    except exceptions.TemplateNotFound as et:
        logger.error(et.message)
        abort(404)
    except Exception as e:
        logger.error(e)
        abort(500)


def record_eeg_data():
    m.inject_end_marker(os.path.abspath(save_path))


@app.route('/end')
@logs.log_record(logger)
def end():
    try:
        t1 = threading.Thread(target=record_eeg_data)
        t1.start()

        global count
        count = 0
        return render_template('end.html')
    except exceptions.TemplateNotFound as et:
        logger.error(et.message)
        abort(404)
    except Exception as e:
        logger.error(e)
        abort(500)


@app.route('/visualization')
def visualization():
    try:
        # return render_template('visualization.html')
        return render_template('visualization.html')
    except exceptions.TemplateNotFound as et:
        logger.error(et.message)
        abort(404)
    except Exception as e:
        logger.error(e)
        abort(500)


def get_eeg_data():
    eeg_data = np.random.random((1, 14, 256))
    return eeg_data


@logs.log_record(logger)
@app.route('/emotion_music', methods=['GET'], strict_slashes=False)
def emotion_music():
    # 17 155 293
    # 获取分类信息，传给前端
    # emotion_class = get_emotion.get_emotion_value(get_eeg_data())
    # emotion_class = [0,100,100,0,0,100]
    # a = str(emotion_class[-1])
    # emotion_index = str(emotion_class.pop())
    emotion_axis = None
    if emotion_class == "0":
        emotion_axis = 17
    elif emotion_class == "1":
        emotion_axis = 155
    elif emotion_class == "2":
        emotion_axis = 293
    else:
        return jsonify({"code": 404, "msg": "success", "emotion": emotion_axis, "music": ""})
    print(emotion_class)
    logger.info("get emotion: {}".format(emotion_class))
    music_name = generate.main(emotion_class)
    print(music_name)
    logger.info("generate music: {}".format(music_name))

    return jsonify({"code": 200, "msg": "success", "emotion": emotion_axis, "music": music_name})


@app.errorhandler(404)
def error(e):
    return redirect(url_for('err1'))


@app.errorhandler(500)
def error(e):
    return redirect(url_for('err2'))


@app.route('/err1')
def err1():
    return '您访问的页面已经去浪迹天涯了'


@app.route('/err2')
def err2():
    return '服务器异常。。。'


def product():
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    while True:
        # sample, timestamp = inlet.pull_sample()
        sample, timestamp = inlet.pull_chunk(timeout=1.0)
        eeg_queue.put(sample)


def consume():
    global time_count, emotion_class
    raw_data_list = []
    while True:
        eeg_data = eeg_queue.get()
        if time_count < 60:
            if len(eeg_data) == 256:
                raw_data_list.append(eeg_data)
            time_count += 1
        else:
            raw_np = np.array(raw_data_list)[:, :, 3:-1]
            raw_np = raw_np.swapaxes(2, 1)
            print(raw_np.shape)
            emotion_list = get_emotion.get_emotion_value(raw_np)
            print(emotion_list)
            emotion_class = str(np.argmax(np.bincount(emotion_list)))
            print(emotion_class)
            logger.info("emotion class is: {}".format(emotion_class))
            time_count = 0
            raw_data_list[:] = []
        eeg_queue.task_done()


if __name__ == '__main__':
    app.run(debug=True)
