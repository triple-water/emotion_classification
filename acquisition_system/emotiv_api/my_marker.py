from pathlib import Path
from acquisition_system.emotiv_api.cortex import Cortex
import time

user = {
    "license": "30a7df02-37ba-462c-8b75-feedfee8c815",
    "client_id": "icAFfErGaI0GbCU8T2IeIVkjaWb4YNKrPg2GgOwK",
    "client_secret": "2SRPRAKqE9ZIUUCXVBTydwi2eheuqkjzSmj4heRDnTTce5v1JPo1sosSgOiRNM4IUTsk6DXrHA2xcgQgCuoAIkekbtnye79OjCb0Q15BTLpsL3OWXbtD9Ot4JtDbDzmB",
    "debit": 100
}

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
            "value": 0,
            "port": "python-app",
            "time": marker_time
        }
        self.c.inject_marker_request(marker)

    def start_record(self, name, description):
        self.c.create_record(name, description)