import json
class Avi:
    def __init__(self, avi_data_path, ini_frame, fps=25):
        """
        :param avi_data_path: str
        :param ini_frame: int
        :param fps: int
        """
        self.ini_frame = ini_frame
        self.fps = fps
        with open(avi_data_path) as json_file:
            self.detections = json.load(json_file)

