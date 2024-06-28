import c3d
import numpy as np

class C3D:
    def __init__(self, path, pts_idx, ini_frame, fps=100):
        """
        :param path: str path to c3d file
        :param pts_idx: dict containing c3d index for each point
        """
        self.path = path
        self.pts_idx = pts_idx
        self.ini_frame = ini_frame
        self.fps = fps
        self.vicon_points = {}
        self.extract_data()

    def extract_data(self):
        with open(self.path, 'rb') as handle:
            reader = c3d.Reader(handle)
            for frame, point_list in enumerate(reader.read_frames()):
                frame_number = str(point_list[0])
                points = point_list[1]
                self.vicon_points[frame_number] = {}
                self.vicon_points[frame_number]['S0'] = np.array(points[self.pts_idx['stylus_marker_0']][:3])
                self.vicon_points[frame_number]['S1'] = np.array(points[self.pts_idx['stylus_marker_1']][:3])
                self.vicon_points[frame_number]['S2'] = np.array(points[self.pts_idx['stylus_marker_2']][:3])
                self.vicon_points[frame_number]['S3'] = np.array(points[self.pts_idx['stylus_marker_3']][:3])
                self.vicon_points[frame_number]['E0'] = np.array(points[self.pts_idx['probe_0']][:3])
                self.vicon_points[frame_number]['E1'] = np.array(points[self.pts_idx['probe_1']][:3])
                self.vicon_points[frame_number]['E2'] = np.array(points[self.pts_idx['probe_2']][:3])
                self.vicon_points[frame_number]['E3'] = np.array(points[self.pts_idx['probe_3']][:3])
