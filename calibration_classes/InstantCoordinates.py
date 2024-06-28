import numpy as np

class InstantCoordinates:
    def __init__(self, _u, _v, _s0, _s1, _s2, _s3, _e0, _e1, _e2, _e3):
        """
        :param _u: int
        :param _v: int
        :param _s0: [int, int, int]
        :param _s1: [int, int, int]
        :param _s2: [int, int, int]
        :param _s3: [int, int, int]
        :param _e0: [int, int, int]
        :param _e1: [int, int, int]
        :param _e2: [int, int, int]
        :param _e3: [int, int, int]
        """
        self.point_i = np.array([_u, _v])
        self.s0 = np.array(_s0)
        self.s1 = np.array(_s1)
        self.s2 = np.array(_s2)
        self.s3 = np.array(_s3)
        self.e0 = np.array(_e0)
        self.e1 = np.array(_e1)
        self.e2 = np.array(_e2)
        self.e3 = np.array(_e3)
