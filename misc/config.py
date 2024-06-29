import numpy as np

##############################################################################
Ps_3_markers = np.array([144.37, 141.53, -19.855, 1])
Ps_4_markers = np.array([-108.46930295,  -28.33069981, -141.36270586, 1])
##############################################################################
WORLD = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
##############################################################################
cal1_data_path = 'data/output_cal1.json'
c3d1_data_path = 'data/calibracion1.c3d'
extract_pts_idx_1 = {'stylus_marker_0': 2,
                   'stylus_marker_1': 0,
                   'stylus_marker_2': 3,
                   'stylus_marker_3': 1,
                   'probe_0': 7,
                   'probe_1': 8,
                   'probe_2': 9,
                   'probe_3': 10,
                   'sync': 11,
                   }
##############################################################################
cal2_data_path = 'data/output_cal2.json'  # 'data/output_cal2_29jun.json'
c3d2_data_path = 'data/calibracion2.c3d'
extract_pts_idx_2_3markers = {'stylus_marker_0': 0,
                   'stylus_marker_1': 1,
                   'stylus_marker_2': 2,
                   'stylus_marker_3': 3,
                   'probe_0': 7,
                   'probe_1': 8,
                   'probe_2': 9,
                   'probe_3': 10,
                   'sync': 11,
                   }
extract_pts_idx_2_4markers = {'stylus_marker_0': 2,
                   'stylus_marker_1': 0,
                   'stylus_marker_2': 3,
                   'stylus_marker_3': 1,
                   'probe_0': 7,
                   'probe_1': 8,
                   'probe_2': 9,
                   'probe_3': 10,
                   'sync': 11,
                   }
##############################################################################

