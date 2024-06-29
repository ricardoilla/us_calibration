from calibration_classes import *
from misc import config
from misc.graphical import *
import numpy as np
from matplotlib import pyplot as plt

# c3d1 = C3D(config.c3d1_data_path, config.extract_pts_idx_1, 502)
# avi_data1 = Avi(config.cal1_data_path, 40)
# trial1 = Trial("Trial1", c3d1, avi_data1)
# plot_probe_markers(trial1)
# plot_stylus_markers(trial1)


c3d2 = C3D(config.c3d2_data_path, config.extract_pts_idx_2_4markers, 283)
avi_data2 = Avi(config.cal2_data_path, 28)
trial2 = Trial("Trial2", c3d2, avi_data2, from_idx=25, num_markers=4)
trial2.calculate_pw_pe()

# plot_q(trial2)
# plot_probe_markers(trial2)
# plot_stylus_markers(trial2)
# plot_p(trial2)
# trial2.fit_data()

trial2.fit_data_including_z(method='least_squares3')

# trial2.calculate_result_and_error()

trial2.calculate_result_and_error_including_z()
plot_results_in_e(trial2)

plt.plot(trial2.frames, trial2.w, '.', label="w")

# trial2.calculate_result_in_w()
# plot_results_in_w(trial2)

plot_error(trial2)

print_errors(trial2)



