from .C3D import C3D
from .Avi import Avi
from .InstantCoordinates import InstantCoordinates
import numpy as np
from scipy.signal import medfilt
from misc.calc_methods import *
from misc.config import *
from enum import Enum
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares
from scipy.optimize import minimize

class CoordinateLists:
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []

    def all(self):
        return [self.x, self.y, self.z]

    def get_point(self, index):
        return np.array([self.x[index], self.y[index], self.z[index]])

    def append_point(self, point):
        self.x.append(point[0])
        self.y.append(point[1])
        self.z.append(point[2])

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Index out of range. Valid indexes are 0, 1, and 2.")

    def median_filter(self, window_size):
        self.x = medfilt(self.x, window_size)
        self.y = medfilt(self.y, window_size)
        self.z = medfilt(self.z, window_size)


# Función de error a minimizar
def error_function(params, u, v, p_x_E, p_y_E, p_z_E):
    a = params[:12].reshape(3, 4)  # Parametros de la transformación
    w = params[12:]  # Valores de w

    # Calcula los puntos transformados
    q = np.vstack((u, v, w, np.ones_like(u)))  # Matriz de puntos q_i
    p_E_calculated = a @ q

    # Error entre puntos calculados y puntos observados
    error_x = p_E_calculated[0] - p_x_E
    error_y = p_E_calculated[1] - p_y_E
    error_z = p_E_calculated[2] - p_z_E

    return np.concatenate((error_x, error_y, error_z))


# Función de error a minimizar con regularización
def error_function_minimize(params, u, v, p_x_E, p_y_E, p_z_E, lambda_reg=1e-4):
    a = params[:12].reshape(3, 4)  # Parametros de la transformación
    w = params[12:]  # Valores de w

    # Calcula los puntos transformados
    q = np.vstack((u, v, w, np.ones_like(u)))  # Matriz de puntos q_i
    p_E_calculated = a @ q

    # Error entre puntos calculados y puntos observados
    error_x = p_E_calculated[0] - p_x_E
    error_y = p_E_calculated[1] - p_y_E
    error_z = p_E_calculated[2] - p_z_E

    # Agregar término de regularización
    regularization = lambda_reg * np.sum(params ** 2)

    return np.sum(error_x ** 2 + error_y ** 2 + error_z ** 2) + regularization


def residuals(params, u, v, p_x_E, p_y_E, p_z_E):
    a = params[:12].reshape(3, 4)
    w = params[12:]

    q = np.vstack((u, v, w, np.ones_like(u)))
    p_E_calculated = a @ q

    error_x = p_E_calculated[0] - p_x_E
    error_y = p_E_calculated[1] - p_y_E
    error_z = p_E_calculated[2] - p_z_E

    return np.hstack((error_x, error_y, error_z))


class Trial:
    def __init__(self, name, c3d, avi_data, from_idx=0, num_markers=4):
        """
        :param name: str Name of the Trial
        :param c3d: C3D Object
        :param avi_data: Avi Object
        """
        self.name = name
        self.c3d = c3d
        self.avi = avi_data
        self.avi_points = self.avi.detections
        self.vicon_points = self.c3d.vicon_points
        self.sincronized_data = {}
        self.list_of_positions = []  # List of InstantCoordinates
        self.frames = []
        self.Tx = 0
        self.qty = 0
        self.from_idx = from_idx
        self.num_markers = num_markers
        self.q = CoordinateLists()
        self.s0 = CoordinateLists()
        self.s1 = CoordinateLists()
        self.s2 = CoordinateLists()
        self.s3 = CoordinateLists()
        self.e0 = CoordinateLists()
        self.e1 = CoordinateLists()
        self.e2 = CoordinateLists()
        self.e3 = CoordinateLists()
        self.pe = CoordinateLists()
        self.pw = CoordinateLists()
        self.error = CoordinateLists()
        self.error_in_w = CoordinateLists()
        self.result = CoordinateLists()
        self.result_in_w = CoordinateLists()
        self.sincronize()
        self.process_data(5)

    def sincronize(self):
        """
        Sincronize the Vicon data with the AVI data
        """
        for key, value in self.avi_points.items():
            real_fr = int(key) - self.avi.ini_frame
            if real_fr >= 0:
                t_fr = ((self.c3d.fps / self.avi.fps) * real_fr) + self.c3d.ini_frame + 1
                if int(t_fr) < len(self.vicon_points):
                    self.sincronized_data[str(real_fr)] = {}
                    self.sincronized_data[str(real_fr)]['Q'] = np.array([self.avi_points[key][0],
                                                                         self.avi_points[key][1],
                                                                         0])
                    self.sincronized_data[str(real_fr)]['S0'] = np.array(self.vicon_points[str(int(t_fr))]['S0'])
                    self.sincronized_data[str(real_fr)]['S1'] = np.array(self.vicon_points[str(int(t_fr))]['S1'])
                    self.sincronized_data[str(real_fr)]['S2'] = np.array(self.vicon_points[str(int(t_fr))]['S2'])
                    self.sincronized_data[str(real_fr)]['S3'] = np.array(self.vicon_points[str(int(t_fr))]['S3'])
                    self.sincronized_data[str(real_fr)]['E0'] = np.array(self.vicon_points[str(int(t_fr))]['E0'])
                    self.sincronized_data[str(real_fr)]['E1'] = np.array(self.vicon_points[str(int(t_fr))]['E1'])
                    self.sincronized_data[str(real_fr)]['E2'] = np.array(self.vicon_points[str(int(t_fr))]['E2'])
                    self.sincronized_data[str(real_fr)]['E3'] = np.array(self.vicon_points[str(int(t_fr))]['E3'])
                    self.list_of_positions.append(InstantCoordinates(self.avi_points[key][0],
                                                                     self.avi_points[key][1],
                                                                     self.sincronized_data[str(real_fr)]['S0'],
                                                                     self.sincronized_data[str(real_fr)]['S1'],
                                                                     self.sincronized_data[str(real_fr)]['S2'],
                                                                     self.sincronized_data[str(real_fr)]['S3'],
                                                                     self.sincronized_data[str(real_fr)]['E0'],
                                                                     self.sincronized_data[str(real_fr)]['E1'],
                                                                     self.sincronized_data[str(real_fr)]['E2'],
                                                                     self.sincronized_data[str(real_fr)]['E3']))

    def process_data(self, kernel=3, filtered=False):
        # Ordena las claves del diccionario convirtiéndolas a enteros antes de ordenar
        sorted_keys = sorted(self.sincronized_data.keys(), key=lambda x: int(x))
        sorted_keys = sorted_keys[self.from_idx:]
        # Procesa los datos siguiendo el orden de las claves
        for key in sorted_keys:
            value = self.sincronized_data[key]
            self.frames.append(int(key))  # Convertir el key a entero antes de agregarlo
            for i in range(3):
                self.q[i].append(value['Q'][i])
                self.s0[i].append(value['S0'][i])
                self.s1[i].append(value['S1'][i])
                self.s2[i].append(value['S2'][i])
                self.s3[i].append(value['S3'][i])
                self.e0[i].append(value['E0'][i])
                self.e1[i].append(value['E1'][i])
                self.e2[i].append(value['E2'][i])
                self.e3[i].append(value['E3'][i])
        self.qty = len(self.frames)
        if filtered:
            self.q.median_filter(kernel)
            self.s0.median_filter(kernel)
            self.s1.median_filter(kernel)
            self.s2.median_filter(kernel)
            self.s3.median_filter(kernel)
            self.e0.median_filter(kernel)
            self.e1.median_filter(kernel)
            self.e2.median_filter(kernel)
            self.e3.median_filter(kernel)

    def calculate_pw_pe(self):
        if self.qty == 0:
            print("Need Process Data first")
        else:
            for i in range(self.qty):
                E_cluster = [self.e0.get_point(i), self.e1.get_point(i), self.e2.get_point(i), self.e3.get_point(i)]
                S_cluster = [self.s0.get_point(i), self.s1.get_point(i), self.s2.get_point(i), self.s3.get_point(i)]
                # print('S0: ', self.s0.get_point(i))
                # print('S1: ', self.s1.get_point(i))
                # print('S2: ', self.s2.get_point(i))

                if self.num_markers == 4:
                    E_system, E_center = get_system_from_4_clusters(E_cluster)
                    S_system, S_center = get_system_from_4_clusters(S_cluster)
                elif self.num_markers == 3:
                    E_system, E_center = get_system_from_3_clusters(E_cluster)
                    S_system, S_center = get_system_from_3_clusters(S_cluster)
                else:
                    print("Wrong number of markers")
                # print('ECOGRAFO:', E_system, E_center)
                # print('Stylus:', S_system, S_system)
                # Calculate transformations
                T_w_e = get_transformation(WORLD, np.array([0, 0, 0]), E_system, E_center)
                T_s_w = get_transformation(S_system, S_center, WORLD, np.array([0, 0, 0]))
                # Transform point P
                if self.num_markers == 4:
                    p_W = np.matmul(T_s_w, Ps_4_markers)
                else:
                    p_W = np.matmul(T_s_w, Ps_3_markers)
                p_E = np.matmul(T_w_e, p_W)
                self.pw.append_point(p_W)
                self.pe.append_point(p_E)
                # break

    def fit_data(self):
        A_matrix = []
        b = []
        if self.qty == 0:
            print("Need Process Data first")
        else:
            for i in range(self.qty):
                q_point = self.q.get_point(i)
                A_matrix.extend([
                    [q_point[0], q_point[1], 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, q_point[0], q_point[1], 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, q_point[0], q_point[1], 1]
                ])
                pe_point = self.pe.get_point(i)
                b.extend([
                    [pe_point[0]],
                    [pe_point[1]],
                    [pe_point[2]]
                ])
            A_matrix = np.array(A_matrix)
            b = np.array(b)
            # Reshape A_matrix and b for LinearRegression
            A_matrix_reshaped = A_matrix.reshape(-1, A_matrix.shape[-1])
            b_reshaped = b.flatten()
            model = LinearRegression(fit_intercept=False)
            model.fit(A_matrix_reshaped, b_reshaped)
            x = model.coef_.reshape(-1, 1)
            Tx = [[float(x[0]), float(x[1]), float(0), float(x[2])],
                  [float(x[3]), float(x[4]), float(0), float(x[5])],
                  [float(x[6]), float(x[7]), float(0), float(x[8])],
                  [0, 0, 0, 1]]
            self.Tx = np.array(Tx)
            # print('Transformation T:\n{}'.format(Tx))

    def fit_data_including_z(self, method='minimize'):
        u = np.array(self.q.x)
        v = np.array(self.q.y)
        p_x_E = np.array(self.pe.x)
        p_y_E = np.array(self.pe.y)
        p_z_E = np.array(self.pe.z)

        print('Normalizando datos')
        u = (u - np.mean(u)) / np.std(u)
        v = (v - np.mean(v)) / np.std(v)
        p_x_E = (p_x_E - np.mean(p_x_E)) / np.std(p_x_E)
        p_y_E = (p_y_E - np.mean(p_y_E)) / np.std(p_y_E)
        p_z_E = (p_z_E - np.mean(p_z_E)) / np.std(p_z_E)


        print('Ejecutando algoritmo de optimizacion!')
        # Inicializar w con valores aleatorios o ceros
        # w_init = np.zeros(len(u))
        # Valores iniciales para los parámetros de la transformación (12 parámetros) y w (N valores)
        # initial_params = np.hstack((np.random.rand(12), w_init))

        initial_params = np.hstack((np.random.rand(12), np.random.rand(len(u))))

        # Realizar la optimización
        if method == 'minimize':
            result = minimize(error_function_minimize, initial_params, args=(u, v, p_x_E, p_y_E, p_z_E), method='BFGS')
        elif method == 'minimize2':
            result = minimize(error_function_minimize, initial_params, args=(u, v, p_x_E, p_y_E, p_z_E), method='L-BFGS-B', options={'ftol': 1e-6, 'gtol': 1e-6})
        elif method == 'least_squares':
            result = least_squares(error_function, initial_params, args=(u, v, p_x_E, p_y_E, p_z_E))
        elif method == 'least_squares2':
            result = least_squares(residuals, initial_params, args=(u, v, p_x_E, p_y_E, p_z_E), method='trf', xtol=1e-6,
                                   ftol=1e-6)
        elif method == 'least_squares3':
            result = least_squares(
                residuals,
                initial_params,
                args=(u, v, p_x_E, p_y_E, p_z_E),
                method='trf',
                xtol=1e-6,
                ftol=1e-6,
                max_nfev=10000  # Aumentar el número máximo de evaluaciones
            )
        else:
            result = None
        if result.success:
            # Extraer los parámetros optimizados
            a_optimized = result.x[:12].reshape(3, 4)
            w_optimized = result.x[12:]
            print('Transformation T:\n{}'.format(a_optimized))
            print('w:\n{}'.format(w_optimized))
            self.Tx = a_optimized
            self.w = w_optimized
        else:
            print("La optimización falló:", result.message)
            print('\n')


    def calculate_result_and_error(self):
        for i in range(self.qty):
            q = self.q.get_point(i)
            q = np.append(q, 1)
            res = np.matmul(self.Tx, q)
            self.result.append_point(res)
            gt = self.pe.get_point(i)
            error = np.abs(gt - res[:3])
            self.error.append_point(error)

    def calculate_result_and_error_including_z(self):
        for i in range(self.qty):
            q = self.q.get_point(i)[:2]
            q = np.append(q, self.w[i])
            q = np.append(q, 1)
            res = np.matmul(self.Tx, q)
            self.result.append_point(res)
            gt = self.pe.get_point(i)
            error = np.abs(gt - res[:3])
            self.error.append_point(error)

    def calculate_result_in_w(self):
        for i in range(self.qty):
            E_cluster = [self.e0.get_point(i), self.e1.get_point(i), self.e2.get_point(i), self.e3.get_point(i)]
            S_cluster = [self.s0.get_point(i), self.s1.get_point(i), self.s2.get_point(i), self.s3.get_point(i)]
            if self.num_markers == 4:
                E_system, E_center = get_system_from_4_clusters(E_cluster)
                S_system, S_center = get_system_from_4_clusters(S_cluster)
            elif self.num_markers == 3:
                E_system, E_center = get_system_from_3_clusters(E_cluster)
                S_system, S_center = get_system_from_3_clusters(S_cluster)
            else:
                print("Wrong number of markers")
            # Calculate transformations
            T_e_w = get_transformation(E_system, E_center, WORLD, np.array([0, 0, 0]))
            # Transform point P
            p_W = np.matmul(T_e_w, np.append(self.result.get_point(i), 1))
            self.result_in_w.append_point(p_W)
            gt = self.pw.get_point(i)
            error = np.abs(gt - p_W[:3])
            self.error_in_w.append_point(error)














