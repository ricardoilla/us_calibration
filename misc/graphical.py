from matplotlib import pyplot as plt


def plot_probe_markers(trial):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Crea una cuadrícula de 2x2 subplots

    # Primer subplot
    axs[0, 0].plot(trial.frames, trial.e0.x, '.', label="e0 x")
    axs[0, 0].plot(trial.frames, trial.e0.y, '.', label="e0 y")
    axs[0, 0].plot(trial.frames, trial.e0.z, '.', label="e0 z")
    axs[0, 0].set_title('e0')
    axs[0, 0].legend()

    # Segundo subplot
    axs[0, 1].plot(trial.frames, trial.e1.x, '.', label="e1 x")
    axs[0, 1].plot(trial.frames, trial.e1.y, '.', label="e1 y")
    axs[0, 1].plot(trial.frames, trial.e1.z, '.', label="e1 z")
    axs[0, 1].set_title('e1')
    axs[0, 1].legend()

    # Tercer subplot
    axs[1, 0].plot(trial.frames, trial.e2.x, '.', label="e2 x")
    axs[1, 0].plot(trial.frames, trial.e2.y, '.', label="e2 y")
    axs[1, 0].plot(trial.frames, trial.e2.z, '.', label="e2 z")
    axs[1, 0].set_title('e2')
    axs[1, 0].legend()

    # Cuarto subplot
    axs[1, 1].plot(trial.frames, trial.e3.x, '.', label="e3 x")
    axs[1, 1].plot(trial.frames, trial.e3.y, '.', label="e3 y")
    axs[1, 1].plot(trial.frames, trial.e3.z, '.', label="e3 z")
    axs[1, 1].set_title('e3')
    axs[1, 1].legend()

    fig.suptitle(trial.name, fontsize=16)

    # Ajusta el layout para evitar solapamientos
    plt.tight_layout()

    # Muestra las gráficas
    plt.show()

def plot_stylus_markers(trial):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Crea una cuadrícula de 2x2 subplots

    # Primer subplot
    axs[0, 0].plot(trial.frames, trial.s0.x, '.', label="s0 x")
    axs[0, 0].plot(trial.frames, trial.s0.y, '.', label="s0 y")
    axs[0, 0].plot(trial.frames, trial.s0.z, '.', label="s0 z")
    axs[0, 0].set_title('s0')
    axs[0, 0].legend()

    # Segundo subplot
    axs[0, 1].plot(trial.frames, trial.s1.x, '.', label="s1 x")
    axs[0, 1].plot(trial.frames, trial.s1.y, '.', label="s1 y")
    axs[0, 1].plot(trial.frames, trial.s1.z, '.', label="s1 z")
    axs[0, 1].set_title('s1')
    axs[0, 1].legend()

    # Tercer subplot
    axs[1, 0].plot(trial.frames, trial.s2.x, '.', label="s2 x")
    axs[1, 0].plot(trial.frames, trial.s2.y, '.', label="s2 y")
    axs[1, 0].plot(trial.frames, trial.s2.z, '.', label="s2 z")
    axs[1, 0].set_title('s2')
    axs[1, 0].legend()

    # Cuarto subplot
    axs[1, 1].plot(trial.frames, trial.s3.x, '.', label="s3 x")
    axs[1, 1].plot(trial.frames, trial.s3.y, '.', label="s3 y")
    axs[1, 1].plot(trial.frames, trial.s3.z, '.', label="s3 z")
    axs[1, 1].set_title('s3')
    axs[1, 1].legend()

    fig.suptitle(trial.name, fontsize=16)
    # Ajusta el layout para evitar solapamientos
    plt.tight_layout()

    # Muestra las gráficas
    plt.show()

def plot_p(trial):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))  # Crea una cuadrícula de 1x2 subplots

    # Primer subplot
    axs[0].plot(trial.frames, trial.pw.x, '.', label="pw x")
    axs[0].plot(trial.frames, trial.pw.y, '.', label="pw y")
    axs[0].plot(trial.frames, trial.pw.z, '.', label="pw z")
    axs[0].set_title('pw')
    axs[0].legend()

    # Segundo subplot
    axs[1].plot(trial.frames, trial.pe.x, '.', label="pe x")
    axs[1].plot(trial.frames, trial.pe.y, '.', label="pe y")
    axs[1].plot(trial.frames, trial.pe.z, '.', label="pe z")
    axs[1].set_title('pe')
    axs[1].legend()

    fig.suptitle(trial.name, fontsize=16)
    # Ajusta el layout para evitar solapamientos
    plt.tight_layout()

    # Muestra las gráficas
    plt.show()

def plot_results_in_e(trial):
    fig, axs = plt.subplots(3, constrained_layout=True)
    fig.suptitle('System Solution')
    axs[0].plot(trial.frames, trial.pe.x, '.')
    axs[1].plot(trial.frames, trial.pe.y, '.')
    axs[2].plot(trial.frames, trial.pe.z, '.')
    axs[0].plot(trial.frames, trial.result.x, '.')
    axs[1].plot(trial.frames, trial.result.y, '.')
    axs[2].plot(trial.frames, trial.result.z, '.')

    axs[0].legend(['ground truth', 'calculated'])
    axs[1].legend(['ground truth', 'calculated'])
    axs[2].legend(['ground truth', 'calculated'])

    axs[0].title.set_text('x')
    axs[1].title.set_text('y')
    axs[2].title.set_text('z')

    plt.show()

def plot_results_in_w(trial):
    fig, axs = plt.subplots(3, constrained_layout=True)
    fig.suptitle('System Solution')
    axs[0].plot(trial.frames, trial.pw.x, '.')
    axs[1].plot(trial.frames, trial.pw.y, '.')
    axs[2].plot(trial.frames, trial.pw.z, '.')
    axs[0].plot(trial.frames, trial.result_in_w.x, '.')
    axs[1].plot(trial.frames, trial.result_in_w.y, '.')
    axs[2].plot(trial.frames, trial.result_in_w.z, '.')

    axs[0].legend(['ground truth', 'calculated'])
    axs[1].legend(['ground truth', 'calculated'])
    axs[2].legend(['ground truth', 'calculated'])

    axs[0].title.set_text('x')
    axs[1].title.set_text('y')
    axs[2].title.set_text('z')

    plt.show()


def plot_error(trial):
        fig, axs = plt.subplots(3, constrained_layout=True)
        fig.suptitle('System Solution')
        axs[0].plot(trial.frames, trial.error.x, '.')
        axs[1].plot(trial.frames, trial.error.y, '.')
        axs[2].plot(trial.frames, trial.error.z, '.')
        axs[0].plot(trial.frames, trial.error_in_w.x, '.')
        axs[1].plot(trial.frames, trial.error_in_w.y, '.')
        axs[2].plot(trial.frames, trial.error_in_w.z, '.')
        axs[0].legend(['error in E', 'error in W'])
        axs[1].legend(['error in E', 'error in W'])
        axs[2].legend(['error in E', 'error in W'])
        axs[0].title.set_text('x')
        axs[1].title.set_text('y')
        axs[2].title.set_text('z')
        plt.show()
