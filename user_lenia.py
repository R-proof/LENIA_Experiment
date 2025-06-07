from lenia import *
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def simulation():

    # Param tkinter
    root = tk.Tk()
    dim = ("1080x780")
    root.title("lenia")

    # Param simulation
    beta = input() # np.array([1,0.5])
    R = input() ; dx = 1/R # controle R mais display dx 
    T = input() ; dt = 1/T

    mu = tk.Scale()
    sigma = tk.Scale()


    # other
    space = Label(root, text="")

    # mise sur la grid
    space.grid(row=1,column=1)
    button_load_config.grid(row=1,column=2)
    button_export_config.grid(row=1,column=3)
    button_start.grid(row=1,column=4)
    button_pause.grid(row=1,column=5)
    button_regenerate.grid(row=1,column=6,columspan=2)
    space.grid(row=1,column=8)

    space.grid(row=2,column=1)
    world.grid(row=2,column=2,columspan=2)
    potential.grid(row=2,column=3,columspan=2)
    growth.grid(row=2,column=4,columspan=2)
    space.grid(row=2,column=8)

    beta_label.grid(row=3,column=1)
    beta_input.grid(row=3,column=2)
    beta_value.grid(row=3,column=3)
    graph_kernel_on_grid.grid(row=3,column=4,rowspan=2)
    graph_kernel.grid(row=3,column=5,rowspan=2)
    mu_label.grid(row=3,column=6)
    slider_mu.grid(row=3,column=7)
    mu_value.grid(row=3,column=8)

    dx_label.grid(row=4,column=1)
    slider_dx.grid(row=4,column=2)
    value_dx.grid(row=4,column=3)
    sigma_label.grid(row=4,column=6)
    slider_sigma.grid(row=4,column=7)
    value_sigma.grid(row=4,column=8)

    dt_label.grid(row=5,column=1)
    slider_dt.grid(row=5,column=2)
    value_dt.grid(row=5,column=3)
    space.grid(row=5,column=6,rowspan=3)

    space.grid(row=6,column=1)
    button_quit.grid(row=6,column=2,rowspan=2)
    space.grid(row=6,column=4,rowspan=2)
    graph_growth.grid(row=6,column=6,rowspan=2)
    space.grid(row=6,column=8)



    time = 0
    K, K_FFT = pre_calculate_kernel(beta, dx)    
    world = get_initial_configuration(SIZE_X, SIZE_Y)

    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=1,column=1,columspan=3)


    while True:
        world, growth, potential = run_automaton(world, K, K_FFT, mu, sigma, dt)
        time += dt

        axs[0].cla(); axs[0].imshow(world, cmap='gray'); axs[0].set_title("World")
        axs[1].cla(); axs[1].imshow(potential, cmap='viridis'); axs[1].set_title("Potential")
        axs[2].cla(); axs[2].imshow(growth, cmap='seismic', vmin=-1, vmax=1); axs[2].set_title("Growth")
        plt.pause(0.01)

    root.mainloop()























