from lenia import *
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LeniaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lenia Simulation")
        self.root.geometry("1000x780")

        # === Simulation Parameters ===
        self.SIZE_X, self.SIZE_Y = 51, 51
        self.beta = tk.StringVar(value="1,0.5")
        self.R = tk.DoubleVar(value=2)
        self.T = tk.DoubleVar(value=20)
        self.mu = tk.DoubleVar(value=1)
        self.sigma = tk.DoubleVar(value=0.5)

        self.running = False

        self.build_interface()

    def build_interface(self):
        # === Sliders and Labels ===
        tk.Label(self.root, text="beta (comma sep)").grid(row=0, column=0)
        self.beta_entry = tk.Entry(self.root, textvariable=self.beta)
        self.beta_entry.grid(row=0, column=1)

        tk.Label(self.root, text="R").grid(row=1, column=0)
        tk.Scale(self.root, variable=self.R, from_=1, to=5, resolution=0.1, orient=tk.HORIZONTAL).grid(row=1, column=1)

        tk.Label(self.root, text="T").grid(row=2, column=0)
        tk.Scale(self.root, variable=self.T, from_=5, to=100, resolution=1, orient=tk.HORIZONTAL).grid(row=2, column=1)

        tk.Label(self.root, text="mu").grid(row=3, column=0)
        tk.Scale(self.root, variable=self.mu, from_=0, to=2, resolution=0.1, orient=tk.HORIZONTAL).grid(row=3, column=1)

        tk.Label(self.root, text="sigma").grid(row=4, column=0)
        tk.Scale(self.root, variable=self.sigma, from_=0.1, to=1.5, resolution=0.1, orient=tk.HORIZONTAL).grid(row=4, column=1)

        # === Buttons ===
        tk.Button(self.root, text="Start", command=self.start_simulation).grid(row=5, column=0)
        tk.Button(self.root, text="Pause", command=self.pause_simulation).grid(row=5, column=1)
        tk.Button(self.root, text="Regenerate", command=self.regenerate_world).grid(row=5, column=2)
        tk.Button(self.root, text="Quit", command=self.root.quit).grid(row=5, column=3)

        # === Plotting area ===
        self.fig, self.axs = plt.subplots(1, 3, figsize=(5, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=6, column=0, columnspan=4)

        self.world = get_initial_configuration(self.SIZE_X, self.SIZE_Y)

    def regenerate_world(self):
        self.world = get_initial_configuration(self.SIZE_X, self.SIZE_Y)

    def start_simulation(self):
        self.running = True
        self.run()

    def pause_simulation(self):
        self.running = False

    def run(self):
        if not self.running:
            return

        # Params
        beta = np.array([float(b) for b in self.beta.get().split(",")])
        R = self.R.get()
        dx = 1 / R
        dt = 1 / self.T.get()
        mu = self.mu.get()
        sigma = self.sigma.get()

        # Kernel
        K, K_FFT = pre_calculate_kernel(beta, dx, self.SIZE_X, self.SIZE_Y)

        # Update
        self.world, growth, potential = run_automaton(self.world, K, K_FFT, mu, sigma, dt)

        # Display
        self.axs[0].cla(); self.axs[0].imshow(self.world, cmap='gray'); self.axs[0].set_title("World")
        self.axs[1].cla(); self.axs[1].imshow(potential, cmap='viridis'); self.axs[1].set_title("Potential")
        self.axs[2].cla(); self.axs[2].imshow(growth, cmap='seismic', vmin=-1, vmax=1); self.axs[2].set_title("Growth")
        self.canvas.draw()

        self.root.after(50, self.run)  # Repeat every 50 ms


# === Simulation Kernel helpers ===
def pre_calculate_kernel(beta, dx, size_x, size_y):
    x, y = np.meshgrid(np.linspace(-1, 1, size_x), np.linspace(-1, 1, size_y))
    radius = np.sqrt(x ** 2 + y ** 2) * dx
    B = len(beta)
    Br = B * radius
    floor_Br = np.floor(Br).astype(int)
    rem = Br % 1
    values = np.zeros_like(radius)
    for i in range(B):
        mask = (floor_Br == i)
        values[mask] = beta[i] * kernel_core(rem[mask])
    K = values / np.sum(values)
    return K, fft2(K)


if __name__ == '__main__':
    root = tk.Tk()
    app = LeniaGUI(root)
    root.mainloop()