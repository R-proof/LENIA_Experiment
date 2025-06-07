# =======================================================================================================================
# =======================================================================================================================
# Lenia home-made
# =======================================================================================================================
# =======================================================================================================================

# Packages ==================================================================


np.set_printoptions(precision=2, suppress=True)

# =============================================================================
# Parameters
# =============================================================================

# simulation
SIZE_X = 101
SIZE_Y = 101

# =============================================================================
# Functions génériques
# =============================================================================

def construct_grid_L(x,y):
    grid= np.meshgrid(np.linspace(-1, 1, x),np.linspace(-1, 1, y))
    return(grid)

# grid = construct_grid_L(SIZE_X,SIZE_Y)
# print(grid[0][0,0])  # X[0, 0]
# print(grid[1][0,0])  # Y[0, 0]

def kernel_core(r):
    return (np.exp(4 - 4 / (r * (1 - r))) * (r > 0) * (r < 1))

def growth_mapping(u,mu,sigma):
    return (2 * np.exp(-((u - mu) ** 2) / (2 * sigma ** 2)) - 1)

def get_initial_configuration(x,y):
    A = np.random.rand(x,y)
    red_noise = np.random.rand(x, y) < 0.2 # monde avec densité faible 0.2
    return(A*red_noise)

# =============================================================================
# Fonction simulation
# =============================================================================

def pre_calculate_kernel(beta,dx):
    i,j = construct_grid_L(SIZE_X,SIZE_Y)
    radius = np.sqrt(i**2+j**2)*dx

    B = len(beta)
    Br = B*radius
    floor_Br = Br % 1

    Ks = beta[np.floor(Br).astype(int)]*kernel_core(floor_Br)
    K = Ks/np.sum(Ks)
    K_FFT = fft2(K) # conditions aux bords

    return(K,K_FFT)

def run_automaton(world, K, K_FFT, mu, sigma, dt):
    world_FFT = fft2(world) # conditions aux bords su monde A

    potential_FFT = K_FFT * world_FFT # K*A dans domaine fourrier
    potential = np.real(ifft2(potential_FFT)) # tranformation dans l'espace réel, np.real garde uniquement partie réel
    potential = fftshift(potential) # cencentrage du résultat en (0,0

    growth = growth_mapping(potential, mu, sigma)

    # mise à jour de A
    new_world_raw = world + dt * growth
    new_world = np.clip(new_world_raw, 0, 1) # clip [0,1]

    return (new_world, growth, potential)



















