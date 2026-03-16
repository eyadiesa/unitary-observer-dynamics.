# ==========================================
# MASTER SCRIPT: Observer Dynamics & Decoherence
# Generating all 5 publication-ready figures.
# ==========================================
import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags, kron
from scipy.sparse.linalg import eigsh, expm_multiply
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. Global Plot Settings (Publication Quality)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

# ==========================================
# 1. Global Parameters
# ==========================================
N_x = 64; L = 8.0; m = 1.0; a = 5.0; b = 1.0
omega_bar = 1.0; omega_width = 0.2; lam = 1.0; omega_q = 0.0
g_max = 0.3; kappa_max = 0.1
dt = 0.005; nsteps = 400

seed = 42
rng = np.random.default_rng(seed)

# ==========================================
# 2. Subsystem Builders
# ==========================================
def build_observer():
    x_grid = np.linspace(-L, L, N_x)
    dx = x_grid[1] - x_grid[0]
    X_op = diags(x_grid, 0, format='csr')

    main = -2.0 * np.ones(N_x)
    off = 1.0 * np.ones(N_x - 1)
    lap = diags([off, main, off], [-1, 0, 1], format='csr') / (dx**2)

    T_op = -0.5 * (1.0 / m) * lap
    Vx = -a * (x_grid**2) + b * (x_grid**4)
    V_op = diags(Vx, 0, format='csr')
    H_O = T_op + V_op

    vals, vecs = eigsh(H_O, k=4, which='SA')
    psi_o = vecs[:, 0]
    psi_o /= np.linalg.norm(psi_o)
    return H_O, X_op, psi_o, x_grid, Vx, vals[0]

def build_environment(N_E):
    dimE = 2**N_E
    sigma_z = np.array([[1.,0.],[0.,-1.]], dtype=complex)

    env_sigmas = []
    for j in range(N_E):
        op = None
        for site in range(N_E):
            mat = sigma_z if site == j else np.eye(2)
            op = sp.csr_matrix(mat) if op is None else kron(op, sp.csr_matrix(mat), format='csr')
        env_sigmas.append(op)

    omega_js = rng.normal(omega_bar, omega_width, size=N_E)
    H_E = sp.csr_matrix((dimE, dimE), dtype=complex)
    for j in range(N_E):
        H_E += (omega_js[j]/2.0) * env_sigmas[j]
    return H_E, env_sigmas

def build_hamiltonian(H_O, X_op, H_E, env_sigmas, N_E, use_QE=True, use_EO=True):
    dimE = H_E.shape[0]; dimO = H_O.shape[0]; dimQ = 2
    total_dim = dimQ * dimE * dimO
    sigma_z = np.array([[1.,0.],[0.,-1.]], dtype=complex)

    def lift(opQ, opE, opO):
        return kron(kron(sp.csr_matrix(opQ), sp.csr_matrix(opE), format='csr'), sp.csr_matrix(opO), format='csr')

    H_Q = lift(0.5*omega_q*sigma_z, sp.eye(dimE), sp.eye(dimO))
    H_E_full = lift(np.eye(2), H_E, sp.eye(dimO))
    H_O_full = lift(np.eye(2), sp.eye(dimE), H_O)
    H_QO = lift(lam*sigma_z, sp.eye(dimE), X_op)

    H_total = H_Q + H_E_full + H_O_full + H_QO

    if use_QE:
        g_js = rng.uniform(0.0, g_max, size=N_E)
        for j in range(N_E):
            H_total += g_js[j] * lift(sigma_z, env_sigmas[j], sp.eye(dimO))

    if use_EO:
        kappa_js = rng.uniform(0.0, kappa_max, size=N_E)
        for j in range(N_E):
            H_total += kappa_js[j] * lift(np.eye(2), env_sigmas[j], X_op)

    return H_total.tocsr()

def build_initial_state(psi_o, N_E):
    psi_Q = np.array([1.,1.], dtype=complex)/np.sqrt(2)
    env_kets = []
    for _ in range(N_E):
        v = rng.normal(size=2) + 1j*rng.normal(size=2)
        v /= np.linalg.norm(v)
        env_kets.append(v)

    psi_E = env_kets[0]
    for j in range(1, N_E):
        psi_E = np.kron(psi_E, env_kets[j])

    psi_full = np.kron(np.kron(psi_Q, psi_E), psi_o)
    return psi_full / np.linalg.norm(psi_full)

# ==========================================
# 3. Evolution Engine
# ==========================================
def run_simulation(H_total, psi0, x_grid):
    dimO = len(x_grid)
    dimE = H_total.shape[0] // (2*dimO)
    times = np.arange(nsteps)*dt
    purity_Q = np.zeros(nsteps); x_expect = np.zeros(nsteps)

    psi = psi0.copy()
    for i in range(nsteps):
        psi = expm_multiply((-1j*H_total*dt), psi)
        psi /= np.linalg.norm(psi)

        psi_t = psi.reshape((2, dimE, dimO))

        # Qubit Purity
        matQ = psi_t.reshape((2, dimE*dimO))
        rhoQ = matQ @ matQ.conj().T
        purity_Q[i] = np.real(np.trace(rhoQ @ rhoQ))

        # Position <x>
        probs_x = np.sum(np.abs(psi_t)**2, axis=(0,1))
        x_expect[i] = np.sum(probs_x * x_grid)

    return times, purity_Q, x_expect, psi

# ==========================================
# 4. EXECUTION & PLOTTING
# ==========================================
print("Building observer...")
H_O, X_op, psi_o, x_grid, Vx, E0 = build_observer()

# --- FIGURE 1: Double Well ---
plt.figure(figsize=(6, 4))
plt.plot(x_grid, Vx, color='black', label='$V(x)$')
plt.plot(x_grid, E0 + 15.0*(np.abs(psi_o)**2), color='#2ca02c', fillstyle='bottom', label='Ground State $|\psi_0(x)|^2$ (scaled)')
plt.ylim(E0 - 5, 30)
plt.xlim(-4, 4)
plt.xlabel("Position $x$")
plt.ylabel("Energy")
plt.title("Model of the Physical Observer")
plt.legend()
plt.grid(alpha=0.4)
plt.savefig('Double_well_and_ground_state.pdf')
plt.close()

print("Running Control 1: No Environment...")
N_E_control = 4
H_E_c, sigmas_c = build_environment(N_E_control)
psi0_c = build_initial_state(psi_o, N_E_control)

H_noQE = build_hamiltonian(H_O, X_op, H_E_c, sigmas_c, N_E_control, use_QE=False, use_EO=False)
t_noQE, pur_noQE, x_noQE, _ = run_simulation(H_noQE, psi0_c, x_grid)

# --- FIGURE 2: No QE Control (FIXED SCALING) ---
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(t_noQE, pur_noQE, color='#1f77b4', linewidth=2, label='Qubit Purity (Entangling with Observer)')
ax1.set_xlabel('Time $[\hbar/\omega_0]$')
ax1.set_ylabel('Purity $\\mathrm{Tr}(\\rho_Q^2)$', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_ylim(0.45, 1.05)
ax1.grid(alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(t_noQE, x_noQE, color='#ff7f0e', linestyle='--', linewidth=2, label='Observer Position $\\langle x \\rangle$')
ax2.set_ylabel('Position $\\langle x \\rangle$', color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')
ax2.set_ylim(-1.5, 1.5) # FIX: Prevents fake 0.9 scaling

fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.title("Control: System-Observer Dynamics (No Environment)")
fig.tight_layout()
plt.savefig('No_QE_test.pdf')
plt.close()

print("Running Control 2: Environment ON...")
H_withQE = build_hamiltonian(H_O, X_op, H_E_c, sigmas_c, N_E_control, use_QE=True, use_EO=False)
t_withQE, pur_withQE, x_withQE, _ = run_simulation(H_withQE, psi0_c, x_grid)

# --- FIGURE 3: Decoherence Control (FIXED SCALING) ---
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(t_withQE, pur_withQE, color='#1f77b4', linewidth=2, label='Qubit Purity (Decoherence + Recurrence)')
ax1.set_xlabel('Time $[\hbar/\omega_0]$')
ax1.set_ylabel('Purity $\\mathrm{Tr}(\\rho_Q^2)$', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_ylim(0.45, 1.05)
ax1.grid(alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(t_withQE, x_withQE, color='#d62728', linestyle='--', linewidth=2, label='Observer Position $\\langle x \\rangle$')
ax2.set_ylabel('Position $\\langle x \\rangle$', color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')
ax2.set_ylim(-0.5, 0.5) # FIX: Locks y-axis to show it is flat at 0!

fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.title("Control: Environment-Induced Decoherence ($N_E=4$)")
fig.tight_layout()
plt.savefig('Decoherence_Control.pdf')
plt.close()

print("Running Full Simulation (N_E=8)... This will take a moment.")
N_E_full = 8
H_E_f, sigmas_f = build_environment(N_E_full)
psi0_f = build_initial_state(psi_o, N_E_full)

H_full = build_hamiltonian(H_O, X_op, H_E_f, sigmas_f, N_E_full, use_QE=True, use_EO=True)
t_full, pur_full, x_full, psi_final = run_simulation(H_full, psi0_f, x_grid)

# --- FIGURE 4: Full Unitary Failure ---
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(t_full, pur_full, color='#1f77b4', linewidth=2, label='Qubit Purity $\\mathrm{Tr}(\\rho_Q^2)$')
ax1.set_xlabel('Time $[\hbar/\omega_0]$')
ax1.set_ylabel('Purity', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_ylim(0.45, 1.05)
ax1.grid(alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(t_full, x_full, color='#d62728', linestyle='--', linewidth=2, label='Observer Position $\\langle x \\rangle$')
ax2.set_ylabel('Position $\\langle x \\rangle$', color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')
ax2.set_ylim(-0.5, 0.5)

fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.title('Failure of Spontaneous Collapse (Unitary Evolution)')
fig.tight_layout()
plt.savefig('figure_unitary_failure.pdf')
plt.close()

# --- FIGURE 5: Final Distribution ---
dimO = len(x_grid)
dimE_f = 2**N_E_full
psi_final_reshaped = psi_final.reshape((2, dimE_f, dimO))
prob_x_final = np.sum(np.abs(psi_final_reshaped)**2, axis=(0,1))

plt.figure(figsize=(7, 4.5))
plt.plot(x_grid, prob_x_final, color='purple', linewidth=2, label="Observer Final State $|\\psi(x)|^2$")
plt.xlabel("Position $x$")
plt.ylabel("Probability Density")
plt.title("Final Observer State: A Macroscopic Superposition")
plt.legend()
plt.grid(alpha=0.4)
plt.savefig('figure_final_distribution.pdf')
plt.close()

print("All 5 publication-ready PDFs generated successfully!")
