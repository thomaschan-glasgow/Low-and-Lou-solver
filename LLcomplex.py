import sys
import numpy as np
from scipy.integrate import solve_ivp

# ------------------------------------------------------------
# UFiT path
# ------------------------------------------------------------
UFiT_home_directory = "C:/Users/thoma/Documents/UFiT"
sys.path.append(UFiT_home_directory)
from UFiT_Functions_Python import write_B_file

# ============================================================
# =================  Low & Lou NLFFF  ======================
# ============================================================

def solve_P_shooting(n, m, tol=1e-8, max_iter=50):
    """
    Solve Low & Lou BVP using robust shooting + Newton-Raphson on a^2.
    Returns sol_P(mu) function and positive eigenvalue a2.
    """
    def ode(mu, y, a2):
        P, dP = y
        d2P = -(n*(n+1)*P + a2*(n+1)/n * np.sign(P) * np.abs(P)**(1 + 2/n)) / max(1 - mu**2, 1e-12)
        return [dP, d2P]

    # Initial conditions at mu=0
    if m % 2 == 0:  # even solution
        P0, dP0 = 1.0, 0.0
    else:           # odd solution
        P0, dP0 = 0.0, 1.0

    # Initial guess for a2
    a2 = 0.5
    delta = 1e-6
    mu_eval = np.linspace(0, 1-1e-12, 10)

    for i in range(max_iter):
        sol_pos = solve_ivp(lambda mu, y: ode(mu, y, a2),
                            [0, 1-1e-12], [P0, dP0],
                            t_eval=mu_eval, rtol=1e-8, atol=1e-10)
        P_end = sol_pos.y[0, -1]

        # derivative wrt a2
        sol_pos_d = solve_ivp(lambda mu, y: ode(mu, y, a2+delta),
                              [0, 1-1e-12], [P0, dP0],
                              t_eval=mu_eval, rtol=1e-8, atol=1e-10)
        P_end_d = sol_pos_d.y[0, -1]

        dres_da2 = (P_end_d - P_end)/delta
        a2_new = a2 - P_end/dres_da2
        a2_new = max(a2_new, 1e-8)  # ensure positive eigenvalue

        if abs(a2_new - a2) < tol:
            a2 = a2_new
            break
        a2 = a2_new
    else:
        raise RuntimeError("Shooting method did not converge")

    print(f"Shooting method converged: a2={a2:.8f}")

    # Full solution for mu in [0,1]
    mu_pos = np.linspace(0, 1, 256)
    sol_pos = solve_ivp(lambda mu, y: ode(mu, y, a2),
                        [0, 1], [P0, dP0],
                        t_eval=mu_pos, rtol=1e-8, atol=1e-10)
    P_pos, dP_pos = sol_pos.y

    # Mirror to negative mu using symmetry
    if m % 2 == 0:
        P_neg = P_pos[::-1]
        dP_neg = -dP_pos[::-1]
    else:
        P_neg = -P_pos[::-1]
        dP_neg = dP_pos[::-1]

    P_full = np.concatenate([P_neg, P_pos[1:]])
    dP_full = np.concatenate([dP_neg, dP_pos[1:]])
    mu_full = np.linspace(-1, 1, len(P_full))

    def sol_P(mu_query):
        return np.interp(mu_query, mu_full, P_full), np.interp(mu_query, mu_full, dP_full)

    return sol_P, a2


def get_low_lou_field(n=3, m=1, l=0.3, psi=4*np.pi/5,
                       resolution=320, bounds=[-1,1,-1,1,0,1]):
    """
    Generate 3D Low & Lou NLFFF field.
    Returns coords (Nx,Ny,Nz,3) and B-field (Nx,Ny,Nz,3)
    """
    sol_P, a2 = solve_P_shooting(n, m)

    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[2], bounds[3], resolution)
    z = np.linspace(bounds[4], bounds[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Rotate domain by psi around Y-axis
    Xp = X * np.cos(psi) - (Z + l) * np.sin(psi)
    Yp = Y
    Zp = X * np.sin(psi) + (Z + l) * np.cos(psi)

    r = np.sqrt(Xp**2 + Yp**2 + Zp**2) + 1e-12
    theta = np.arccos(np.clip(Zp / r, -1, 1))
    phi = np.arctan2(Yp, Xp)
    mu = np.cos(theta)

    P, dP_dmu = sol_P(mu)

    A = P / r**n
    dA_dtheta = -np.sin(theta) * dP_dmu / r**n
    dA_dr = -n * P / r**(n+1)
    Q = np.sqrt(a2) * A * np.abs(A)**(1/n)

    # Spherical B components
    Br = dA_dtheta / (r**2 * np.sin(theta))
    Btheta = -dA_dr / (r * np.sin(theta))
    Bphi = Q / (r * np.sin(theta))

    # Convert to Cartesian
    BX = Br*np.sin(theta)*np.cos(phi) + Btheta*np.cos(theta)*np.cos(phi) - Bphi*np.sin(phi)
    BY = Br*np.sin(theta)*np.sin(phi) + Btheta*np.cos(theta)*np.sin(phi) + Bphi*np.cos(phi)
    BZ = Br*np.cos(theta) - Btheta*np.sin(theta)

    # Rotate back
    Bx = BX * np.cos(psi) + BZ * np.sin(psi)
    By = BY
    Bz = -BX * np.sin(psi) + BZ * np.cos(psi)

    B = np.stack([Bx, By, Bz], axis=-1)
    coords = np.stack([X, Y, Z], axis=-1)

    return coords, B


# ============================================================
# ======================   MAIN   ============================
# ============================================================

coords, B = get_low_lou_field(
    n=1, m=1, l=0.3, psi=np.pi/4,
    resolution=320, bounds=[-2,2,-2,2,0,1]
)

# UFiT grid vectors
x = coords[:,0,0,0]
y = coords[0,:,0,1]
z = coords[0,0,:,2]

B_ufit = np.transpose(B, (3,0,1,2))

write_B_file(x, y, z, B_ufit, "LL.bin")

print("Low & Lou NLFFF field written for UFiT")
print("B shape:", B_ufit.shape)
print("B min/max:", B.min(), B.max())
