import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Permet d'importer src/bar_1d.py quel que soit le répertoire d'exécution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bar_1D import generate_uniform_mesh, solve_bar_1d


def u_exact(x, L, E, A, b0):
    return (b0 * x / (E * A)) * (L - 0.5 * x)


def sigma_exact(x, L, A, b0):
    return b0 * (L - x) / A


def run_validation():
    L = 2.0        # m
    E = 70e9       # Pa (ordre de grandeur aluminium)
    A = 1e-4       # m^2 (100 mm^2)
    b0 = 5000.0    # N/m

    # 1. Comparaison détaillée sur un maillage modéré
    n_elem_demo = 8
    x_nodes, elements = generate_uniform_mesh(L, n_elem_demo)
    res = solve_bar_1d(
        x_nodes=x_nodes,
        elements=elements,
        E=E,
        A=A,
        b_elem=np.full(n_elem_demo, b0),
        dirichlet_bc={0: 0.0},
    )

    u_ref = u_exact(x_nodes, L, E, A, b0)
    print(res["U"])
    err_u_max = np.max(np.abs(res["U"] - u_ref))
    r_expected = -b0 * L
    r_computed = res["reactions"][0]

    print(f"Réaction en x=0 : calculée = {r_computed:.6f} N | attendue = {r_expected:.6f} N")
    print(f"Erreur max en déplacement (N={n_elem_demo}) : {err_u_max:.3e} m")

    # 2. Étude de convergence en maillage
    mesh_sizes = [4, 8, 16, 32, 64, 128]
    h_vals = []
    err_u_l2 = []
    err_sigma_elem = []

    for n_el in mesh_sizes:
        x_n, elems = generate_uniform_mesh(L, n_el)
        out = solve_bar_1d(
                x_nodes=x_n,
                elements=elems,
                E=E,
                A=A,
                b_elem=np.full(n_el, b0),
                dirichlet_bc={0: 0.0},
            )
        h = L / n_el
        h_vals.append(h)
        
        # Erreur relative discrète sur les déplacements nodaux
        u_ref = u_exact(x_n, L, E, A, b0)
        rel_err_u = np.linalg.norm(u_ref - out["U"])/np.linalg.norm(u_ref)
        err_u_l2.append(rel_err_u)

        # Erreur relative sur les contraintes évaluées au centre des éléments
        sig_ex_c = sigma_exact(out["x_centers"], L, A, b0)
        rel_err_sig = np.linalg.norm(out["stresses"] - sig_ex_c) / np.linalg.norm(sig_ex_c)
        err_sigma_elem.append(rel_err_sig)

    h_vals = np.array(h_vals)
    err_u_l2 = np.array(err_u_l2)
    err_sigma_elem = np.array(err_sigma_elem)

    # Pente en échelle log-log (ordre de convergence)
    p_u = np.polyfit(np.log(h_vals), np.log(err_u_l2), 1)[0]
    p_sig = np.polyfit(np.log(h_vals), np.log(err_sigma_elem), 1)[0]
    print(f"Ordre de convergence déplacement : ~ {p_u:.2f}")
    print(f"Ordre de convergence contrainte (au centre) : ~ {p_sig:.2f}")

    # 3. Tracé récapitulatif
    x_fine = np.linspace(0.0, L, 300)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Déplacements
    axes[0].plot(x_fine, u_exact(x_fine, L, E, A, b0) * 1e3, "k-", label="Analytique")
    axes[0].plot(x_nodes, res["U"] * 1e3, "o--", label=f"EF P1 ({n_elem_demo} él.)")
    axes[0].set_title("Déplacement axial u(x)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("u [mm]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Contraintes (constantes par élément)
    axes[1].plot(x_fine, sigma_exact(x_fine, L, A, b0) / 1e6, "k-", label="Analytique")
    for e, (i, j) in enumerate(elements):
        sig_mpa = res["stresses"][e] / 1e6
        axes[1].plot([x_nodes[i], x_nodes[j]], [sig_mpa, sig_mpa], "s-", color="tab:orange")
    axes[1].plot([], [], "s-", color="tab:orange", label=f"EF P1 ({n_elem_demo} él., constant/él.)")
    axes[1].set_title("Contrainte axiale σ(x)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("σ [MPa]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Convergence
    axes[2].loglog(h_vals, err_u_l2, "o-", label=f"Déplacement (pente ≈ {p_u:.2f})")
    axes[2].loglog(h_vals, err_sigma_elem, "s-", label=f"Contrainte centre (pente ≈ {p_sig:.2f})")
    axes[2].set_title("Convergence en fonction de h")
    axes[2].set_xlabel("Taille d'élément h [m]")
    axes[2].set_ylabel("Erreur relative")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_validation()



"""
#test avec une forca axiale en bout de poutre
L = 2.0        # m
E = 70e9       # Pa (ordre de grandeur aluminium)
A = 1e-4       # m^2 (100 mm^2)
b0 = 5000    # N/m
n_elem = 16

noeuds, elem = generate_uniform_mesh(L, n_elem)


out = solve_bar_1d(
            x_nodes=noeuds,
            elements=elem,
            E= E,
            A=A,
            b_elem=np.full(n_elem , b0),
            dirichlet_bc={0: 0.0},
            point_loads={n_elem: 0}
        )

print("Déplacements nodaux : ", out["U"])
print("Contraintes par élément : ", out["stresses"])
print("Déformations par élément : ", out["strains"])
"""

