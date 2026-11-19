import numpy as np


def generate_uniform_mesh(length: float, n_elements: int) -> tuple[np.ndarray, np.ndarray]:
    #Génère un maillage 1D régulier sur [0, length].
    if length <= 0:
        raise ValueError("La longueur de la barre doit être strictement positive.")
    if n_elements < 1:
        raise ValueError("Le nombre d'éléments doit être au moins égal à 1.")

    x_nodes = np.linspace(0.0, length, n_elements + 1)
    elements = np.column_stack((np.arange(n_elements), np.arange(1, n_elements + 1))) #un élément est défini par les coordonnées des noeuds qui l'entourent
    return x_nodes, elements


def solve_bar_1d(
    x_nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    A: float,
    b_elem: np.ndarray,
    dirichlet_bc: dict[int, float],
    point_loads: dict[int, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Résout un problème de barre 1D en élasticité linéaire (éléments P1 à 2 nœuds).

    Paramètres
    ----------
    x_nodes : (n_nodes,) coordonnées x des nœuds
    elements : (n_elem, 2) connectivité [noeud_i, noeud_j]
    E : module d'Young supposé constant
    A : Aire de la section de la poutre (supposée constante)
    b_elem : (n_elem,) densités de force axiale linéique par élément [N/m]
    dirichlet_bc : dict {indice_noeud: deplacement_impose}
    point_loads : dict {indice_noeud: effort_ponctuel_N} (optionnel)

    Retourne
    --------
    dict avec 'U', 'strains', 'stresses', 'reactions', 'x_centers'
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    E = np.asarray(E, dtype=float)
    A = np.asarray(A, dtype=float)
    b_elem = np.asarray(b_elem, dtype=float)

    n_nodes = len(x_nodes)
    n_elem = len(elements)

    if not ( len(b_elem) == n_elem):
        raise ValueError("b_elem doit avoir la même taille que le nombre d'éléments.")

    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)

    # 1. Assemblage élémentaire
    for e in range(n_elem):
        i, j = elements[e]
        Le = x_nodes[j] - x_nodes[i]
        if Le <= 0.0:
            raise ValueError(f"Longueur d'élément non positive sur l'élément {e}: Le = {Le}")

        k_e = (E * A / Le) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        f_e = (b_elem[e] * Le / 2.0) * np.array([1.0, 1.0])

        idx = [i, j]
        K[np.ix_(idx, idx)] += k_e
        F[idx] += f_e

    # 2. Efforts nodaux imposés (Neumann)
    if point_loads:
        for node_idx, force_val in point_loads.items():
            F[node_idx] += force_val

    # Sauvegarde avant réduction pour calculer les réactions exactes ensuite
    K_raw = K.copy()
    F_raw = F.copy()

    # 3. Conditions aux limites de Dirichlet
    all_dofs = np.arange(n_nodes)
    fixed_dofs = np.array(sorted(dirichlet_bc.keys()), dtype=int)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    U = np.zeros(n_nodes, dtype=float)
    for dof, u_val in dirichlet_bc.items():
        U[dof] = u_val

    if len(free_dofs) > 0:
        if len(fixed_dofs) > 0:
            F_free = F[free_dofs] - K[np.ix_(free_dofs, fixed_dofs)] @ U[fixed_dofs]
        else:
            F_free = F[free_dofs]

        K_free = K[np.ix_(free_dofs, free_dofs)]
        U[free_dofs] = np.linalg.solve(K_free, F_free)

    # 4. Post-traitement : réactions, déformations et contraintes par élément
    reactions = K_raw @ U - F_raw
    # Sur les degrés de liberté libres, la réaction résiduelle est ~0 (à la précision machine)
    reactions[free_dofs] = 0.0

    strains = np.zeros(n_elem, dtype=float)
    stresses = np.zeros(n_elem, dtype=float)
    x_centers = np.zeros(n_elem, dtype=float)

    for e in range(n_elem):
        i, j = elements[e]
        Le = x_nodes[j] - x_nodes[i]
        strains[e] = (U[j] - U[i]) / Le                  #calcul des déformations
        stresses[e] = E * strains[e]                     #calcul des contraintes
        x_centers[e] = 0.5 * (x_nodes[i] + x_nodes[j])   #

    return {
        "U": U,
        "strains": strains,
        "stresses": stresses,
        "reactions": reactions,
        "x_centers": x_centers,
    }