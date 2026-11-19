# Solveur Éléments Finis — Élasticité Linéaire

Développement progressif d'un solveur éléments finis en Python appliqué à l'élasticité linéaire, en commençant par la **barre 1D** avant d'étendre le cadre vers la **2D (contraintes / déformations planes)**.

## Objectifs du projet
- Comprendre et implémenter les étapes clés de la méthode des éléments finis : discrétisation, formulation élémentaire, assemblage matriciel, prise en compte des conditions aux limites et calcul des réactions d'appui.
- Vérifier le solveur sur des cas analytiques connus et analyser la **convergence en maillage**.
- Construire une base modulaire et documentée, facile à faire évoluer.

## État d'avancement
- [x] Cadrage mathématique et arborescence du dépôt
- [ ] Implémentation du solveur barre 1D (`src/bar_1d.py`)
- [ ] Validation analytique et étude de convergence (`validation/test_bar_1d.py`)
- [ ] Extension vers l'élasticité plane 2D

## Structure du dépôt
- `src/bar_1d.py` : assemblage de la rigidité, forces linéiques/ponctuelles, conditions de Dirichlet, résolution et post-traitement.
- `validation/test_bar_1d.py` : comparaison aux solutions exactes et suivi de l'erreur en fonction du raffinement du maillage.
- `assets/` : graphiques de validation et illustrations.

## Prérequis et exécution
```bash
pip install numpy matplotlib
python validation/test_bar_1d.py