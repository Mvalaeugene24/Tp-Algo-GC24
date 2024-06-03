import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial
from typing import Callable, Tuple

def systeme_masse_ressort_amortisseur(x: np.ndarray, t: float, m: float, a: float, k: float, F: Callable[[float], float]) -> np.ndarray:
    """
    Fonction décrivant le système masse-ressort-amortisseur.
    Args:
        x (np.ndarray): Vecteur d'état [position, vitesse].
        t (float): Temps.
        m (float): Masse.
        a (float): Coefficient de frottement de l'amortisseur.
        k (float): Constante de raideur du ressort.
        F (Callable[[float], float]): Force extérieure en fonction du temps.
    Returns:
        np.ndarray: Dérivées du vecteur d'état.
    """
    dxdt = np.zeros_like(x)
    dxdt[0] = x[1]
    dxdt[1] = (F(t) - a * x[1] - k * x[0]) / m
    return dxdt

def force_exterieure(t: float, F0: float, w: float) -> float:
    """
    Fonction décrivant la force extérieure en fonction du temps.
    Args:
        t (float): Temps.
        F0 (float): Amplitude de la force.
        w (float): Fréquence de la force.
    Returns:
        float: Valeur de la force à l'instant t.
    """
    return F0 * np.cos(w * t)

def resoudre_systeme(
    m: float, a: float, k: float, x_init: np.ndarray, t: np.ndarray, F: Callable[[float], float]
) -> np.ndarray:
    """
    Résout le système d'équations différentielles.
    Args:
        m (float): Masse.
        a (float): Coefficient de frottement.
        k (float): Constante de raideur.
        x_init (np.ndarray): Vecteur d'état initial.
        t (np.ndarray): Tableau des temps.
        F (Callable[[float], float]): Force extérieure.
    Returns:
        np.ndarray: Solution du système.
    """
    return odeint(systeme_masse_ressort_amortisseur, x_init, t, args=(m, a, k, F))

def plot_reponse_systeme(t: np.ndarray, x: np.ndarray, titre: str) -> None:
    """
    Affiche la réponse du système.
    Args:
        t (np.ndarray): Tableau des temps.
        x (np.ndarray): Solutions du système (position et vitesse).
        titre (str): Titre du graphique.
    """
    plt.figure()
    plt.plot(t, x[:, 0], label='Position (m)')
    plt.plot(t, x[:, 1], label='Vitesse (m/s)')
    plt.xlabel('Temps (s)')
    plt.ylabel('Valeurs')
    plt.legend()
    plt.title(titre)
    plt.show()

def calculer_energies(t: np.ndarray, x: np.ndarray, m: float, k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les énergies cinétique, potentielle et mécanique.
    Args:
        t (np.ndarray): Tableau des temps.
        x (np.ndarray): Solutions du système (position et vitesse).
        m (float): Masse.
        k (float): Constante de raideur.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Energies cinétique, potentielle et mécanique.
    """
    E_cinetique = 0.5 * m * x[:, 1]**2
    E_potentielle = 0.5 * k * x[:, 0]**2
    E_mecanique = E_cinetique + E_potentielle
    return E_cinetique, E_potentielle, E_mecanique

def plot_energies(t: np.ndarray, E_cinetique: np.ndarray, E_potentielle: np.ndarray, E_mecanique: np.ndarray) -> None:
    """
    Affiche les énergies cinétique, potentielle et mécanique.
    Args:
        t (np.ndarray): Tableau des temps.
        E_cinetique (np.ndarray): Energie cinétique.
        E_potentielle (np.ndarray): Energie potentielle.
        E_mecanique (np.ndarray): Energie mécanique.
    """
    plt.figure()
    plt.plot(t, E_cinetique, label='Energie cinétique')
    plt.plot(t, E_potentielle, label='Energie potentielle')
    plt.plot(t, E_mecanique, label='Energie mécanique')
    plt.xlabel('Temps (s)')
    plt.ylabel('Energie (J)')
    plt.legend()
    plt.title('Energies cinétique, potentielle et mécanique')
    plt.show()

# Paramètres du système
m = 10.0  # Masse en kg
a = 20.0  # Coefficient de frottement de l'amortisseur en Ns/m
k = 4000.0  # Constante de raideur du ressort en N/m
x0 = 0.01  # Position initiale en m
v0 = 0.0  # Vitesse initiale en m/s

# Fonction pour la force extérieure F(t)
F0 = 100.0  # Amplitude de la force en N
w = 10.0  # Fréquence de la force en rad/s

# Intervalle de temps
t_start = 0.0
t_end = 10.0  # Temps final
dt = 0.01  # Pas de temps
t = np.arange(t_start, t_end, dt)

# Vecteur d'état initial
x_init = np.array([x0, v0])

# Cas a) Les oscillations sont libres (force extérieure nulle)
F_null = lambda t: 0.0
x_a = resoudre_systeme(m, a, k, x_init, t, F_null)

# Cas b) Une force extérieure F(t) = F0*cos(wt)
F_ext = partial(force_exterieure, F0=F0, w=w)
x_b = resoudre_systeme(m, a, k, x_init, t, F_ext)

# Plot de la réponse du système pour les deux cas
plot_reponse_systeme(t, x_a, 'Réponse du système masse-ressort-amortisseur (Cas a)')
plot_reponse_systeme(t, x_b, 'Réponse du système masse-ressort-amortisseur (Cas b)')

# Calcul des énergies pour le cas a)
E_cinetique_a, E_potentielle_a, E_mecanique_a = calculer_energies(t, x_a, m, k)

# Plot des énergies
plot_energies(t, E_cinetique_a, E_potentielle_a, E_mecanique_a)
