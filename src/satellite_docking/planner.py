import ast
from dataclasses import dataclass, field
from typing import Union
from functools import lru_cache
import heapq
import numpy as np

try:
    from rtree import index as rtree_index

    HAS_RTREE = True
except Exception:
    HAS_RTREE = False

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)

from pdm4ar.exercises.ex13.discretization import *
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams

from dataclasses import dataclass


@dataclass
class ObstacleSlackDebug:
    t_grid: np.ndarray  # (N+1,)
    slack: np.ndarray  # (N+1,) pour 1 astéroïde
    dist_model: np.ndarray  # (N+1,) distance modèle sat-ast
    R_safe: float  # rayon de sécurité


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    # weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time
    weight_p: NDArray = field(default_factory=lambda: 2 * np.array([[1.0]]).reshape((1, -1)))

    tr_radius: float = 4  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # tf_max: float = 60.0  # exemple
    tf_max: float = 40  # exemple

    tf_min: float = 5  # <-- AJOUT

    pos_tolerance: float = 0.5
    dir_tolerance: float = 0.1
    vel_tolerance: float = 0.2
    w_u: float = 1  # cetait 1
    w_t: float = 5  # cetait 1
    w_du: float = 0.0  # poids de continuité des forces (à tuner)
    N_obs_sub: int = 1  # sous-échantillonnage astéroïdes seulement
    alpha_x: float = 1.0
    alpha_u: float = 1.0
    alpha_p: float = 1.0

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-3  # cetait 1e-5  # Stopping criteria constant


class SatellitePlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    satellite: SatelliteDyn
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
    ):
        """
        Initialise le planificateur, l'intégrateur et les variables CVXPY.
        """
        # 1. Stockage des informations de l'environnement et du satellite
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp

        # 1) Choix de tf_max / tf_min en fonction des astéroïdes
        if len(self.asteroids) == 0:
            # config sans astéroïdes : horizon plus court, plus agressif
            tf_max = 40.0
            tf_min = 5.0
        else:
            # config avec astéroïdes : on laisse plus de temps pour éviter proprement
            tf_max = 60.0
            tf_min = 15.0

        # Debug container pour les slacks d'obstacles
        self.obs_slack_debug: dict[PlayerName, ObstacleSlackDebug] = {}
        # 2. Chargement des paramètres du solveur
        self.params = SolverParameters()
        self.tr_radius = self.params.tr_radius
        self.last_linear_cost = None
        # 3. Initialisation de la dynamique du satellite
        self.satellite = SatelliteDyn(self.sg, self.sp)

        # 4. Initialisation de l'intégrateur (Discrétisation)
        # K est le nombre de points, N_sub est la précision de l'intégration ODE
        self.integrator = ZeroOrderHold(self.satellite, self.params.K, self.params.N_sub)

        # 5. Vérification de la dynamique (CRITIQUE)
        # Si cela échoue, l'intégrateur ne correspond pas au modèle physique
        if not self.integrator.check_dynamics():
            raise ValueError("Dynamics check failed: The integrator does not match the satellite dynamics.")
        else:
            print("Dynamics check passed.")

        # 6. Allocation des variables d'optimisation CVXPY
        # (X, U, p, et les variables de 'slack' statiques)
        self.variables = self._get_variables()

        # 7. Allocation des paramètres CVXPY
        # (Matrices A_bar, B_bar, conditions initiales/finales, etc.)
        self.problem_parameters = self._get_problem_parameters()

        # 8. Initialisation de la trajectoire de référence (Guess)
        # X_bar et U_bar à zéro pour l'instant (seront écrasés par initial_guess)
        self.X_bar = np.zeros((self.satellite.n_x, self.params.K))
        self.U_bar = np.zeros((self.satellite.n_u, self.params.K))

        # CORRECTION IMPORTANTE : p_bar ne doit PAS être 0 initialement.
        # Si p=0, dt = 0, et la linéarisation (matrices A, B) peut contenir des NaNs ou infinis.
        # On l'initialise à la moitié du temps max ou une valeur raisonnable (>0).
        safe_initial_time = self.params.tf_max / 2.0
        self.p_bar = np.array([safe_initial_time])

        # Le problème CVXPY sera construit dynamiquement dans compute_trajectory
        # car les contraintes d'obstacles (nu_obs) changent à chaque itération.
        self.problem = None

    def _log_slacks(self, it: int):
        """Print slack stats for debugging (read-only)."""

        def stats(name, arr):
            if arr is None:
                print(f"  {name}: None")
                return
            a = np.asarray(arr)
            print(
                f"  {name}: "
                f"L1={np.sum(np.abs(a)):.3e}, "
                f"L2={np.linalg.norm(a):.3e}, "
                f"max={np.max(a):.3e}, "
                f"min={np.min(a):.3e}"
            )

        print(f"\n--- Slack diagnostics @ iter {it} ---")

        stats("nu_dyn", self.variables["nu_dyn"].value)
        stats("nu_goal", self.variables["nu_goal"].value)
        stats("nu_dir", self.variables["nu_dir"].value)
        stats("nu_vel", self.variables["nu_vel"].value)
        stats("nu_u", self.variables["nu_u"].value)

        nu_obs_var = self.variables.get("nu_obs", None)
        if nu_obs_var is not None:
            stats("nu_obs", nu_obs_var.value)
        else:
            print("  nu_obs: (not active)")

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Main SCvx loop (compatible avec:
        - _compute_rho ETH-compatible (pas de last_linear_cost)
        - _check_convergence(J_bar, L_star, X_star, p_star)
        """

        # ------------------------------------------------------------
        # 1) Store states
        # ------------------------------------------------------------
        self.init_state = init_state
        self.goal_state = goal_state
        self.tr_radius = self.params.tr_radius

        # ------------------------------------------------------------
        # 2) Set goal parameters
        # ------------------------------------------------------------
        self._set_goal()

        # ------------------------------------------------------------
        # 3) Initial guess
        # ------------------------------------------------------------
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess(init_state, goal_state)

        # Construire init_state param UNE FOIS à partir du vrai état
        x0 = np.array(
            [
                [init_state.x],
                [init_state.y],
                [init_state.psi],
                [init_state.vx],
                [init_state.vy],
                [init_state.dpsi],
            ]
        )
        self.problem_parameters["init_state"].value = x0

        # S’assurer que la première colonne de la référence correspond bien à l’état réel
        self.X_bar[:, 0] = x0.reshape(-1)

        # ------------------------------------------------------------
        # 4) Initialize SCvx loop
        # ------------------------------------------------------------
        max_iter = self.params.max_iterations
        converged = False

        for it in range(max_iter):

            # (a) Convexification step
            self._convexification()

            # (b) Build obstacle constraints (autour de X_bar,p_bar)
            self._build_obstacle_constraints(self.X_bar, float(self.p_bar[0]))

            # (c) Build the CVX problem
            constraints = self._get_constraints()
            objective = self._get_objective()
            self.problem = cvx.Problem(objective, constraints)

            # (d) Solve convex subproblem
            try:
                _ = self.problem.solve(verbose=False, solver=self.params.solver)
            except Exception:
                # Solver exception -> on sort de la boucle
                break

            self._log_slacks(it)

            if self.problem.status in ["infeasible", "infeasible_inaccurate"]:
                # Sous-problème infaisable -> on sort
                break

            # (e) Extract new solution
            X_new = self.variables["X"].value
            U_new = self.variables["U"].value
            p_new = float(self.variables["p"].value)

            # (f) Compute accuracy ratio ρ (ETH-compatible)
            rho = self._compute_rho(X_new, U_new, p_new)

            # (g) Update trust region
            accept = self._update_trust_region(rho)

            # (h) Accept / Reject
            if not accept:
                continue

            # (i) Convergence check AVANT overwrite bar
            J_bar = self._calculate_nonlinear_cost(self.X_bar, self.U_bar, self.p_bar)
            J_star = self._calculate_nonlinear_cost(X_new, U_new, np.array([p_new]))
            L_star = self.problem.value

            if self._check_convergence(J_bar, J_star, L_star, X_new, p_new):
                self.X_bar = X_new
                self.U_bar = U_new
                self.p_bar = np.array([p_new])
                converged = True
                break

            # sinon on accepte vraiment et on continue
            self.X_bar = X_new
            self.U_bar = U_new
            self.p_bar = np.array([p_new])

        # ------------------------------------------------------------
        # 5-bis) Reconstruire les "slacks" géométriques d’obstacles
        #        pour debug / comparaison avec collision_model
        # ------------------------------------------------------------
        try:
            tf = float(self.p_bar[0])
            if tf > 0.0:
                X_final = self.X_bar  # shape (nx, K)
                K = X_final.shape[1]
                N = K - 1
                t_grid = np.linspace(0.0, tf, N + 1)

                # Position du satellite sur la trajectoire finale
                sat_x = X_final[0, :]  # x
                sat_y = X_final[1, :]  # y

                # Reset ancien debug
                if hasattr(self, "obs_slack_debug"):
                    self.obs_slack_debug.clear()
                else:
                    self.obs_slack_debug = {}

                # Boucle sur les astéroïdes
                for name, ast_params in self.asteroids.items():
                    # Rayon de sécurité consistant avec les contraintes
                    R_safe = self._safe_radius(ast_params)

                    # Position modèle de l’astéroïde sur t_grid
                    cx0, cy0, vx, vy, _ = self._obstacle_base_params(ast_params)
                    ast_x = cx0 + vx * t_grid
                    ast_y = cy0 + vy * t_grid

                    # Distance modèle sur la grille
                    dist_model = np.sqrt((ast_x - sat_x) ** 2 + (ast_y - sat_y) ** 2)

                    # Slack géométrique : positif = safe, négatif = inside
                    slack = dist_model - R_safe

                    self._debug_check_nu_obs_vs_radius(self.variables["X"].value, self.variables["p"].value)
        except Exception:
            # On ignore silencieusement les erreurs de debug slack
            pass

        # ------------------------------------------------------------
        # 5) Export final trajectory
        # ------------------------------------------------------------
        cmds, states = self._extract_seq_from_array()
        return cmds, states

    def _linear_cost_no_slack(self, dX, dU, dp):
        """
        Linearized predicted cost WITHOUT slack terms.
        Used for rho and convergence to stay consistent with _real_cost.
        """
        K = self.params.K
        w_u = self.params.w_u
        w_t = self.params.w_t

        X_lin = self.X_bar + dX
        U_lin = self.U_bar + dU
        p_lin = self._p_scalar(self.p_bar) + float(dp)

        dt = 1 / (K - 1)

        J_u = w_u * dt * np.sum(U_lin**2)
        J_t = w_t * p_lin
        return J_u + J_t

    def _freeze_asteroids_over_time(self, p_guess: float):
        """
        Retourne disks_time[k] = list[(cx, cy, R_safe)] à l'instant tau_k,
        en utilisant systématiquement les vitesses MONDE de _obstacle_base_params.
        """
        K = self.params.K
        disks_time = [[] for _ in range(K)]

        for k in range(K):
            tau = k / (K - 1)
            for ast in self.asteroids.values():
                cx0, cy0, vx, vy, R = self._obstacle_base_params(ast)
                cx = cx0 + vx * tau * p_guess
                cy = cy0 + vy * tau * p_guess
                disks_time[k].append((float(cx), float(cy), float(R)))

        return disks_time

    def _build_rtrees_over_time(self, disks_time):
        """
        Construit une liste rtrees_time[k], un R-tree par tranche de temps k.
        Chaque R-tree indexe les bboxes des disques à tau_k.
        """
        if not HAS_RTREE:
            return None

        rtrees_time = []
        for disks in disks_time:
            p = rtree_index.Property()
            p.dimension = 2
            idx = rtree_index.Index(properties=p)

            for i, (cx, cy, R) in enumerate(disks):
                idx.insert(i, (cx - R, cy - R, cx + R, cy + R))

            rtrees_time.append(idx)

        return rtrees_time

    def _point_in_any_disk(self, x, y, disks, rtree_idx=None):
        """
        Test si (x,y) est dans au moins un disque statique.
        """
        if rtree_idx is not None:
            cand = list(rtree_idx.intersection((x, y, x, y)))
            for i in cand:
                cx, cy, R = disks[i]
                if (x - cx) ** 2 + (y - cy) ** 2 <= R**2:
                    return True
            return False

        # fallback sans rtree
        for cx, cy, R in disks:
            if (x - cx) ** 2 + (y - cy) ** 2 <= R**2:
                return True
        return False

    def _astar_grid_path_time(
        self,
        start_xy,
        goal_xy,
        disks_time,
        rtrees_time=None,
        bounds=(-11.0, 11.0, -11.0, 11.0),
        res=0.5,
    ):
        """
        A* sur grille spatio-temporelle discrète.
        Chaque step avance aussi dans le temps : k_time -> k_time+1.

        - start_xy, goal_xy: (x,y)
        - disks_time[k]: obstacles à tau_k
        - res: pas de grille (m)
        """
        K = self.params.K
        x_min, x_max, y_min, y_max = bounds

        def to_idx(p):
            x, y = p
            ix = int(round((x - x_min) / res))
            iy = int(round((y - y_min) / res))
            return ix, iy

        def to_xy(idx):
            ix, iy = idx
            x = x_min + ix * res
            y = y_min + iy * res
            return x, y

        def in_bounds(ix, iy):
            x, y = to_xy((ix, iy))
            return (x_min <= x <= x_max) and (y_min <= y <= y_max)

        start = to_idx(start_xy)
        goal = to_idx(goal_xy)

        if not in_bounds(*start) or not in_bounds(*goal):
            return [start_xy, goal_xy]

        # start/goal doivent être libres au temps 0 et temps final
        sx, sy = to_xy(start)
        gx, gy = to_xy(goal)
        if self._point_in_any_disk_time(sx, sy, 0, disks_time, rtrees_time):
            return [start_xy, goal_xy]
        if self._point_in_any_disk_time(gx, gy, K - 1, disks_time, rtrees_time):
            return [start_xy, goal_xy]

        # 8-connexité
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        def heuristic(a, b):
            ax, ay = a
            bx, by = b
            return np.hypot(ax - bx, ay - by)

        # heap state: (f_score, (ix,iy), k_time)
        open_heap = []
        heapq.heappush(open_heap, (0.0, start, 0))

        came_from = {}  # key=(node,k) -> prev(node,k)
        g_score = {(start, 0): 0.0}

        while open_heap:
            _, current, k_time = heapq.heappop(open_heap)

            if current == goal:
                # reconstruit chemin (ignore temps, garde positions)
                path = [current]
                key = (current, k_time)
                while key in came_from:
                    key = came_from[key]
                    path.append(key[0])
                path.reverse()
                return [to_xy(i) for i in path]

            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not in_bounds(*nxt):
                    continue

                k2 = min(K - 1, k_time + 1)  # avance dans le temps
                x2, y2 = to_xy(nxt)

                if self._point_in_any_disk_time(x2, y2, k2, disks_time, rtrees_time):
                    continue

                step_cost = np.hypot(dx, dy) * res
                tentative = g_score[(current, k_time)] + step_cost
                key2 = (nxt, k2)

                if key2 not in g_score or tentative < g_score[key2]:
                    came_from[key2] = (current, k_time)
                    g_score[key2] = tentative
                    f = tentative + heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f, nxt, k2))

        # pas de chemin trouvé
        return [start_xy, goal_xy]

    def _point_in_any_disk_time(self, x, y, k, disks_time, rtrees_time=None):
        """
        Test si (x,y) est dans un disque à l'instant discret k.
        """
        disks_k = disks_time[k]
        rtree_k = None if rtrees_time is None else rtrees_time[k]
        return self._point_in_any_disk(x, y, disks_k, rtree_k)

    def _resample_polyline(self, pts, K):
        """
        Re-échantillonne une polyline en K points uniformes en longueur d'arc.
        pts: list[(x,y)]
        return: array shape (K,2)
        """
        pts = np.array(pts, dtype=float)
        if len(pts) == 1:
            return np.repeat(pts, K, axis=0)

        segs = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(segs, axis=1)
        s = np.hstack([[0.0], np.cumsum(seg_len)])
        total = s[-1] + 1e-9

        target_s = np.linspace(0, total, K)
        out = np.zeros((K, 2))

        j = 0
        for i, ts in enumerate(target_s):
            while j < len(s) - 2 and ts > s[j + 1]:
                j += 1
            t = (ts - s[j]) / (s[j + 1] - s[j] + 1e-9)
            out[i] = (1 - t) * pts[j] + t * pts[j + 1]

        return out

    def initial_guess(self, init_state: SatelliteState, goal_state: DynObstacleState):
        K = self.params.K
        nx = self.satellite.n_x
        nu = self.satellite.n_u
        np_p = self.satellite.n_p

        X = np.zeros((nx, K))
        U = np.zeros((nu, K))
        p = np.zeros(np_p)

        # ----- temps guess sûr -----
        p_guess = 0.5 * self.params.tf_max
        p[0] = p_guess

        # ----- start/goal positions -----
        start_xy = (init_state.x, init_state.y)
        goal_xy = (goal_state.x, goal_state.y)

        # ==========================================================
        #  DYNAMIC ASTEROIDS INITIAL GUESS (time-sliced R-trees)
        # ==========================================================
        if len(self.asteroids) > 0:
            # 1) Disques dynamiques au cours du temps
            disks_time = self._freeze_asteroids_over_time(p_guess)

            # 2) R-tree par tranche de temps
            rtrees_time = self._build_rtrees_over_time(disks_time)

            # 3) A* spatio-temporel discret
            bounds = (-11.0, 11.0, -11.0, 11.0)  # mêmes bornes que tes contraintes
            res = 0.5  # taille grille (m)

            path_xy = self._astar_grid_path_time(
                start_xy=start_xy,
                goal_xy=goal_xy,
                disks_time=disks_time,
                rtrees_time=rtrees_time,
                bounds=bounds,
                res=res,
            )
        else:
            path_xy = [start_xy, goal_xy]

        # 4) Re-échantillonnage du chemin A* en K points
        pts = self._resample_polyline(path_xy, K)  # shape (K,2)

        # ----- gestion intelligente de psi -----
        psi_start = init_state.psi
        psi_end = goal_state.psi
        diff = psi_end - psi_start
        delta_psi = (diff + np.pi) % (2 * np.pi) - np.pi
        psi_target_interp = psi_start + delta_psi

        x0 = np.array([init_state.x, init_state.y, psi_start, init_state.vx, init_state.vy, init_state.dpsi])
        xf = np.array([goal_state.x, goal_state.y, psi_target_interp, goal_state.vx, goal_state.vy, goal_state.dpsi])

        # ----- remplissage X̄ : positions = A*, reste linéaire -----
        for k in range(K):
            t = k / (K - 1)
            X[:, k] = (1 - t) * x0 + t * xf
            X[0, k] = pts[k, 0]
            X[1, k] = pts[k, 1]

        # ----- bruit soft (évite singularités) -----
        noise = np.random.normal(0, 0.05, (2, K))
        X[0:2, 1:-1] += noise[:, 1:-1]

        # ----- contrôle initial à zéro -----
        U[:, :] = 0.0

        return X, U, p

    def _set_goal(self):
        xf = np.array(
            [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                self.goal_state.dpsi,
            ]
        )

        # ***** IMPORTANT : reshape en colonne (n,1) *****
        self.problem_parameters["goal_pos"].value = xf[0:2].reshape(2, 1)
        self.problem_parameters["goal_theta"].value = np.array([[xf[2]]])  # shape (1,1)
        self.problem_parameters["goal_vel_lin"].value = xf[3:5].reshape(2, 1)
        self.problem_parameters["goal_vel_ang"].value = np.array([[xf[5]]])  # shape (1,1)

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        n_x = self.satellite.n_x  # = 6 typically
        n_u = self.satellite.n_u  # = 3 typically
        n_p = self.satellite.n_p  # often 1 (final time)
        K = self.params.K

        variables = {
            # --- Trajectory variables ---
            # K discrete states X_0, ..., X_{K-1}
            "X": cvx.Variable((n_x, K)),
            # K control inputs U_0, ..., U_{K-1}
            "U": cvx.Variable((n_u, K)),
            # Parameter(s) such as final time
            "p": cvx.Variable((n_p, 1)),
            # --- Slacks ---
            # Slacks for initial & final control: U(0)=0 and U(tf)=0
            # We need one n_u-dimensional slack for each boundary input.
            "nu_u": cvx.Variable((n_u, 2)),
            # Terminal state slacks:
            #   - position error slack (x,y)  → dimension 2
            #   - orientation slack (psi)    → dimension 1
            #   - velocity slack (vx,vy,dpsi)→ dimension 3
            "nu_goal": cvx.Variable((2, 1)),
            "nu_dir": cvx.Variable((1, 1)),
            "nu_vel": cvx.Variable((3, 1)),
            # Virtual control (dynamic linearization slack):
            # One n_x-dimensional slack for each dynamical interval (K-1)
            "nu_dyn": cvx.Variable((n_x, K - 1)),
        }

        return variables

    def _get_problem_parameters(self) -> dict:

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p
        K = self.params.K

        problem_parameters = {
            # initial state
            "init_state": cvx.Parameter((n_x, 1)),
            # goal components
            "goal_pos": cvx.Parameter((2, 1)),
            "goal_theta": cvx.Parameter((1, 1)),
            "goal_vel_lin": cvx.Parameter((2, 1)),
            "goal_vel_ang": cvx.Parameter((1, 1)),
            # dynamics linearization matrices
            "A_bar": cvx.Parameter((n_x * n_x, K - 1)),
            "B_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "F_bar": cvx.Parameter((n_x * n_p, K - 1)),
            "r_bar": cvx.Parameter((n_x, K - 1)),
        }

        return problem_parameters

    def _build_rect_buffer(self) -> NDArray:
        """
        Rectangle buffer (bounding box) du satellite dans son repère LOCAL.
        On utilise :
            x ∈ [-(w_half + w_panel), +(w_half + w_panel)]
            y ∈ [-l_r, l_f + l_c]
        """
        w_half = self.sg.w_half
        w_panel = self.sg.w_panel
        l_f = self.sg.l_f
        l_c = self.sg.l_c
        l_r = self.sg.l_r

        x_max = w_half + w_panel
        x_min = -x_max
        y_max = l_f + l_c
        y_min = -l_r

        V = np.array(
            [
                [x_min, y_max],  # top-left
                [x_max, y_max],  # top-right
                [x_max, y_min],  # bottom-right
                [x_min, y_min],  # bottom-left
            ]
        )

        return V

    def _build_trapezoid_buffer(self) -> NDArray:
        """
        Buffer trapézoïdal plus précis, toujours dans le repère LOCAL du satellite.

        Géométrie :
            x ∈ [-(w_half + w_panel), +(w_half + w_panel)]
            y_rear = -l_r
            y_front(x) =
                -l_f                   si |x| >= w_half
                -(l_f + l_c) + (l_c / w_half) * |x|   sinon
        """

        w_half = self.sg.w_half
        w_panel = self.sg.w_panel
        l_f = self.sg.l_f
        l_c = self.sg.l_c
        l_r = self.sg.l_r

        x_max = w_half + w_panel
        x_min = -x_max

        x_L = -w_half
        x_R = w_half

        def y_front(x: float) -> float:
            if abs(x) >= w_half:
                return l_f
            return (l_f + l_c) - (l_c / w_half) * abs(x)

        V = np.array(
            [
                [x_min, y_front(x_min)],  # front-left outer
                [x_L, y_front(x_L)],  # front-left inner
                [x_R, y_front(x_R)],  # front-right inner
                [x_max, y_front(x_max)],  # front-right outer
                [x_max, -l_r],  # rear-right
                [x_min, -l_r],  # rear-left
            ]
        )

        return V

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_obstacle_symbolics():
        """
        Génère symboliquement :
            s(x,y,p) = -((x - cx)^2 + (y - cy)^2) + R^2
        où :
            cx = cx0 + vx * (k/K) * p
            cy = cy0 + vy * (k/K) * p

        Retourne les lambdas NumPy :
            s_f, ds_dx_f, ds_dy_f, ds_dp_f
        """
        x, y, p = spy.symbols("x y p", real=True)
        cx0, cy0 = spy.symbols("cx0 cy0", real=True)
        vx, vy = spy.symbols("vx vy", real=True)
        k, K = spy.symbols("k K", real=True)
        R = spy.symbols("R", real=True, positive=True)

        # t_k = k * p / K
        t_k = k * p / (K - 1)
        cx = cx0 + vx * t_k
        cy = cy0 + vy * t_k

        s = -((x - cx) ** 2 + (y - cy) ** 2) + R**2

        ds_dx = spy.diff(s, x)
        ds_dy = spy.diff(s, y)
        ds_dp = spy.diff(s, p)

        args = (x, y, p, cx0, cy0, vx, vy, k, K, R)

        s_f = spy.lambdify(args, s, "numpy")
        ds_dx_f = spy.lambdify(args, ds_dx, "numpy")
        ds_dy_f = spy.lambdify(args, ds_dy, "numpy")
        ds_dp_f = spy.lambdify(args, ds_dp, "numpy")

        return s_f, ds_dx_f, ds_dy_f, ds_dp_f

    def _safe_radius(self, obs) -> float:
        base_buf = float(np.max(np.linalg.norm(self._build_trapezoid_buffer(), axis=1)))

        planet_scale = 0.7
        asteroid_scale = 0.7
        margin_planet = 0.3
        margin_asteroid = 0.3

        if isinstance(obs, PlanetParams):
            return obs.radius + planet_scale * base_buf + margin_planet
        if isinstance(obs, AsteroidParams):
            return obs.radius + asteroid_scale * base_buf + margin_asteroid

        raise ValueError("Unknown obstacle type.")

    def _obstacle_base_params(self, obs):
        """
        Returns cx0, cy0, vx, vy, R_safe
        """
        base_buf = float(np.max(np.linalg.norm(self._build_trapezoid_buffer(), axis=1)))

        # Marges distinctes
        # margin_planet = 0.3
        # margin_asteroid = 0.05  # ← ENCORE PLUS GRAND

        # Échelles pour la taille du satellite
        # planet_scale = 0.7
        # asteroid_scale = 1 # ← GROS DISQUE AUTOUR DES ASTÉROÏDES

        if isinstance(obs, PlanetParams):
            cx0, cy0 = obs.center
            vx, vy = 0.0, 0.0
            # R_safe = obs.radius + planet_scale * base_buf + margin_planet
            R_safe = self._safe_radius(obs)

            return float(cx0), float(cy0), float(vx), float(vy), float(R_safe)

        if isinstance(obs, AsteroidParams):
            x0, y0 = obs.start
            vx_local, vy_local = obs.velocity
            theta = obs.orientation  # orientation de l’astéroïde

            c = np.cos(theta)
            s = np.sin(theta)

            # rotation du repère local → repère monde
            vx = c * vx_local - s * vy_local
            vy = s * vx_local + c * vy_local

            R_safe = self._safe_radius(obs)

            return float(x0), float(y0), float(vx), float(vy), float(R_safe)

        raise ValueError("Unknown obstacle type.")

    def _build_obstacle_constraints(self, X_bar, p_bar):

        self.C_list = []
        self.G_list = []
        self.rp_list = []
        self.k_list = []  # k de départ
        self.k_next_list = []  # k de fin (même k si point noeud)
        self.theta_list = []  # poids d'interpolation
        self.obs_ref_list = []  # 🔥 nouvel array : obstacle associé

        K = self.params.K
        M = self.params.N_obs_sub

        planets = list(self.planets.values())
        asteroids = list(self.asteroids.values())

        def obs_center(cx0, cy0, vx, vy, tau, p):
            return cx0 + vx * tau * p, cy0 + vy * tau * p

        # ---------------------------
        # A) PLANÈTES : seulement noeuds
        # ---------------------------
        for k in range(K):
            tau = k / (K - 1)
            xbar_interp = X_bar[:, k]
            xk = xbar_interp[0]
            yk = xbar_interp[1]

            for planet in planets:
                cx0, cy0, vx, vy, R = self._obstacle_base_params(planet)  # vx=vy=0 ici
                cx, cy = obs_center(cx0, cy0, vx, vy, tau, p_bar)

                dx = xk - cx
                dy = yk - cy
                sval = -(dx**2 + dy**2) + R**2

                dsx = -2 * dx
                dsy = -2 * dy
                dsp = 2 * tau * (dx * vx + dy * vy)  # =0 pour planètes

                Ck = np.zeros((1, self.satellite.n_x))
                Ck[0, 0] = dsx
                Ck[0, 1] = dsy
                Gk = float(dsp)

                rp = sval - (Ck @ xbar_interp) - Gk * p_bar

                self.C_list.append(Ck)
                self.G_list.append(Gk)
                self.rp_list.append(rp)
                self.k_list.append(k)
                self.k_next_list.append(k)  # même noeud
                self.theta_list.append(0.0)  # pas d'interpolation
                self.obs_ref_list.append(planet)

        # ---------------------------
        # B) ASTÉROÏDES : noeuds + sous-échantillonnage
        # ---------------------------

        # B1) noeuds
        for k in range(K):
            tau = k / (K - 1)
            xbar_interp = X_bar[:, k]
            xk = xbar_interp[0]
            yk = xbar_interp[1]

            for ast in asteroids:
                cx0, cy0, vx, vy, R = self._obstacle_base_params(ast)
                cx, cy = obs_center(cx0, cy0, vx, vy, tau, p_bar)

                dx = xk - cx
                dy = yk - cy
                sval = -(dx**2 + dy**2) + R**2

                dsx = -2 * dx
                dsy = -2 * dy
                dsp = 2 * tau * (dx * vx + dy * vy)

                Ck = np.zeros((1, self.satellite.n_x))
                Ck[0, 0] = dsx
                Ck[0, 1] = dsy
                Gk = float(dsp)

                rp = float(sval - (Ck @ xbar_interp) - Gk * p_bar)

                self.C_list.append(Ck)
                self.G_list.append(Gk)
                self.rp_list.append(rp)
                self.k_list.append(k)
                self.k_next_list.append(k)
                self.theta_list.append(0.0)
                self.obs_ref_list.append(planet)
                self.obs_ref_list.append(ast)

        # B2) points internes entre k et k+1
        for k in range(K - 1):
            for j in range(1, M):  # internes seulement
                theta = j / M
                tau = (k + theta) / (K - 1)

                xbar_interp = (1 - theta) * X_bar[:, k] + theta * X_bar[:, k + 1]
                xk = xbar_interp[0]
                yk = xbar_interp[1]

                for ast in asteroids:
                    cx0, cy0, vx, vy, R = self._obstacle_base_params(ast)
                    cx, cy = obs_center(cx0, cy0, vx, vy, tau, p_bar)

                    dx = xk - cx
                    dy = yk - cy
                    sval = -(dx**2 + dy**2) + R**2

                    dsx = -2 * dx
                    dsy = -2 * dy
                    dsp = 2 * tau * (dx * vx + dy * vy)

                    Ck = np.zeros((1, self.satellite.n_x))
                    Ck[0, 0] = dsx
                    Ck[0, 1] = dsy
                    Gk = float(dsp)

                    rp = float(sval - (Ck @ xbar_interp) - Gk * p_bar)

                    self.C_list.append(Ck)
                    self.G_list.append(Gk)
                    self.rp_list.append(rp)
                    self.k_list.append(k)
                    self.k_next_list.append(k + 1)
                    self.theta_list.append(theta)
                    self.obs_ref_list.append(planet)
                    self.obs_ref_list.append(ast)

    def _debug_check_nu_obs_vs_radius(self, X_sol, p_sol, tol_geom=1e-3, tol_nu=1e-6):
        """Check numériquement :
        si dist réelle < R_safe  ⇒ nu_obs > 0 (à tol près).
        """
        if "nu_obs" not in self.variables:
            print("[DEBUG OBS] Pas de nu_obs dans self.variables")
            return

        if not hasattr(self, "obs_ref_list") or len(self.obs_ref_list) == 0:
            print("[DEBUG OBS] obs_ref_list absent ou vide (appelle _build_obstacle_constraints avant).")
            return

        nu_obs_val = np.asarray(self.variables["nu_obs"].value).reshape(-1)
        if nu_obs_val is None:
            print("[DEBUG OBS] nu_obs.value est None")
            return

        K = self.params.K
        p_val = float(np.asarray(p_sol).reshape(-1)[0])

        n_inside = 0
        n_inside_nu_zero = 0
        n_outside_nu_pos = 0

        for i in range(len(self.C_list)):
            k = self.k_list[i]
            k2 = self.k_next_list[i]
            theta = self.theta_list[i]
            obs = self.obs_ref_list[i]

            # état interpolé au temps (k + theta)
            xk = X_sol[:, k]
            xk2 = X_sol[:, k2]
            x_aff = (1.0 - theta) * xk + theta * xk2
            x = float(x_aff[0])
            y = float(x_aff[1])

            # temps normalisé
            tau = (k + theta) / (K - 1)

            # paramètres de l'obstacle
            cx0, cy0, vx, vy, R_safe = self._obstacle_base_params(obs)
            cx = cx0 + vx * tau * p_val
            cy = cy0 + vy * tau * p_val

            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            inside = dist < R_safe - tol_geom
            nu_i = nu_obs_val[i]

            if inside:
                n_inside += 1
                if nu_i <= tol_nu:
                    n_inside_nu_zero += 1
                    print(
                        f"[DEBUG OBS] INCONSISTANCE : inside R_safe mais nu_obs≈0 "
                        f"(i={i}, k={k}, theta={theta:.2f}, dist={dist:.3f}, R={R_safe:.3f}, nu={nu_i:.3e})"
                    )
            else:
                if nu_i > tol_nu:
                    n_outside_nu_pos += 1

        print(
            f"[DEBUG OBS] inside: {n_inside}, "
            f"inside & nu≈0: {n_inside_nu_zero}, "
            f"outside & nu>0: {n_outside_nu_pos}"
        )

    def _get_constraints(self) -> list[cvx.Constraint]:
        constraints = []

        # Variables
        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]

        # Slacks
        nu_u = self.variables["nu_u"]
        nu_dyn = self.variables["nu_dyn"]
        nu_goal = self.variables["nu_goal"]
        nu_dir = self.variables["nu_dir"]
        nu_vel = self.variables["nu_vel"]

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p
        K = self.params.K

        # 1) Initial state
        constraints += [X[:, [0]] == self.problem_parameters["init_state"]]

        # 2) Control boundary
        constraints += [
            U[:, 0] == nu_u[:, 0],
            U[:, K - 1] == nu_u[:, 1],
        ]

        # 3) Control bounds & Time
        Fmax = self.sg.F_max
        # constraints += [U <= Fmax, U >= -Fmax, p >= 0, p <= self.params.tf_max]
        # APRÈS
        constraints += [U <= Fmax, U >= -Fmax, p >= self.params.tf_min, p <= self.params.tf_max]  # borne basse
        x_min, x_max = -11.0, 11.0
        y_min, y_max = -11.0, 11.0

        for k in range(K):
            constraints += [
                X[0, k] >= x_min,
                X[0, k] <= x_max,
                X[1, k] >= y_min,
                X[1, k] <= y_max,
            ]

        # 4) Terminal constraints
        x_final = X[:, [K - 1]]
        constraints += [
            cvx.abs(x_final[0:2, :] - self.problem_parameters["goal_pos"]) <= self.params.pos_tolerance + nu_goal,
            cvx.abs(x_final[2, :] - self.problem_parameters["goal_theta"]) <= self.params.dir_tolerance + nu_dir,
            cvx.abs(x_final[3:5, :] - self.problem_parameters["goal_vel_lin"])
            <= self.params.vel_tolerance + nu_vel[0:2],
            cvx.abs(x_final[5, :] - self.problem_parameters["goal_vel_ang"]) <= self.params.vel_tolerance + nu_vel[2],
        ]

        # 5) Dynamics
        for k in range(K - 1):
            A_k = cvx.reshape(self.problem_parameters["A_bar"][:, k], (n_x, n_x), order="F")
            B_k = cvx.reshape(self.problem_parameters["B_bar"][:, k], (n_x, n_u), order="F")
            F_k = cvx.reshape(self.problem_parameters["F_bar"][:, k], (n_x, n_p), order="F")
            r_k = self.problem_parameters["r_bar"][:, [k]]

            dyn_residual = X[:, [k + 1]] - (A_k @ X[:, [k]] + B_k @ U[:, [k]] + F_k @ p + r_k)
            constraints += [cvx.abs(dyn_residual) <= nu_dyn[:, k : k + 1]]  # Vectorized slack constraint

        # 6) Obstacles (CORRECTION ICI)
        # 6) Obstacles (planètes aux noeuds, astéroïdes sous-échantillonnés)
        if hasattr(self, "C_list") and len(self.C_list) > 0:

            nu_obs = cvx.Variable(len(self.C_list))
            self.variables["nu_obs"] = nu_obs

            for i in range(len(self.C_list)):
                k = self.k_list[i]
                k2 = self.k_next_list[i]
                theta = self.theta_list[i]

                Ck = self.C_list[i]
                Gk = self.G_list[i]
                rpk = self.rp_list[i]

                # état interpolé affine
                x_aff = (1 - theta) * X[:, [k]] + theta * X[:, [k2]]

                constraints += [Ck @ x_aff + Gk * p + rpk <= nu_obs[i]]

            constraints += [nu_obs >= 0]

        else:
            # Nettoyage si pas d'obstacles
            if "nu_obs" in self.variables:
                del self.variables["nu_obs"]

        # 7) Trust Region
        # --- TRUST REGION (ECOS-FRIENDLY SINGLE SOC) ---
        # 7) Trust Region
        for k in range(K):
            delta_x = X[:, k] - self.X_bar[:, k]
            delta_u = U[:, k] - self.U_bar[:, k]

            # FIX CRITIQUE : faire de delta_p un vecteur (1,) pour ECOS
            delta_p = cvx.reshape(p - self.p_bar.item(), (1,))

            constraints += [
                cvx.norm(
                    cvx.hstack(
                        [self.params.alpha_x * delta_x, self.params.alpha_u * delta_u, self.params.alpha_p * delta_p]
                    ),
                    2,
                )
                <= self.tr_radius
            ]

        # 8) Slacks positive
        constraints += [
            nu_goal >= 0,
            nu_dir >= 0,
            nu_vel >= 0,
            nu_dyn >= 0,
        ]

        return constraints

    def _get_objective(self):

        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]

        nu_u = self.variables["nu_u"]
        nu_dyn = self.variables["nu_dyn"]
        nu_goal = self.variables["nu_goal"]
        nu_dir = self.variables["nu_dir"]
        nu_vel = self.variables["nu_vel"]

        nu_obs = self.variables.get("nu_obs", None)

        K = self.params.K
        Fmax = self.sg.F_max
        lambda_nu = self.params.lambda_nu

        # ============================
        # 1) CONTROL COST (DCP SAFE)
        # ============================
        # FIX dt to previous iteration
        # 1) CONTROL COST (DCP SAFE)
        dt = float(self.p_bar) / (K - 1)

        # On normalise par Fmax², mais on garde un facteur réglable via params.w_u
        w_u = self.params.w_u / (Fmax**2)

        J_u = w_u * dt * cvx.sum(cvx.sum_squares(U))
        w_du = self.params.w_du / (Fmax**2)  # normalisation comme J_u
        dU = U[:, 1:] - U[:, :-1]  # différences entre pas
        J_du = w_du * cvx.sum_squares(dU)

        # ============================
        # 2) TIME COST (linear)
        # ============================
        J_t = self.params.weight_p @ p

        # ============================
        # 3) SLACKS
        # ============================
        slack_terms = [
            cvx.norm1(nu_dyn),
            cvx.norm1(nu_goal),
            cvx.norm1(nu_dir),
            cvx.norm1(nu_vel),
            cvx.norm1(nu_u),
        ]

        if nu_obs is not None:
            slack_terms.append(cvx.norm1(nu_obs))

        J_slack = lambda_nu * cvx.sum(slack_terms)
        # ============================
        # 4) GOAL TRACKING COST
        # ============================
        x_final = X[:, [K - 1]]
        goal_pos = self.problem_parameters["goal_pos"]
        goal_theta = self.problem_parameters["goal_theta"]
        goal_vel_lin = self.problem_parameters["goal_vel_lin"]
        goal_vel_ang = self.problem_parameters["goal_vel_ang"]

        w_goal_pos = 50.0  # position finale très importante
        w_goal_theta = 20.0  # bien orienté pour le docking
        w_goal_vel = 10.0  # faible vitesse à l'arrivée

        J_goal = (
            w_goal_pos * cvx.sum_squares(x_final[0:2, :] - goal_pos)
            + w_goal_theta * cvx.sum_squares(x_final[2, :] - goal_theta)
            + w_goal_vel
            * (cvx.sum_squares(x_final[3:5, :] - goal_vel_lin) + cvx.sum_squares(x_final[5, :] - goal_vel_ang))
        )
        # après avoir défini X, K
        dX_pos = X[0:2, 1:] - X[0:2, :-1]
        w_path = 0.005  # à tuner
        J_path = w_path * cvx.sum_squares(dX_pos)
        w_vel_traj = 0  # à tuner
        J_vel_traj = w_vel_traj * cvx.sum_squares(X[3:5, :])

        return cvx.Minimize(J_u + J_t + J_slack + J_goal + J_du + J_path + J_vel_traj)

    def _convexification(self):
        """
        Linearize and discretize the dynamics using ZOH only.
        Update CVX parameters A_bar, B_bar, F_bar, r_bar accordingly.
        """

        K = self.params.K
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        # ============================================================
        # 1. DISCRETIZATION (ZOH dynamics linearization)
        # ============================================================

        A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # NOTE: Your integrator returns flattened matrices (column-major)

        # ============================================================
        # 2. UPDATE CVX PARAMETERS
        # ============================================================

        # Initial state
        # self.problem_parameters["init_state"].value = self.X_bar[:, 0].reshape(6, 1)

        # Assign discretized system matrices to CVX parameters
        self.problem_parameters["A_bar"].value = A_bar.reshape((n_x * n_x, K - 1), order="F")

        self.problem_parameters["B_bar"].value = B_bar.reshape((n_x * n_u, K - 1), order="F")

        self.problem_parameters["F_bar"].value = F_bar.reshape((n_x * n_p, K - 1), order="F")

        self.problem_parameters["r_bar"].value = r_bar.reshape((n_x, K - 1), order="F")

    def _p_scalar(self, p) -> float:
        """Return p as a python float whether p is float, (1,), (1,1), etc."""
        return float(np.asarray(p).reshape(-1)[0])

    def _real_cost(self, X, U, p):
        K = self.params.K
        w_u = self.params.w_u
        w_t = self.params.w_t

        p_s = self._p_scalar(p)
        dt = p_s / (K - 1)

        J_u = w_u * dt * np.sum(U**2)
        J_t = w_t * p_s
        return J_u + J_t

    def _linear_cost(self, dX, dU, dp):
        K = self.params.K
        w_u = self.params.w_u
        w_t = self.params.w_t
        lambda_nu = self.params.lambda_nu

        X_lin = self.X_bar + dX
        U_lin = self.U_bar + dU
        p_lin = self._p_scalar(self.p_bar) + float(dp)

        dt = p_lin / (K - 1)

        J_u = w_u * dt * np.sum(U_lin**2)
        J_t = w_t * p_lin

        nu_dyn = self.variables["nu_dyn"].value
        nu_goal = self.variables["nu_goal"].value
        nu_dir = self.variables["nu_dir"].value
        nu_vel = self.variables["nu_vel"].value
        nu_u = self.variables["nu_u"].value

        slack_sum = (
            np.sum(np.abs(nu_dyn))
            + np.sum(np.abs(nu_goal))
            + np.sum(np.abs(nu_dir))
            + np.sum(np.abs(nu_vel))
            + np.sum(np.abs(nu_u))
        )

        nu_obs_var = self.variables.get("nu_obs", None)
        if nu_obs_var is not None and nu_obs_var.value is not None:
            slack_sum += np.sum(np.abs(nu_obs_var.value))

        J_slack = lambda_nu * slack_sum
        return J_u + J_t + J_slack

    def _compute_rho(self, X_new, U_new, p_new):
        """
        rho ETH (slide):
        rho = (J_lambda(bar) - J_lambda(star)) / (J_lambda(bar) - L_lambda(star))

        IMPORTANT:
        - J_lambda(*) = vrai coût pénalisé (original, nonconvexe)
        - L_lambda(*) = coût du modèle convexifié (objectif CVX)
        - au point bar, J_lambda(bar) = L_lambda(bar) si le modèle est construit exact au bar.
        Dans ton cas l’objectif CVX gèle dt et normalise w_u, donc on utilise L_bar pour la
        réduction prédite, pour rester cohérent avec le sous-problème.
        """

        # coût réel sur la trajectoire de référence (bar)
        J_bar = self._calculate_nonlinear_cost(self.X_bar, self.U_bar, self.p_bar)

        # coût réel sur la nouvelle solution
        J_star = self._calculate_nonlinear_cost(X_new, U_new, np.array([p_new]))

        # coût du modèle convexifié (objectif CVX résolu)
        L_star = float(self.problem.value)

        denom = J_bar - L_star
        if abs(denom) <= 1e-6:
            return 0.0

        return (J_bar - J_star) / denom

    def _check_convergence(self, J_bar, J_star, L_star, X_star, p_star):
        """
        Critères d'arrêt SCvx inspirés des slides :
        1) J_bar - L_star <= eps_abs
        2) ||p* - p̄|| + max_k ||x*_k - x̄_k|| <= eps_step
        3) J_bar - J_star <= eps_rel * |J_bar|
        """

        eps_abs = self.params.stop_crit  # ex: 1e-3 plutôt que 1e-5
        eps_step = self.params.stop_crit  # même seuil pour la taille de pas
        eps_rel = getattr(self.params, "stop_crit_rel", 1e-3)

        # --- Critère 1 : amélioration prédite très petite ---
        # (J_bar - L_star) ~ 0  ⇒ plus rien à gagner
        crit1 = (J_bar - L_star) <= eps_abs

        # --- Critère 2 : pas très petit ---
        dp = abs(float(p_star) - float(self.p_bar))
        dx_max = np.max(np.linalg.norm(X_star - self.X_bar, axis=0))
        crit2 = (dp + dx_max) <= eps_step

        # --- Critère 3 : amélioration réelle très petite (relative) ---
        if abs(J_bar) < 1e-12:
            crit3 = False
        else:
            crit3 = (J_bar - J_star) <= eps_rel * abs(J_bar)

        return crit1 or crit2 or crit3

    def _update_trust_region(self, rho):
        """
        Update trust region radius based on SCvx rules.
        Returns: accept (bool)
        """

        # --- FIX: use an internal mutable trust-region radius ---
        if not hasattr(self, "tr_radius") or self.tr_radius is None:
            self.tr_radius = self.params.tr_radius
        eta = self.tr_radius
        # --------------------------------------------------------

        eta_min = self.params.min_tr_radius
        eta_max = self.params.max_tr_radius

        rho0 = self.params.rho_0
        rho1 = self.params.rho_1
        rho2 = self.params.rho_2

        alpha = self.params.alpha
        beta = self.params.beta

        # Case 1 — Very poor accuracy
        if rho < rho0:
            new_eta = max(eta_min, eta / alpha)
            accept = False

        # Case 2 — Poor accuracy
        elif rho < rho1:
            new_eta = eta
            accept = True

        # Case 3 — Good accuracy
        elif rho < rho2:
            new_eta = min(eta_max, alpha * eta)
            accept = True

        # Case 4 — Excellent accuracy
        else:
            new_eta = min(eta_max, beta * eta)
            accept = True

        # Update TR
        # self.params.tr_radius = new_eta  # ❌ Frozen dataclass -> interdit
        self.tr_radius = new_eta  # ✅ mutable internal state

        return accept

    def _extract_seq_from_array(self) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Convertit les résultats optimisés en objets DgSampledSequence.
        PAS DE @staticmethod CAR ON UTILISE SELF.
        """
        tf = float(self.p_bar[0])
        K = self.params.K
        ts = np.linspace(0, tf, K)

        # Extraction commandes
        cmds_list = []
        for k in range(K):
            u_k = self.U_bar[:, k]
            # Assurez-vous que l'ordre est bon (F_right, F_left ou l'inverse selon SatelliteCommands)
            cmd = SatelliteCommands(u_k[0], u_k[1])
            cmds_list.append(cmd)
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # Extraction états
        states_list = []
        for k in range(K):
            x_k = self.X_bar[:, k]
            state = SatelliteState(x_k[0], x_k[1], x_k[2], x_k[3], x_k[4], x_k[5])
            states_list.append(state)
        mystates = DgSampledSequence[SatelliteState](timestamps=ts, values=states_list)

        return mycmds, mystates

    def _calculate_nonlinear_cost(self, X: NDArray, U: NDArray, p: NDArray) -> float:
        """Coût réel non-convexe (temps + fuel + chemin + pénalités)."""

        p_val = float(np.asarray(p).reshape(-1)[0])
        K = self.params.K

        # --- 1) Coûts "lisses" ---
        weight_time = self.params.w_t
        weight_fuel = self.params.w_u
        weight_path = 0.05  # tu peux tuner

        time_cost = weight_time * p_val
        fuel_cost = weight_fuel * np.sum(np.linalg.norm(U, ord=1, axis=0))

        path_length = 0.0
        for k in range(K - 1):
            dx = X[0, k + 1] - X[0, k]
            dy = X[1, k + 1] - X[1, k]
            path_length += np.sqrt(dx**2 + dy**2)
        path_cost = weight_path * path_length

        cost = time_cost + fuel_cost + path_cost

        # --- 2) Pénalités type SCvx (lambda_nu) ---
        lambda_nu = self.params.lambda_nu
        penalty = 0.0

        # 2.a) Dynamiques non linéaires vs linéarisées
        X_nl = self.integrator.integrate_nonlinear_piecewise(X, U, p)
        penalty += lambda_nu * np.sum(np.linalg.norm(X[:, 1:] - X_nl[:, 1:], ord=1, axis=0))

        # 2.b) État initial et final + commandes aux bornes
        init_arr = np.array(
            [
                self.init_state.x,
                self.init_state.y,
                self.init_state.psi,
                self.init_state.vx,
                self.init_state.vy,
                self.init_state.dpsi,
            ]
        )

        goal_arr = np.array(
            [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                self.goal_state.dpsi,
            ]
        )

        penalty += lambda_nu * np.linalg.norm(X[:, 0] - init_arr, 1)
        penalty += lambda_nu * np.linalg.norm(X[:, -1] - goal_arr, 1)
        penalty += lambda_nu * np.linalg.norm(U[:, 0], 1)
        penalty += lambda_nu * np.linalg.norm(U[:, -1], 1)

        # 2.c) Obstacles avec buffer cohérent avec _obstacle_base_params
        base_buf = float(np.max(np.linalg.norm(self._build_trapezoid_buffer(), axis=1)))

        # même philosophie que dans _obstacle_base_params,
        # mais sans reprendre exactement les margins (on reste un peu plus soft)
        r_buf_planet = 0.7 * base_buf
        r_buf_asteroid = 0.7 * base_buf

        # Planètes statiques
        for planet in self.planets.values():
            R_safe_p = self._safe_radius(planet)
            dist_sq = (X[0, :] - planet.center[0]) ** 2 + (X[1, :] - planet.center[1]) ** 2
            penalty += lambda_nu * np.sum(np.maximum(0.0, R_safe_p**2 - dist_sq))

        # Astéroïdes (centre qui bouge avec le temps)
        tau = np.linspace(0.0, 1.0, K)
        for asteroid in self.asteroids.values():
            cx0, cy0, vx, vy, R_safe_a = self._obstacle_base_params(asteroid)
            ax = cx0 + vx * tau * p_val
            ay = cy0 + vy * tau * p_val
            dist_sq = (X[0, :] - ax) ** 2 + (X[1, :] - ay) ** 2
            penalty += lambda_nu * np.sum(np.maximum(0.0, R_safe_a**2 - dist_sq))

        return float(cost + penalty)