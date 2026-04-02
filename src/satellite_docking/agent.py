from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters

from planner import SatellitePlanner
from goal import SpaceshipTarget, DockingTarget
from utils_params import PlanetParams, AsteroidParams
from utils_plot import plot_traj


class Config:
    PLOT = True
    VERBOSE = True  # Set to True to see initial computation logs
    VERBOSE = True  # Plus utilisé pour des prints mais je le laisse si tu en as besoin ailleurs


@dataclass(frozen=True)
class MyAgentParams:
    my_tol: float = 0.5  # cetait 0.3


class SatelliteAgent(Agent):
    """
    PDM4AR Agent - Single Pass Execution Mode.
    Computes trajectory once at init and executes blindly to visualize result.
    """

    init_state: SatelliteState
    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SatelliteCommands]
    state_traj: DgSampledSequence[SatelliteState]
    myname: PlayerName
    planner: SatellitePlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: MyAgentParams

    def __init__(
        self,
        init_state: SatelliteState,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
    ):
        self.actual_trajectory = []
        self.init_state = init_state
        self.planets = planets
        self.asteroids = asteroids
        self.params = MyAgentParams()

        # IMPORTANT : init des trajs et du temps de départ
        self.cmd_traj: DgSampledSequence[SatelliteCommands] | None = None
        self.state_traj: DgSampledSequence[SatelliteState] | None = None
        self.start_time: float = 0.0

    def _debug_asteroids_vs_model(self, sim_obs: SimObservations, local_t: float):
        if not self.asteroids or self.planner is None or self.planner.p_bar is None:
            return

        tf = float(self.planner.p_bar[0])
        if tf <= 0.0:
            return

        tau = float(np.clip(local_t / tf, 0.0, 1.0))

        sat_obs = sim_obs.players[self.myname]
        sat_state = cast(SatelliteState, sat_obs.state)
        sx, sy = sat_state.x, sat_state.y

        for name, ast_params in self.asteroids.items():
            if name not in sim_obs.players:
                continue

            ast_obs = sim_obs.players[name]
            ast_state = cast(DynObstacleState, ast_obs.state)
            ax_true, ay_true = ast_state.x, ast_state.y

            # Modèle (cohérent avec le planner)
            cx0, cy0, vx, vy, _ = self.planner._obstacle_base_params(ast_params)
            ax_model = cx0 + vx * tau * tf
            ay_model = cy0 + vy * tau * tf

            dist_true = float(np.hypot(ax_true - sx, ay_true - sy))
            dist_model = float(np.hypot(ax_model - sx, ay_model - sy))

            R_safe = self.planner._safe_radius(ast_params)

            collision_model = dist_model <= R_safe

            err = float(np.hypot(ax_true - ax_model, ay_true - ay_model))

            # DEBUG SLACK vs COLLISION_MODEL (calculs conservés, mais sans prints)
            s_val = None
            slack_ok = None
            eps = 1e-3  # tolérance num pour dire "≈0"

            dbg_dict = getattr(self.planner, "obs_slack_debug", None)
            if dbg_dict is not None and name in dbg_dict:
                dbg = dbg_dict[name]
                t_grid = dbg.t_grid
                # On prend l'indice le plus proche de local_t
                # (en supposant t_grid monotone, de 0 à tf)
                k = int(np.clip(np.searchsorted(t_grid, local_t), 0, len(t_grid) - 1))
                s_val = float(dbg.slack[k])

                # Test logique : si collision_model, slack devrait être > 0 (à eps près)
                if collision_model:
                    slack_ok = s_val > eps
                else:
                    slack_ok = abs(s_val) <= eps

            # Les "warnings" sont supprimés (pas de print)

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self.myname = init_sim_obs.my_name

        self.sg = cast(SatelliteGeometry, init_sim_obs.model_geometry)
        self.sp = cast(SatelliteParameters, init_sim_obs.model_params)

        assert isinstance(init_sim_obs.goal, (SpaceshipTarget, DockingTarget))
        self.goal_state = init_sim_obs.goal.target

        # ----- ICI : on récupère dock_offset / arms_length si c'est un DockingTarget -----
        dock_offset = 0.0
        dock_arms_length = 0.0
        if isinstance(init_sim_obs.goal, DockingTarget):
            dock_offset = init_sim_obs.goal.offset
            dock_arms_length = init_sim_obs.goal.arms_length

        # On crée le planner
        self.planner = SatellitePlanner(
            planets=self.planets,
            asteroids=self.asteroids,
            sg=self.sg,
            sp=self.sp,
        )

        # --- Docking line : on passe A1, A2 + init_state au planner ---
        if isinstance(init_sim_obs.goal, DockingTarget):
            A, B, C, A1, A2, half_p_angle = init_sim_obs.goal.get_landing_constraint_points()
            if Config.PLOT:
                init_sim_obs.goal.plot_landing_points(A, B, C, A1, A2)

            # NEW : on informe le planner de la Docking line
            self.planner.set_docking_line(A1, A2, self.init_state)

        # Calcul de la trajectoire
        self.cmd_traj, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)

    def get_commands(self, sim_obs: SimObservations) -> SatelliteCommands:
        """
        Execute the pre-calculated plan blindly. No replanning.
        """
        # État courant du satellite
        my_player = sim_obs.players[self.myname]
        current_state = cast(SatelliteState, my_player.state)

        # On stocke la trajectoire réelle pour debug / plotting
        self.actual_trajectory.append(current_state)

        # Temps global de la simu en float (sim_obs.time est un Decimal)
        current_time = float(sim_obs.time)

        # Temps local du plan courant (t = 0 au début du plan)
        local_t = current_time - self.start_time
        if local_t < 0.0:
            local_t = 0.0
        self._debug_asteroids_vs_model(sim_obs, local_t)

        # État attendu d'après la trajectoire planifiée (au même temps local !)
        expected_state = None
        if self.state_traj is not None:
            expected_state = self.state_traj.at_interp(local_t)

        # Optionnel : plot (on garde le temps global pour l’horloge du plot)
        if Config.PLOT and int(10 * current_time) % 25 == 0:
            plot_traj(self.state_traj, self.actual_trajectory)

        # ---------- Schéma de replanning ----------
        if expected_state is not None:
            dist_err = np.sqrt((current_state.x - expected_state.x) ** 2 + (current_state.y - expected_state.y) ** 2)
        else:
            dist_err = 999.0  # si pas de plan, force un replanning (mais voir condition ci-dessous)

        #  Replanning UNIQUEMENT s'il n'y a PAS d'astéroïdes
        # ⚠️ Replanning UNIQUEMENT s'il n'y a PAS d'astéroïdes
        if dist_err > self.params.my_tol and len(self.asteroids) == 0:
            if Config.VERBOSE:
                print(f"[Agent] Deviation {dist_err:.3f} > {self.params.my_tol}. Replanning...")

            try:
                # on recalcule une trajectoire à partir de l'état courant
                cmds, states = self.planner.compute_trajectory(current_state, self.goal_state)

                self.cmd_traj = cmds
                self.state_traj = states
                # IMPORTANT : on stocke le temps de départ du plan en float
                self.start_time = current_time
            except Exception as e:
                if Config.VERBOSE:
                    print("[Agent] Replanning failed:", e)
            except Exception:
                # Échec de replanning silencieux
                pass

        # ---------- Suivi de trajectoire ----------
        if self.cmd_traj is None:
            # sécurité : pas de plan -> zéro commande
            return SatelliteCommands(0.0, 0.0)

        # Recalcule du temps local (au cas où on vient de replannifier juste au-dessus)
        local_t = current_time - self.start_time
        if local_t < 0.0:
            local_t = 0.0

        # On prend la commande prévue au temps local courant
        planned_cmd = self.cmd_traj.at_interp(local_t)
        return planned_cmd
