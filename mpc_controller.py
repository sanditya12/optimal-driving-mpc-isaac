
from omni.isaac.core.controllers import BaseController
from racing_data.center_points import center_points
from racing_data.left_boundary_points import left_boundary_points
from racing_data.right_boundary_points import right_boundary_points
from mpc_racing import MPCComponent
from casadi import sin, cos, pi
import random
import numpy as np

class MPCController(BaseController):
    def __init__(self, prediction_horizon: int,max_distance: int, state_init: np.ndarray, vision_horizon:int = 10, rob_diameter: float = 0.17):
        super().__init__(name="mpc_controller"),
        

        self.mpc = MPCComponent(prediction_horizon, vision_horizon=vision_horizon, rob_diameter= rob_diameter)
        self.mpc.init_symbolic_vars()
        self.mpc.init_cost_fn_and_g_constraints()
        self.mpc.add_track_constraints(max_distance)
        self.mpc.init_solver()
        self.mpc.init_constraint_args()
        self.mpc.add_track_args()
        self.mpc.prepare_step(state_init)
        
        return


    def forward(
        self,
        start_state:np.ndarray,
        u_desired:np.ndarray,
        visible_center_points: np.ndarray
    ):
        u, predicted_states = self.mpc.step(start_state, u_desired, visible_center_points)
        command = np.array(u[:,0].full())
        # print("command 1:" , command)
        command = np.array([*command[0],*command[1]])

        predicted_states_xyz = predicted_states
        predicted_states_xyz[2,:] = 0.05 # Replacing omega with z values
        # print("command 2:" , command)

        return command, predicted_states_xyz