import gym
from gym import spaces
import numpy as np
import math
import carb
import copy
from racing_data.center_points import center_points
from racing_data.left_boundary_points import left_boundary_points
from racing_data.right_boundary_points import right_boundary_points
import utils
from time import time
import csv

from reference_generator import ReferenceGenerator
from mpc_racing import MPCComponent


class MPCController():
    def __init__(self, prediction_horizon: int,max_distance: int, state_init: np.ndarray, vision_horizon:int = 10, rob_diameter: float = 0.17):

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
        target_state:np.ndarray,
        visible_center_points: np.ndarray
    ):
        u, predicted_states = self.mpc.step(start_state, target_state, visible_center_points)
        command = np.array(u[:,0].full())
        # print("command 1:" , command)
        command = np.array([*command[0],*command[1]])
        
        predicted_states_xyz = predicted_states
        predicted_states_xyz[2,:] = 0.05 # Replacing omega with z values
        # print("command 2:" , command)

        return command, predicted_states_xyz


class RacingEnv(gym.Env):
    def __init__(
            self,
            skip_frame=1,
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
            max_episode_length=2048,
            seed=0,
            headless=True,
            safety_filter=True
    ) -> None:
        from omni.isaac.kit import SimulationApp
        self.headless = headless
        self.simulation_app = SimulationApp({"headless" : self.headless, "anti_aliasing": 0})
        self.skip_frame = skip_frame
        self.dt = physics_dt * self.skip_frame
        self.max_episode_length = max_episode_length
        self.steps_after_reset = int(rendering_dt/physics_dt)
        self.safety_filter = safety_filter
        
        from omni.isaac.core import World
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.debug_draw import _debug_draw
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.get_euler_angle = quat_to_euler_angles
        

        self.world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return  
        


        #Init Jetbot
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self.pos_init_xyz = np.array([-1.64, -1.62, 0])
        self.jetbot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Jetbot",
                name="jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position = self.pos_init_xyz
            )
        )
        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.0325, wheel_base=0.1125)

        # set seed
        self.seed(seed)

        #Robot Constraints
        # self.max_velocity = 1
        # self.min_velocity = 0.3
        self.max_ray_length = 3
        self.rob_radius = 0.07
        self.max_distance_from_center = 0.375
                
        
        self.vision_horzion = 7
        center_points_2d = [sublist[:2] for sublist in center_points]
        self.rg = ReferenceGenerator(horizon = self.vision_horzion, center_points= center_points_2d)

        self.mpc_controller = MPCController(5, self.max_distance_from_center, self.get_jetbot_state(), self.vision_horzion, self.rob_radius*2)
        self.count = 0
        self.n_collisions = 0

        self.cat_states = []
        self.cat_forwards = []

        self.lap_time = 0
        self.lap_timestep =0


        self.finish_line = utils.get_boundary_for_point(center_points_2d[-2], center_points_2d[-1],None, self.max_distance_from_center*2)
        # print(self.finish_line)
        return
    
    def get_dt(self):
        return self._dt
    
    def step(self):
        self.draw_track()

        state_init = self.get_jetbot_state()
        visible_center_points = self.rg.generate_map((state_init[0], state_init[1]))
        x_ref, y_ref= visible_center_points[-1]
        state_target = np.array([x_ref, y_ref, 0])
        visible_center_points_1d = np.array(visible_center_points).flatten()
        self.draw.draw_points([(x_ref, y_ref, 0)],[(1,1,0,1)], [7])

        u, predicted_states_xyz = self.mpc_controller.forward(state_init, state_target, visible_center_points_1d)
        # print("forward v: ", u[0])

        # APPLYING ACTION
        for i in range(self.skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=u)
            )
            self.world.step(render=False)
        # END APPLYING ACTION

        done = False

        # current_jetbot_position, _ = self.jetbot.get_world_pose()
        state_current = self.get_jetbot_state()
        rays, _ = self.get_rays(state_current[0],state_current[1],state_current[2])
        curr_position = np.array([state_current[0], state_current[1]])

        self.world.render()
        self.draw.clear_lines()

        map_points = []
        for ray in rays:
            # print(ray)
            # ray_xyz = ([sublist+[0] for sublist in ray])
            # self.draw.draw_lines([ray_xyz[0]], [ray_xyz[1]], [(0, 0, 0, 1)],[1])
            map_points.append(ray[1])

        # CHECK FINISH LINE
        if utils.is_circle_intersecting_with_line(curr_position, self.rob_radius, self.finish_line[0], self.finish_line[1]):
            # if self.world.current_time_step_index - self.steps_after_reset>= 128:
            # print("Time for one lap: ", time()-self.start_time)
            print("Time for one lap: ", self.world.current_time_step_index - self.steps_after_reset)
            self.lap_time = time()-self.start_time
            self.lap_timestep = self.world.current_time_step_index - self.steps_after_reset
            done = True
                
        self.draw.clear_points()
        color = (0,1,0,1)
        for predicted_state in predicted_states_xyz.T:
            self.draw.draw_points([predicted_state], [color], [5])
        if utils.is_circle_intersecting_with_points(curr_position, self.rob_radius, map_points):
            self.n_collisions += 1
            # print("Collides! number of collisions so far: ", self.n_collisions)
            # done = True

        if(self.count %10 == 0):
            self.cat_states.append(curr_position)
            # velocities = self.jetbot.get_linear_velocity()
            # print(forward_velocity) 
            self.cat_forwards.append(u[0])

        self.count+=1
        return done
    
    def reset(self):
        self.world.reset()
        self.reset_counter = 0
        self.start_time = time()
        self.cat_states = []
        self.cat_forwards=[]
        x,y,th = self.get_jetbot_state()
        _, self.obs = self.get_rays(x, y ,th)

        self.lap_time = 0
        self.lap_timestep = 0

        self.world.render()
    
    def render(self, mode="human"):
        return
    
    def close(self):
        self.simulation_app.close()
        print(self.get_formatted_cat_states(self.cat_states))
        with open("test.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Lap Time",
                self.lap_time,
            ])
            writer.writerow([
                "Lap Timestep",
                self.lap_timestep
            ])
            writer.writerow([
                "Cat Positions",
                self.get_formatted_cat_states(self.cat_states)
            ])
            writer.writerow([
                "Cat Speed",
                self.get_formatted_cat(self.cat_forwards)
            ])

        return
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def get_jetbot_state(self)-> np.ndarray:
        jetbot_world_position, jetbot_world_orientation = self.jetbot.get_world_pose()
        jetbot_world_theta= self.get_euler_angle(jetbot_world_orientation)[-1]
        return np.array([jetbot_world_position[0], jetbot_world_position[1], jetbot_world_theta])

    def draw_track(self)-> None:
        
        self.draw.draw_lines_spline(center_points, (1, 1, 0, 1),1, False)
        self.draw.draw_lines_spline(left_boundary_points, (0, 0, 0, 1),8, False)
        self.draw.draw_lines_spline(right_boundary_points, (0, 0, 0, 1),8, False)

        left_finish_3d, right_finish_3d = [sublist+[0] for sublist in self.finish_line]
        self.draw.draw_lines([left_finish_3d], [right_finish_3d], [(0, 1, 0, 1)],[1])

    def get_rays(self, x:float, y:float ,th:float) -> np.ndarray:
        right_boundary_points_xy = [sublist[:2] for sublist in right_boundary_points]
        left_boundary_points_xy = [sublist[:2] for sublist in left_boundary_points]
        rays = []
        obs = []
        for i in range(-10,10):
            ray = utils.create_line((x,y), th+i*math.pi/15, self.max_ray_length)
            intersection = ray[1]           
            intersection_left = utils.is_intersecting_with_points(ray, left_boundary_points_xy)
            intersection_right = utils.is_intersecting_with_points(ray, right_boundary_points_xy)
            if intersection_left and intersection_right:
                if utils.get_distance(ray[0], intersection_left)< utils.get_distance(ray[0], intersection_right):
                    intersection = intersection_left
                else:
                    intersection = intersection_right
            elif intersection_left:
                intersection = intersection_left
            elif intersection_right:
                intersection = intersection_right

            ray = [ray[0], intersection]
            observation = (self.max_ray_length - utils.get_distance(ray[0], ray[1]))/self.max_ray_length
            rays.append(ray)
            obs.append(observation)
        return rays, obs
    
    def get_formatted_cat_states(self, cat_states):
        formatted = "["
        for state in cat_states:
            formatted+="["+ str(state[0])+ ", "+ str(state[1])+ "],"
        formatted+= "]"
        return formatted
    
    def get_formatted_cat(self, cat):
        formatted = "["
        for state in cat:
            formatted+=str(state)+ ","
        formatted+= "]"
        return formatted

env = RacingEnv(headless=False)
for i in range(2):
    env.reset()
    done = False
    while not done:
        done = env.step()
        env.render()

env.close()