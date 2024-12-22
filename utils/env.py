import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces

from constant import DEFAULT_SC2_OBSERVATION_SPACE_SHAPE, DEFAULT_SC2_ACTION_SPACE, DEFAULT_SC2_MAX_STEPS
from pysc2.lib import actions, features, units

def make_env_fn(env_key, render_mode=None, frame_stack=1):
    def _f():
        env = gym.make(env_key, render_mode=render_mode)
        if frame_stack > 1:
            env = FrameStack(env, frame_stack)
        return env
    return _f


# Gives a vectorized interface to a single environment
class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        # assert action.ndim == 1
        env_return = self.env.step(action)
        if len(env_return) == 4:
            state, reward, terminated, info = env_return
        else:
            state, reward, terminated, truncated, info = env_return
        if isinstance(state, dict):
            state = state['image']
        return np.array([state]), np.array([reward]), np.array([terminated]), np.array([info])

    def render(self):
        self.env.render()

    def reset(self, seed=None):
        state, *_ = self.env.reset(seed=seed)
        if isinstance(state, tuple):
            ## gym state is tuple type
            return np.array([state[0]])
        elif isinstance(state, dict):
            ## minigrid state is dict type
            return np.array([state['image']])
        else:
            return np.array([state])
        
class WrapSC2Env:
    def __init__(self, env_fn):
        self.env = env_fn()
        self.observation_space = self.observation_space()
        self.action_space = gym.spaces.Discrete(DEFAULT_SC2_ACTION_SPACE)
        self.max_steps = DEFAULT_SC2_MAX_STEPS
        self.agent_view_size = 0

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        timesteps = self.env.step([action])
        state = self.get_state(timesteps[0])
        reward = timesteps[0].reward
        terminated = timesteps[0].last()
        info = {}
        return state, reward, terminated, info

    def render(self):
        self.env.render()

    def reset(self):
        timesteps = self.env.reset()
        return self.get_state(timesteps[0])
        
    def observation_space(self):
        space_shape = DEFAULT_SC2_OBSERVATION_SPACE_SHAPE
        return gym.spaces.Box(low=0, high=255, shape=(space_shape, 1, 1), dtype=np.uint8)
    
    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
            return True
        
        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True
        
        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]
    
    def can_do(self, obs, action):
        return action in obs.observation.available_actions
    
    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type 
                and unit.alliance == features.PlayerRelative.SELF]
    
    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        
        queued_marines = (completed_barrackses[0].order_length 
                        if len(completed_barrackses) > 0 else 0)
        
        free_supply = (obs.observation.player.food_cap - 
                    obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100
        
        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
        
        return [len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_supply_depots),
                len(enemy_completed_supply_depots),
                len(enemy_barrackses),
                len(enemy_completed_barrackses),
                len(enemy_marines)]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space['image'].shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space['image'].dtype)

    def reset(self, seed=None):
        ob = self.env.reset(seed=seed)[0]
        ob = ob['image']
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), {}

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        # ob, reward, done, info = self.env.step(action)
        ob = ob['image']
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=-1)