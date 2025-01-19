import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces

from constant import DEFAULT_SC2_OBSERVATION_SPACE_SHAPE, DEFAULT_SC2_ACTION_SPACE, DEFAULT_SC2_MAX_STEPS, DEFAULT_SC2_OBSERVATION_SPACE_SHAPE
from pysc2.lib import actions, features, units
from utils.base_terran_agent import BaseTerranAgent

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
        self.agent = BaseTerranAgent()
        self.observation_space = self.get_observation_space()
        self.action_space = gym.spaces.Discrete(DEFAULT_SC2_ACTION_SPACE)
        self.max_steps = DEFAULT_SC2_MAX_STEPS
        self.agent_view_size = 0

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        self.agent.step(self.timesteps[0])
        real_action = self.translate_action(action)
        print("real_action: ", real_action)
        self.timesteps = self.env.step([real_action])
        state = self.get_state(self.timesteps[0])
        reward = np.array([self.timesteps[0].reward])
        terminated = self.timesteps[0].last()
        if terminated:
            # 勝敗を出力
            print("reward: ", reward)
        info = {}
        return state, reward, terminated, info

    def render(self):
        self.env.render()

    def reset(self, seed=None):
        self.timesteps = self.env.reset()
        return self.get_state(self.timesteps[0])
    
    def get_observation_space(self):
        space_shape = DEFAULT_SC2_OBSERVATION_SPACE_SHAPE
        return gym.spaces.Box(low=0, high=255, shape=(space_shape, 1, 1), dtype=np.uint8)
    
    def get_state(self, obs):
        scvs = self.agent.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.agent.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.agent.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.agent.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.agent.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.agent.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.agent.get_my_units_by_type(obs, units.Terran.Marine)
        
        queued_marines = (completed_barrackses[0].order_length 
                        if len(completed_barrackses) > 0 else 0)
        
        free_supply = (obs.observation.player.food_cap - 
                    obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100
        
        enemy_scvs = self.agent.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.agent.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_completed_command_centers = self.agent.get_enemy_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.agent.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.agent.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.agent.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.agent.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.agent.get_enemy_units_by_type(obs, units.Terran.Marine)
        
        state = np.array(
                [
                len(command_centers),
                # len(scvs),
                # len(idle_scvs),
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
                # len(enemy_command_centers),
                # len(enemy_scvs),
                # len(enemy_idle_scvs),
                # len(enemy_supply_depots),
                # len(enemy_completed_supply_depots),
                # len(enemy_barrackses),
                # len(enemy_completed_barrackses),
                # len(enemy_marines)
                ]
        )
        
        return state.reshape(1, 1, DEFAULT_SC2_OBSERVATION_SPACE_SHAPE, 1)
        
    def translate_action(self, action):
        if action == 0:
            return self.agent.do_nothing(self.timesteps[0])
        elif action == 1:
            return self.agent.harvest_minerals(self.timesteps[0])
        elif action == 2:
            return self.agent.build_supply_depot(self.timesteps[0])
        elif action == 3:
            return self.agent.build_barracks(self.timesteps[0])
        elif action == 4:
            return self.agent.train_marine(self.timesteps[0])
        elif action == 5:
            return self.agent.attack(self.timesteps[0])


class WrapSC2_2Env(WrapSC2Env):
    def __init__(self, env_fn):
        super().__init__(env_fn)
        self.action_space = gym.spaces.Discrete(DEFAULT_SC2_ACTION_SPACE + 2)
        self.observation_space = self.get_observation_space()
    
    def get_observation_space(self):
        space_shape = DEFAULT_SC2_OBSERVATION_SPACE_SHAPE + 2
        return gym.spaces.Box(low=0, high=255, shape=(space_shape, 1, 1), dtype=np.uint8)
    
    def reset(self, seed=None):
        self.timesteps = self.env.reset()
        self.agent.main_base_facility_status, self.agent.sub_base_facility_status = 1, 1
        return self.get_state(self.timesteps[0])
    
    def get_state(self, obs):
        command_centers = self.agent.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.agent.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.agent.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.agent.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.agent.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.agent.get_my_units_by_type(obs, units.Terran.Marine)
        
        queued_marines = (completed_barrackses[0].order_length 
                        if len(completed_barrackses) > 0 else 0)
        
        free_supply = (obs.observation.player.food_cap - 
                    obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100
                
        state = np.array(
                [
                len(command_centers),
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
                self.agent.main_base_facility_status,
                self.agent.sub_base_facility_status
                ]
        )
        return state.reshape(1, 1, DEFAULT_SC2_OBSERVATION_SPACE_SHAPE + 2, 1)
    
    def translate_action(self, action):
        if action == 0:
            return self.agent.do_nothing(self.timesteps[0])
        elif action == 1:
            return self.agent.harvest_minerals(self.timesteps[0])
        elif action == 2:
            return self.agent.build_supply_depot(self.timesteps[0])
        elif action == 3:
            return self.agent.build_barracks(self.timesteps[0])
        elif action == 4:
            return self.agent.train_marine(self.timesteps[0])
        elif action == 5:
            return self.agent.attack_main_base(self.timesteps[0])
        elif action == 6:
            return self.agent.attack_sub_base(self.timesteps[0])
        elif action == 7:
            return self.agent.attack_remaining_hidden_structures(self.timesteps[0])

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