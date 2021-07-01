import gym
from gym import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import logging
import numpy as np

logger = logging.getLogger(__name__)

# class EnvTemplate(gym.Env):
#     metadata = {'render.modes': ['human']}
    
#     def __init__(self, **kwargs):
#         pass
    
#     def step(self, action):
#         pass
      
#     def reset(self):
#         pass

#     def close(self):
#         pass

#     def render(self, mode='human', close=False):
#         pass

class DZBEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    default_settings = {
        'map_name': "DefeatZerglingsAndBanelings",
        'players': [sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.hard)],
        'agent_interface_format': features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64),
        'realtime': False,
        'visualize': False,
        'step_mul': 1
        # 'game_steps_per_episode': 0

    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        self.marines = []
        self.banelings = []
        self.zerglings = []
        # 0 no operation
        # 1~32 move
        # 33~122 attack
        self.action_space = spaces.Discrete(123)
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(19,3)
        )

    def reset(self):
        if self.env is None:
            self.init_env()
        self.marines = []
        self.banelings = []
        self.zerglings = []
        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        obs = np.zeros((19,3), dtype=np.uint8)
        # 1 indicates my own unit, 4 indicates enemy's
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4)
        banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)
        self.marines = []
        self.banelings = []
        self.zerglings = []
        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i] = np.array([m.x, m.y, m[2]])
        for i, b in enumerate(banelings):
            self.banelings.append(b)
            obs[i+9] = np.array([b.x, b.y, b[2]])
        for i, z in enumerate(zerglings):
            self.zerglings.append(z)
            obs[i+13] = np.array([z.x, z.y, z[2]])
        return obs
    
    def get_units_by_type(self, obs, unit_type, player_relative=0):
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == player_relative]

    def step(self, action):
        raw_obs = self.take_action(action) # take safe action
        reward = raw_obs.reward # get reward from the env
        obs = self.get_derived_obs(raw_obs) # get derived observation
        # print('reward = {}'.format(reward))
        # print('obs = {}'.format(obs))
        return obs, reward, raw_obs.last(), {}  # return obs, reward and whether episode ends

    def take_action(self, action):
        # map value to action
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action<=32:
            derived_action = np.floor((action-1)/8)
            # get unit idx
            idx = (action-1)%8
            if derived_action == 0:
                action_mapped = self.move_up(idx)
            elif derived_action == 1:
                action_mapped = self.move_down(idx)
            elif derived_action == 2:
                action_mapped = self.move_left(idx)
            else:
                action_mapped = self.move_right(idx)
        else:
            # get attacker and enemy unit idx
            eidx = np.floor((action-33)/9)
            aidx = (action-33)%9
            action_mapped = self.attack(aidx, eidx)
        # execute action
        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def move_up(self, idx):
        idx = np.floor(idx)
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y-2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_down(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y+2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_left(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x-2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_right(self, idx):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x+2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def attack(self, aidx, eidx):
        try:
            selected = self.marines[aidx]
            if edix>3:
                # attack zerglines
                target = self.zerglines[eidx-4]
            else:
                target = self.banelings[eidx]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, targeted.tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()


    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass