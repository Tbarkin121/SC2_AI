from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import numpy as np
import tensorflow as tf

from sc2_Qtables import QLearningTable
from sc2_Qnet import QLearner
import time

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectarmy'
MOVE_TO_M0 = 'm0'
MOVE_TO_M1 = 'm1'
MOVE_TO_M2 = 'm2'
MOVE_TO_M3 = 'm3'
MOVE_TO_M4 = 'm4'
MOVE_TO_M5 = 'm5'
MOVE_TO_M6 = 'm6'
MOVE_TO_M7 = 'm7'
MOVE_TO_M8 = 'm8'
MOVE_TO_M9 = 'm9'
MOVE_TO_M10 = 'm10'
MOVE_TO_M11 = 'm11'
MOVE_TO_M12 = 'm12'
MOVE_TO_M13 = 'm13'
MOVE_TO_M14 = 'm14'
MOVE_TO_M15 = 'm15'
MOVE_TO_M16 = 'm16'
MOVE_TO_M17 = 'm17'
MOVE_TO_M18 = 'm18'
MOVE_TO_M19 = 'm19'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    MOVE_TO_M0,
    MOVE_TO_M1,
    MOVE_TO_M2,
    MOVE_TO_M3,
    MOVE_TO_M4,
    MOVE_TO_M5,
    MOVE_TO_M6,
    MOVE_TO_M7,
    MOVE_TO_M8,
    MOVE_TO_M9,
    MOVE_TO_M10,
    MOVE_TO_M11,
    MOVE_TO_M12,
    MOVE_TO_M13,
    MOVE_TO_M14,
    MOVE_TO_M15,
    MOVE_TO_M16,
    MOVE_TO_M17,
    MOVE_TO_M18,
    MOVE_TO_M19,
]

class MarineAgent(base_agent.BaseAgent):
    def __init__(self):
        super(MarineAgent, self).__init__()
        print('Agent Init')
        self.first_step = True
        self.qlearn = QLearner(actions=list(range(len(smart_actions))), load_model=False)
        self.previous_state = None
        self.previous_action = tf.constant([0])
        self.are_units_selected = 0
        
    def reset(self):
        super(MarineAgent, self).reset()
        self.first_step = True
        self.are_units_selected = 0

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False

    def check_all_units(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type and unit.alliance == obs.observation.player.player_id]

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type and unit.alliance == obs.observation.player.player_id]
  
    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(MarineAgent, self).step(obs)
        if(self.first_step):
            self.first_step = False
            self.minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                  if unit.alliance == features.PlayerRelative.NEUTRAL]
            
            self.mineral_available = np.ones(len(self.minerals))
            
        # Checking which mineral patches remain 
        remaining_minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                  if unit.alliance == features.PlayerRelative.NEUTRAL]
        reward = -0.01
        for i in range(len(self.minerals)): 
            if( (self.minerals[i] not in remaining_minerals) and (self.mineral_available[i] == 1)):
                self.mineral_available[i] = 0
                # print('WE GOT MINERALS!!!')
                reward += 1
                
        self.marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == features.PlayerRelative.SELF]
        
        marine_unit = self.marines[0]
        marine_xy = [marine_unit.x, marine_unit.y]
            
        # function_id = np.random.choice(obs.observation.available_actions)
        # args = [[np.random.randint(0, size) for size in arg.sizes]
        #         for arg in self.action_spec[0].functions[function_id].args]
        # return actions.FunctionCall(function_id, args)

        current_state = [
            self.are_units_selected,
            marine_unit.x/84.,
            marine_unit.y/84.,
        ]
        for i in range(len(self.minerals)):
            current_state.append(self.minerals[i][0]/84.)
            current_state.append(self.minerals[i][1]/84.)
        for i in range(len(self.mineral_available)):
            current_state.append(self.mineral_available[i])
        # print('current state')
        # print(current_state)
        if self.previous_state is not None:
            self.qlearn.learn(self.previous_state, self.previous_action, reward, current_state)
        
        
        rl_action = self.qlearn.choose_action(current_state)
        smart_action = smart_actions[rl_action.numpy()[0]]
        
        self.previous_action = rl_action
        self.previous_state = current_state
        
        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_ARMY:
            self.are_units_selected = 1
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        elif smart_action == MOVE_TO_M0:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[0])
        
        elif smart_action == MOVE_TO_M1:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[1])
        
        elif smart_action == MOVE_TO_M2:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[2])
            
        elif smart_action == MOVE_TO_M3:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[3])
        
        elif smart_action == MOVE_TO_M4:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[4])
        
        elif smart_action == MOVE_TO_M5:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[5])
            
        elif smart_action == MOVE_TO_M6:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[6])
        
        elif smart_action == MOVE_TO_M7:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[7])
        
        elif smart_action == MOVE_TO_M8:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[8])
            
        elif smart_action == MOVE_TO_M9:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[9])
        
        elif smart_action == MOVE_TO_M10:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[10])
        
        elif smart_action == MOVE_TO_M11:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[11]) 
    
        elif smart_action == MOVE_TO_M12:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[12])
        
        elif smart_action == MOVE_TO_M13:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[13])
            
        elif smart_action == MOVE_TO_M14:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[14])
        
        elif smart_action == MOVE_TO_M15:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[15])
        
        elif smart_action == MOVE_TO_M16:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[16])
            
        elif smart_action == MOVE_TO_M17:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[17])
        
        elif smart_action == MOVE_TO_M18:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[18])
        
        elif smart_action == MOVE_TO_M19:
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.minerals[19]) 
        
        # self.previous_action = tf.constant([0])
        return actions.FUNCTIONS.no_op()
        
def main(unused_argv):
    agent = MarineAgent()
    try:
        # for _ in range(1):
        while True:
            with sc2_env.SC2Env(
                map_name="CollectMineralShards",
                players=[sc2_env.Agent(sc2_env.Race.zerg),
                        sc2_env.Bot(sc2_env.Race.random,
                                    sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True,
                    use_raw_units=True),
                step_mul=16,
                game_steps_per_episode=0,
                visualize=True) as env:
                
                agent.setup(env.observation_spec(), env.action_spec())
                
                timesteps = env.reset()
                agent.reset()
                
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
            
    except KeyboardInterrupt:
        pass
  
if __name__ == "__main__":
  app.run(main)

#%%


minerals = [[unit.x, unit.y] for unit in timesteps[0].observation.feature_units
    if unit.alliance == features.PlayerRelative.NEUTRAL]
            
mineral_available = np.ones(len(minerals))

marines = [unit for unit in timesteps[0].observation.feature_units
    if unit.alliance == features.PlayerRelative.SELF]

#%%
list_a = [1,2,3,4,5]
list_b = [1,3,4]
one_hots = np.ones(len(list_a))
for i in range(len(list_a)):
    if(list_a[i] not in list_b):
        one_hots[i] = 0
        
print(one_hots)