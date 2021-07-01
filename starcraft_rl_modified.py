from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import numpy as np
import tensorflow as tf

from sc2_Qtables import QLearningTable
from sc2_Qnet import QLearner
from q_learner_config import smart_actions
from q_learner_config import KILL_UNIT_REWARD, KILL_BUILDING_REWARD, MAKE_ZERGLINGS
from q_learner_config import ACTION_DO_NOTHING, ACTION_SELECT_DRONE, ACTION_BUILD_OVERLORD, ACTION_BUILD_SPAWNING_POOL, ACTION_BUILD_SPINE
from q_learner_config import ACTION_SELECT_LARVA, ACTION_BUILD_ZERGLING, ACTION_SELECT_ARMY, ACTION_ATTACK, ACTION_BUILD_DRONE
from q_learner_config import ACTION_SELECT_HATCHERY
import time

class ZergAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ZergAgent, self).__init__()
        print('Agent Init')
        self.attack_coordinates = None
        # self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.qlearn = QLearner(actions=list(range(len(smart_actions))))
        # Setting up a dense reward system
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_zergling_count = 0
        self.previous_action = tf.constant([0])
        # self.previous_action = None
        self.previous_state = None
        self.is_hatch_selected = False
        self.is_larva_selected = False
        self.previous_worker_supply = 0
        self.previous_overlord_supply = 0

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
        super(ZergAgent, self).step(obs)
        # Determining Start Position and Enemy Start Position (Naive)
        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()
        
            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)

        overlord_count = len(self.check_all_units(obs, units.Zerg.Overlord))
        larva_count = len(self.check_all_units(obs, units.Zerg.Larva))
        zergling_count = len(self.check_all_units(obs, units.Zerg.Zergling))
        spawningpool_count = len(self.check_all_units(obs, units.Zerg.SpawningPool))
        spinecrawler_count = len(self.check_all_units(obs, units.Zerg.SpineCrawler))
        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_army
        worker_supply = obs.observation.player.food_workers

        killed_unit_score = obs.observation.score_cumulative.killed_value_units
        killed_building_score = obs.observation.score_cumulative.killed_value_structures 
        mineral_collect_score = obs.observation.score_cumulative.collection_rate_minerals 

        current_state = [
            supply_limit/200,
            # army_supply/200,
            worker_supply/200,
            # spawningpool_count/200,
            # spinecrawler_count/200,
            # obs.observation.player.minerals/100,
            # self.previous_action.numpy()[0]/10, 
            self.is_hatch_selected,
            self.is_larva_selected,      
        ]
    
        if self.previous_state is not None:
            # reward = mineral_collect_score/1000
            # reward = worker_supply/200
            # reward -= (supply_limit - worker_supply)/200
            reward = 0
            if worker_supply > self.previous_worker_supply:
                reward += 1
            # if killed_unit_score > self.previous_killed_unit_score:
            #     reward += KILL_UNIT_REWARD
            #     # pass
            # if killed_building_score > self.previous_killed_building_score:
            #     reward += KILL_BUILDING_REWARD
            #     # pass
            # if(zergling_count > self.previous_zergling_count):
            #     print('Zergling Count = {}'.format(zergling_count))
            #     reward += MAKE_ZERGLINGS

            self.qlearn.learn(self.previous_state, self.previous_action, reward, current_state)
    
        # print('!!!!!!!!!!!!!!!!')
        # print(current_state)
        rl_action = self.qlearn.choose_action(current_state)
        # print('rl_action = {}'.format(rl_action))
        # print('!!!!!!!!!!!!!!!!')
        smart_action = smart_actions[rl_action.numpy()[0]]
        # print(smart_action)
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_mineral_collect_score = mineral_collect_score
        self.previous_zergling_count = zergling_count
        self.previous_state = current_state
        self.previous_action = rl_action
        self.previous_worker_supply = worker_supply
        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_HATCHERY:
            self.is_hatch_selected = True
            self.is_larva_selected = False
            hatch = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatch) > 0:
                hatch = random.choice(hatch)
                return actions.FUNCTIONS.select_point("select_all_type", (hatch.x,
                                                                        hatch.y))
        elif smart_action == ACTION_SELECT_LARVA:
            self.is_hatch_selected = False
            self.is_larva_selected = True
            if self.can_do(obs, actions.FUNCTIONS.select_larva.id):
                return actions.FUNCTIONS.select_larva("now")
            # larva = self.get_units_by_type(obs, units.Zerg.Larva)
            # if len(larva) > 0:
            #     larva = random.choice(larva)
            #     return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                        # larva.y))
        elif smart_action == ACTION_BUILD_DRONE:
            self.is_hatch_selected = False
            self.is_larva_selected = False
            if self.unit_type_is_selected(obs, units.Zerg.Larva):
                if self.can_do(obs, actions.FUNCTIONS.Train_Drone_quick.id):
                    return actions.FUNCTIONS.Train_Drone_quick("now")

        elif smart_action == ACTION_BUILD_OVERLORD:
            self.is_hatch_selected = False
            self.is_larva_selected = False
            if self.unit_type_is_selected(obs, units.Zerg.Larva):
                if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")

        elif smart_action == ACTION_SELECT_DRONE:
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            if len(drones) > 0:
                drone = random.choice(drones)
                return actions.FUNCTIONS.select_point("select_all_type", (np.clip(drone.x,0,83),
                                                                    np.clip(drone.y,0,83)))

        elif smart_action == ACTION_BUILD_SPAWNING_POOL:
            if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                x = random.randint(0, 83)
                y = random.randint(0, 83)
                return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))

        elif smart_action == ACTION_BUILD_SPINE:
            if self.can_do(obs, actions.FUNCTIONS.Build_SpineCrawler_screen.id):
                x = random.randint(0, 83)
                y = random.randint(0, 83)
                return actions.FUNCTIONS.Build_SpineCrawler_screen("now", (x, y))
        
        elif smart_action == ACTION_BUILD_ZERGLING:
            if self.unit_type_is_selected(obs, units.Zerg.Larva):
                if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                    return actions.FUNCTIONS.Train_Zergling_quick("now")

        elif smart_action == ACTION_SELECT_ARMY:
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        elif smart_action == ACTION_ATTACK:
            # if self.unit_type_is_selected(obs, units.Zerg.Zergling):!
            if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                return actions.FUNCTIONS.Attack_minimap("now",self.attack_coordinates)

        # IF Action we chose failed
        self.previous_action = tf.constant([0])
        return actions.FUNCTIONS.no_op()


def main(unused_argv):
    agent = ZergAgent()
    try:
        # for _ in range(1):
        while True:
            with sc2_env.SC2Env(
                map_name="Simple64",
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
                
            # with sc2_env.SC2Env(
            #     map_name="Simple64",
            #     players=[sc2_env.Agent(sc2_env.Race.zerg),
            #             sc2_env.Agent(sc2_env.Race.zerg)],
            #     agent_interface_format=features.AgentInterfaceFormat(
            #         feature_dimensions=features.Dimensions(screen=84, minimap=64),
            #         use_feature_units=True,
            #         use_raw_units=True),
            #     step_mul=16,
            #     game_steps_per_episode=0,
            #     visualize=True) as env:
                
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
# print(dir(timesteps[0].observation.feature_units))
# print(timesteps[0].observation.feature_units)
# print(len(timesteps[0].observation.feature_units))


for unit in timesteps[0].observation.feature_units:
    print(unit.unit_type)
    # if(unit.alliance == timesteps[0].observation.player.player_id):
    #     print('{}, {}'.format(unit.unit_type, unit.alliance))
    #     if(unit.unit_type == units.Zerg.Drone):
    #         print('Found a drone')
        
print()
print(units.Zerg.SpawningPool)

#%%
for unit in timesteps[0].observation.raw_units:
    print(unit.unit_type)
    # if(unit.unit_type == units.Zerg.Drone and unit.alliance == timesteps[0].observation.player.player_id):
        # print('{}, {}'.format(unit.unit_type, unit.alliance))
        # print('Found a drone')

print()
print(units.Zerg.SpawningPool)
#%%
print(timesteps[0].observation.raw_units)
print()
print(timesteps[0].observation.feature_units)

#%%
import tensorflow as tf
from sc2_Qnet import QNet
test = QNet(10, 32)
test.compile(optimizer='adam')
#%%
test_input = tf.Variable([0,1,2,3,4])
test_input = tf.expand_dims(test_input, 0)
print(test_input)
logits = test(test_input)
print(logits[0])
print(tf.math.reduce_max(logits).numpy())