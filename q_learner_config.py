from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features


ACTION_DO_NOTHING = 'donothing'

ACTION_SELECT_DRONE = 'selectdrone'
ACTION_BUILD_SPAWNING_POOL = 'buildspawningpool'
ACTION_BUILD_SPINE = 'buildspine'

ACTION_SELECT_HATCHERY = 'selecthatchEry'
ACTION_SELECT_LARVA = 'selectlarva'
ACTION_BUILD_ZERGLING = 'buildzergling'
ACTION_BUILD_OVERLORD = 'buildoverlord'
ACTION_BUILD_DRONE = 'builddrone'

ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_HATCHERY,
    ACTION_SELECT_LARVA,
    
    ACTION_BUILD_DRONE,
    ACTION_BUILD_OVERLORD,
    # ACTION_BUILD_ZERGLING,
    
    # ACTION_SELECT_DRONE,
    # ACTION_BUILD_SPAWNING_POOL,
    # ACTION_BUILD_SPINE,
    
    # ACTION_SELECT_ARMY,
    # ACTION_ATTACK,
]

# for mm_x in range(0, 64):
    # for mm_y in range(0, 64):
        # smart_actions.append(ACTION_ATTACK + ‘_’ + str(mm_x) + ‘_’ + str(mm_y))
        
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5
COLLECT_MINERALS = 0.1
MAKE_ZERGLINGS = 0.1