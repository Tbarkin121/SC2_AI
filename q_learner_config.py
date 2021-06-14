from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]

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
    
    ACTION_SELECT_DRONE,
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