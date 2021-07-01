# from gym.envs.registration import register

# register(
#     id='customizd-env1-v0',
#     entry_point='customized_env.envs:CustomizedEnv1',
# )
# register(
#     id='customzied-env2-v0',
#     entry_point='customized_env.envs:CustomizedEnv2',
# )
 
from gym.envs.registration import register

register(
    id='defeat-zerglings-banelings-v0',
    entry_point='sc2_gym.envs:DZBEnv',
)



