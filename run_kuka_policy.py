import os

from baselines.common.models import mlp
from pybullet_envs.bullet import KukaGymEnv



if __name__ == '__main__':
  model_file = os.environ["HOME"]+"/data/tmp/kuka_model.pkl"
  env = KukaGymEnv(renders=True, isDiscrete=True)
  act = mlp.load(model_file)
  print(act)
  while True:
    obs, done = env.reset(), False
    print("===================================")
    print("obs")
    print(obs)
    episode_rew = 0
    while not done:
      env.render()
      obs, rew, done, _ = env.step(act(obs[None])[0])
      episode_rew += rew
    print("Episode reward", episode_rew)
