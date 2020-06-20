import os
import torch

from baselines import deepq
from baselines.common.models import mlp
from baselines.common.tf_util import load_variables
from baselines.deepq import load_act
from pybullet_envs.bullet import KukaGymEnv
from tensorflow.python.saved_model.load import load

from kuka_robot import KukaRobotEnv


def enjoy_kuka(model_file):
  agent = torch.load(model_file)
  agent.eval()

  env = KukaRobotEnv(renders=True, maxSteps=1000, isDiscrete=True)
  from gym.wrappers import Monitor
  env = Monitor(env, "./vid",
                video_callable=lambda episode_id: True, force=True)
  while True:
    state, ep_reward = env.reset(), 0
    for t in range(1, 10000):  # Don't infinite loop while learning
      action, _ = agent.step(state)
      state, reward, done, _ = env.step(action)
      if done:
        break


if __name__ == '__main__':
  model_file = os.environ["HOME"]+"/gunther/data/kuka_models/model.pt"
  enjoy_kuka(model_file)