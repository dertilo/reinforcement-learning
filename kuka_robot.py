# based on: examples/pybullet/gym/pybullet_envs/baselines/train_kuka_grasping.py
from baselines.common.models import mlp
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from baselines import deepq


def callback(lcl, glb):
  # stop training if reward exceeds 199
  total = sum(lcl['episode_rewards'][-101:-1]) / 100
  totalt = lcl['t']
  is_solved = totalt > 2000 and total >= 10
  return is_solved


def main():

  env = KukaGymEnv(renders=False, isDiscrete=True)
  model = mlp()
  act = deepq.learn(env,
                    network=model,
                    lr=1e-3,
                    max_timesteps=10000000,
                    buffer_size=50000,
                    exploration_fraction=0.1,
                    exploration_final_eps=0.02,
                    print_freq=10,
                    callback=callback)
  print("Saving model to kuka_model.pkl")
  act.save("kuka_model.pkl")


if __name__ == '__main__':
  main()
