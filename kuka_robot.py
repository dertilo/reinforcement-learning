# based on: examples/pybullet/gym/pybullet_envs/baselines/train_kuka_grasping.py
from baselines.common.models import mlp
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from baselines import deepq
import pybullet as pb


class KukaRobotEnv(KukaGymEnv):

    distance = 0

    def _reward(self):
        # rewards is height of target object
        blockPos, blockOrn = pb.getBasePositionAndOrientation(self.blockUid)
        closestPoints = pb.getClosestPoints(
            self.blockUid, self._kuka.kukaUid, 1000, -1, self._kuka.kukaEndEffectorIndex
        )
        distance = closestPoints[0][8]
        if distance < self.distance:
            reward = -1.0
        else:
            reward = 1.0
        self.distance = distance
        return reward


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl["episode_rewards"][-101:-1]) / 100
    totalt = lcl["t"]
    is_solved = totalt > 200 and total >= 100
    if is_solved:
        print("is_solved")
    return is_solved


if __name__ == "__main__":

    env = KukaRobotEnv(renders=False, maxSteps=1000, isDiscrete=True)
    model = mlp()
    act = deepq.learn(
        env,
        network=model,
        lr=1e-3,
        max_timesteps=10000000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
    )
    model_file = "/docker-share/data/tmp/kuka_model.pkl"
    act.save(model_file)
