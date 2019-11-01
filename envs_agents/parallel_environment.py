import torch
from multiprocessing import Process, Pipe
from typing import List, Callable, Dict

import gym

def worker(conn, env_suppliers:list):
    envs = [supplier() for supplier in env_suppliers]
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            conn.send([env.step(d) for env,d in zip(envs,data)])
        elif cmd == "reset":
            conn.send([env.reset() for env in envs])
        elif cmd == 'get_spaces':
            conn.send((envs[0].observation_space, envs[0].action_space))
        else:
            raise NotImplementedError

def receive_process_answers(pipes):
    LD = [x for p in pipes for x in p.recv()]
    DL = {k: [dic[k] for dic in LD] for k in LD[0]}
    return DL

class ParallelEnv(gym.Env):

    def __init__(self,env_suppliers:List[Callable[[],gym.Env]],num_processes=1):
        assert len(env_suppliers)%num_processes==0
        self.envs_per_process=len(env_suppliers)//num_processes
        self.locals = []
        self.num_processes = num_processes

        for k in range(num_processes):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, self.kth_slice(env_suppliers,k)))
            p.daemon = True
            p.start()
            remote.close()

        self.locals[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.locals[0].recv()

    def kth_slice(self,x,k):
        return x[k*self.envs_per_process:(k+1)*self.envs_per_process]

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        return receive_process_answers(self.locals)


    def step(self, actions):
        for k,local in enumerate(self.locals):
            local.send(("step", self.kth_slice(actions,k)))
        return receive_process_answers(self.locals)

    def render(self):
        raise NotImplementedError

    @staticmethod
    def build(build_env_supplier, num_envs,num_processes):
        return ParallelEnv([build_env_supplier(i) for i in range(num_envs)],num_processes)

class SingleEnvWrapper(gym.Env):
    def __init__(self,env:gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return {k:[v] for k,v in self.env.step(action).items()}

    def reset(self):
        return {k:[v] for k,v in self.env.reset().items()}

    def render(self, mode='human'):
        return self.env.render(mode)