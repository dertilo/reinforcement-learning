import torch


def train_batch(agent, batch, optimizer):

    agent.train()
    loss = agent.loss(batch)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(),max_norm=0.5)
    optimizer.step()
    return batch.env_steps.done.numpy()