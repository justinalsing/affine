import torch
from tqdm import tqdm

def sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2):

    # Progress-bar
    pbar = tqdm(total=n_steps, desc="Sampling")  # Jupyter notebook or qtconsole

    # Initialize current state
    current_state1 = torch.as_tensor(walkers1)
    current_state2 = torch.as_tensor(walkers2)

    # Initial target log prob for the walkers (and set any nans to -inf)...
    logp_current1 = log_prob(current_state1)
    logp_current2 = log_prob(current_state2)

    logp_current1 = torch.as_tensor(logp_current1)
    logp_current2 = torch.as_tensor(logp_current2)

    logp_current1 = torch.where(
        torch.isnan(logp_current1),
        torch.ones_like(logp_current1).fill_(float("inf")),
        logp_current1)
    logp_current2 = torch.where(
        torch.isnan(logp_current2),
        torch.ones_like(logp_current2).fill_(float("inf")),
        logp_current2)

    # Holder for the whole chain
    chain = [torch.cat([current_state1, current_state2], axis=0)]


    # MCMC loop
    for epoch in range(1, n_steps):

        # FIRST SET OF WALKERS:
        # Proposals
        #idx1 = torch.as_tensor(np.random.randint(0, n_walkers, n_walkers))
        idx1 = torch.randint(low=0, high=n_walkers, size=(n_walkers,))
        partners1 = current_state2[idx1]
        z1 = 0.5 * (torch.rand((n_walkers,)) + 1) ** 2
        proposed_state1 = partners1 + (z1 * (current_state1 - partners1).T).T

        # Target log prob at proposed points
        logp_proposed1 = log_prob(proposed_state1)
        logp_proposed1 = torch.as_tensor(logp_proposed1)
        logp_proposed1 = torch.where(
            torch.isnan(logp_proposed1),
            torch.ones_like(logp_proposed1).fill_(float("inf")),
            logp_proposed1)

        # Acceptance probability
        p_accept1 = torch.minimum(
            torch.ones(n_walkers),
            z1 ** (n_params - 1) * torch.exp(logp_proposed1 - logp_current1))

        # Accept or not
        accept1_ = torch.rand((n_walkers,)) <= p_accept1
        accept1 = accept1_.type(torch.float32)

        # Update the state
        current_state1 = (
            (current_state1).T * (1 - accept1) + (proposed_state1).T * accept1).T
        logp_current1 = torch.where(accept1_, logp_proposed1, logp_current1)

        # SECOND SET OF WALKERS:
        # Proposals
        #idx2 = torch.as_tensor(np.random.randint(0, n_walkers, n_walkers))
        idx2 = torch.randint(low=0, high=n_walkers, size=(n_walkers,))
        partners2 = current_state1[idx2]
        z2 = 0.5 * (torch.rand((n_walkers,)) + 1) ** 2
        proposed_state2 = partners2 + (z2 * (current_state2 - partners2).T).T

        # Target log prob at proposed points
        logp_proposed2 = log_prob(proposed_state2)
        logp_proposed2 = torch.as_tensor(logp_proposed2)
        logp_proposed2 = torch.where(
            torch.isnan(logp_proposed2),
            torch.ones_like(logp_proposed2).fill_(float("inf")),
            logp_proposed2)

        # Acceptance probability
        p_accept2 = torch.minimum(
            torch.ones(n_walkers),
            z2 ** (n_params - 1) * torch.exp(logp_proposed2 - logp_current2))

        # Accept or not
        accept2_ = torch.rand((n_walkers,)) <= p_accept2
        accept2 = accept2_.type(torch.float32)

        # Update the state
        current_state2 = (
            (current_state2).T * (1 - accept2) + (proposed_state2).T * accept2).T
        logp_current2 = torch.where(accept2_, logp_proposed2, logp_current2)

        # Append to chain
        chain.append(torch.cat([current_state1, current_state2], axis=0))

        # Update the progressbar
        pbar.update(1)

    # Stack up the chain
    chain = torch.stack(chain, axis=0)

    # Chain = np.unique(chain, axis=0) # this may need to be here,
    return chain[1:, :, :]

# state variables have shape: (n_walkers, n_batch, n_params)
def sample_batch(log_prob, n_steps, current_state, args=[], progressbar=True):
    # Split the current state
    current_state1, current_state2 = current_state

    # Pull out the number of parameters, walkers, and batch size
    n_walkers, n_batch, n_params = current_state1.shape

    # Initial target log prob for the walkers (and set any nans to -inf)...
    logp_current1 = log_prob(current_state1, *args)
    logp_current2 = log_prob(current_state2, *args)
    logp_current1[torch.isnan(logp_current1)] = -float('inf')
    logp_current2[torch.isnan(logp_current2)] = -float('inf')

    # Holder for the whole chain
    chain = [torch.unsqueeze(torch.cat([current_state1, current_state2], dim=0), dim=0)]

    # Progress bar?
    loop = tqdm if progressbar else range

    # MCMC loop
    for epoch in loop(1, n_steps):
        # First set of walkers:
        # Proposals
        partners1 = current_state2[np.random.randint(0, n_walkers, n_walkers)]
        z1 = 0.5 * (torch.rand(n_walkers, n_batch) + 1) ** 2
        proposed_state1 = partners1 + (z1 * (current_state1 - partners1).permute(2, 0, 1)).permute(1, 2, 0)

        # Target log prob at proposed points
        logp_proposed1 = log_prob(proposed_state1, *args)
        logp_proposed1[torch.isnan(logp_proposed1)] = -float('inf')

        # Acceptance probability
        p_accept1 = torch.minimum(torch.ones([n_walkers, n_batch]), z1**(n_params-1) * torch.exp(logp_proposed1 - logp_current1))

        # Accept or not
        accept1_ = (torch.rand([n_walkers, n_batch]) <= p_accept1)
        accept1 = accept1_.type(torch.float32)

        # Update the state
        current_state1 = (current_state1.permute(2, 0, 1) * (1 - accept1) + proposed_state1.permute(2, 0, 1) * accept1).permute(1, 2, 0)
        logp_current1[accept1_.bool()] = logp_proposed1[accept1_.bool()]

        # Second set of walkers:
        # Proposals
        partners2 = current_state1[np.random.randint(0, n_walkers, n_walkers)]
        z2 = 0.5 * (torch.rand(n_walkers, n_batch) + 1) ** 2
        proposed_state2 = partners2 + (z2 * (current_state2 - partners2).permute(2, 0, 1)).permute(1, 2, 0)

        # Target log prob at proposed points
        logp_proposed2 = log_prob(proposed_state2, *args)
        logp_proposed2[torch.isnan(logp_proposed2)] = -float('inf')

        # Acceptance probability
        p_accept2 = torch.minimum(torch.ones([n_walkers, n_batch]), z2**(n_params-1) * torch.exp(logp_proposed2 - logp_current2))

        # Accept or not
        accept2_ = (torch.rand([n_walkers, n_batch]) <= p_accept2)
        accept2 = accept2_.type(torch.float32)

        # Update the state
        current_state2 = (current_state2.permute(2, 0, 1) * (1 - accept2) + proposed_state2.permute(2, 0, 1) * accept2).permute(1, 2, 0)
        logp_current2[accept2_.bool()] = logp_proposed2[accept2_.bool()]

        # Append to chain
        chain.append(torch.unsqueeze(torch.cat([current_state1, current_state2], dim=0), dim=0))

    # Stack up the chain and return
    return torch.cat(chain, dim=0)
