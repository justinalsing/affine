import torch
from tqdm import tqdm

def sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2, progress=True, save_lp=False):

    # Progress-bar
    if progress:
        pbar = tqdm(total=n_steps, desc="Sampling")  # Jupyter notebook or qtconsole

    # Initialize current state
    current_state1 = torch.as_tensor(walkers1)
    current_state2 = torch.as_tensor(walkers2)

    # Initial target log prob for the walkers (and set any nans to -inf)...
    logp_current1 = log_prob(current_state1)
    logp_current2 = log_prob(current_state2)

    logp_current1 = torch.as_tensor(logp_current1)
    logp_current2 = torch.as_tensor(logp_current2)

    # sort out any nans
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
    if save_lp is True:
        lp_chain = [torch.cat([logp_current1, logp_current2], axis=0)]


    # MCMC loop
    for epoch in range(1, n_steps):

        # FIRST SET OF WALKERS:

        # Proposals
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
        if save_lp is True:
            lp_chain.append(torch.cat([logp_current1, logp_current2], axis=0))

        # Update the progressbar
        if progress:
            pbar.update(1)

    # Stack up the chain
    chain = torch.stack(chain, axis=0)
    if save_lp is True:
        lp_chain = torch.stack(lp_chain, axis=0)

    if save_lp is True:
        return chain, lp_chain
    else:
        # Chain = np.unique(chain, axis=0) # this may need to be here,
        return chain

def sample_batch(log_prob, n_steps, current_state, n_burnin=0, thin=1, args=[], progress=True, save_lp=False, device="cpu"):
    # Split the current state
    current_state1, current_state2 = current_state

    #Move states to GPU if required
    if device=="cuda":
        current_state1 = current_state1.cuda()
        current_state2 = current_state2.cuda()

    # Pull out the number of parameters, walkers, and batch size
    n_walkers, n_batch, n_params = current_state1.shape

    # Initial target log prob for the walkers (and set any nans to -inf)...
    logp_current1 = log_prob(current_state1, *args)
    logp_current2 = log_prob(current_state2, *args)
    logp_current1[torch.isnan(logp_current1)] = -float('inf')
    logp_current2[torch.isnan(logp_current2)] = -float('inf')

    # Holder for the whole chain
    chain = torch.zeros((int((n_steps-n_burnin)/thin), n_walkers*2, n_batch, n_params), device=device)
    if save_lp is True:
        lpchain = torch.zeros((int((n_steps-n_burnin)/thin), n_walkers*2, n_batch), device=device)

    # Progress bar?
    loop = tqdm(range(1, n_steps)) if progress else range(1, n_steps)

    # counter variable
    counter = 0

    # MCMC loop
    for epoch in loop:
        # First set of walkers:
        # Proposals
        partners1 = current_state2[torch.randint(0, n_walkers, (n_walkers,))]
        z1 = 0.5 * (torch.rand(n_walkers, n_batch, device=device) + 1) ** 2
        proposed_state1 = partners1 + (z1 * (current_state1 - partners1).permute(2, 0, 1)).permute(1, 2, 0)

        # Target log prob at proposed points
        logp_proposed1 = log_prob(proposed_state1, *args)
        logp_proposed1[torch.isnan(logp_proposed1)] = -float('inf')

        # Acceptance probability
        p_accept1 = torch.minimum(torch.ones([n_walkers, n_batch], device=device), z1**(n_params-1) * torch.exp(logp_proposed1 - logp_current1))

        # Accept or not
        accept1_ = (torch.rand([n_walkers, n_batch], device=device) <= p_accept1)
        accept1 = accept1_.type(torch.float32)

        # Update the state
        current_state1 = (current_state1.permute(2, 0, 1) * (1 - accept1) + proposed_state1.permute(2, 0, 1) * accept1).permute(1, 2, 0)
        logp_current1[accept1_.bool()] = logp_proposed1[accept1_.bool()]

        # Second set of walkers:
        # Proposals
        partners2 = current_state1[torch.randint(0, n_walkers, (n_walkers,))]
        z2 = 0.5 * (torch.rand(n_walkers, n_batch, device=device) + 1) ** 2
        proposed_state2 = partners2 + (z2 * (current_state2 - partners2).permute(2, 0, 1)).permute(1, 2, 0)

        # Target log prob at proposed points
        logp_proposed2 = log_prob(proposed_state2, *args)
        logp_proposed2[torch.isnan(logp_proposed2)] = -float('inf')

        # Acceptance probability
        p_accept2 = torch.minimum(torch.ones([n_walkers, n_batch], device=device), z2**(n_params-1) * torch.exp(logp_proposed2 - logp_current2))

        # Accept or not
        accept2_ = (torch.rand([n_walkers, n_batch], device=device) <= p_accept2)
        accept2 = accept2_.type(torch.float32)

        # Update the state
        current_state2 = (current_state2.permute(2, 0, 1) * (1 - accept2) + proposed_state2.permute(2, 0, 1) * accept2).permute(1, 2, 0)
        logp_current2[accept2_.bool()] = logp_proposed2[accept2_.bool()]

        # Append to chain if we're past burnin
        if epoch >= n_burnin and epoch%thin == (thin-1):
            chain[counter] = torch.unsqueeze(torch.cat([current_state1, current_state2], dim=0), dim=0)
            if save_lp is True:
                lpchain[counter] = torch.unsqueeze(torch.cat([logp_current1, logp_current2], dim=0), dim=0)
            counter += 1

    # Stack up the chain and return
    if save_lp is True:
        return chain, lpchain
    else:
        return chain
