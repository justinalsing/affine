import torch
from tqdm import tqdm

def sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2, progress=True, save_lp=False, save_ar=False):
    """
    Run affine inavriant MCMC to sample a single posterior.

    This uses the parallel stretch move (Foreman-Mackey et al. 2013)
    to evolve two simultaneous ensembles of walkers.

    Parameters
    ----------
    log_prob : callable
        Function for evaluating the target log posterior. Should take a
        single `torch.Tensor` of parameters as input, and should return
        a `torch.Tensor` of log probabilities. For an input of shape
        `(n_walkers, n_params)`, `log_prob` should return shape `(n_walkers,)`.
    n_params : int
        Number of model parameters being sampled.
    n_walkers : int
        Number of walkers per ensemble. Total number of chains will 
        be `2*n_walkers`.
    n_steps : int
        Number of steps to take. One step evolves all walkers in the
        two ensembles, so the total number of posterior samples per
        parameter will be `2 * n_walkers * n_steps`.
    walkers1 : torch.Tensor
        Initial positions for the first ensemble of walkers. Should have
        shape `(n_walkers, n_params)`.
    walkers2 : torch.Tensor
        Initial positions for the second ensemble of walkers. Should have
        shape `(n_walkers, n_params)`.
    progress : bool, optional
        If `True`, shows a progress bar using `tqdm`. Default is `True`.
    save_lp : bool, optional
        If `True`, saves the log probability of each MCMC sample.
        Default is `False`.
    save_ar : bool, optional
        If `True`, saves the acceptance rate of proposed steps.
        Default is `False`. 

    Returns
    -------
    chain : torch.Tensor
        MCMC samples of parameters. Has shape `(n_steps, 2*n_walkers, n_params)`.
    lp_chain : torch.Tensor, optional
        Log probability for each MCMC sample. Shape `(n_steps, 2*n_walkers)`.
        Only returned if `save_lp=True`.
    ar : torch.Tensor, optional
        Proposal acceptance rate for each walker. Shape  `(2*n_walkers,)`.
        Only returned if `save_ar=True`.
    """

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
    if save_ar is True:
        ar = torch.zeros(2*n_walkers)

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
        if save_ar is True:
            ar[:n_walkers] = ar[:n_walkers] + accept1/(n_steps)
            ar[n_walkers:] = ar[n_walkers:] + accept2/(n_steps)

        # Update the progressbar
        if progress:
            pbar.update(1)

    # Stack up the chain
    chain = torch.stack(chain, axis=0)
    if save_lp is True:
        lp_chain = torch.stack(lp_chain, axis=0)

    if save_lp is True:
        if save_ar is True:
            return chain, lp_chain, ar
        else:
            return chain, lp_chain
    else:
        if save_ar is True:
            return chain, ar
        else:
            return chain

def sample_batch(log_prob, n_steps, current_state, n_burnin=0, thin=1, args=[], progress=True, save_lp=False, save_ar=False, device="cpu"):
    """
    Run affine inavriant MCMC to sample a batch of posteriors.

    This uses the parallel stretch move (Foreman-Mackey et al. 2013)
    to evolve two simultaneous ensembles of walkers.

    Calls to the `log_prob` will be vectorized over multiple posteriors
    that are samples simultaneously (e.g. if fitting multiple independent
    datasets with the same model).

    Parameters
    ----------
    log_prob : callable
        Function for evaluating the target log posterior. Should take a
        `torch.Tensor` of parameters as input, and should return a
        `torch.Tensor` of log probabilities. For an input of shape
        `(n_walkers, n_batch, n_params)`, `log_prob` should return 
        shape`(n_walkers, n_batch)`.
    n_steps : int
        Number of steps to take. One step evolves all walkers in the
        two ensembles, for all posteriors in the batch, so the total number 
        of samples per parameter per posterior will be `2 * n_walkers * n_steps`.
    current_state : tuple of torch.Tensor
        Initial positions for the walkers. The tuple should contain two
        tensors, corresponding to the initial positions of the two
        parallel ensembles. Each tensor should have shape
        `(n_walkers, n_batch, n_params)`, where `n_walkers` is the number
        of walkers per parallel ensemble, `n_batch` is the batch size (i.e.
        number of posteriors to be sampled simultaneously), and `n_params`
        is the number of parameters being sampled.
    n_burnin : int, optional
        Chains will only be stored after `n_burnin` steps have been passed.
        Only recommended if memory / storage are an issue. Default is 0 (i.e.
        all steps from the beginning are stored).
    thin : int, optional
        Only stores samples every `thin` iterations. Default is 1 (i.e. all
        iterations are stored).
    args : list, optional
        Additional positional arguments to be handed to the `log_prob`.
    progress : bool, optional
        If `True`, shows a progress bar using `tqdm`. Default is `True`.
    save_lp : bool, optional
        If `True`, saves the log probability of each MCMC sample.
        Default is `False`.
    save_ar : bool, optional
        If `True`, returns the acceptance rate for each walker. Default is `False`.
    device : str or torch.device, optional
        Device to perform operations on. Default is `'cpu'`.

    Returns
    -------
    chain : torch.Tensor
        MCMC samples of parameters for all posteriors in the batch.
        Has shape `[(n_steps-n_burnin)/thin, 2*n_walkers, n_batch, n_params]`.
    lp_chain : torch.Tensor, optional
        Log probability for each MCMC sample (if `save_lp=True`).
        Has shape `[(n_steps-n_burnin)/thin, 2*n_walkers, n_batch]`.
    acceptance_rate : torch.Tensor, optional
        Proposal acceptance rate for each walker (if `save_ar=True`).
        Has shape  `[2*n_walkers, n_batch]`.
    """
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
    if save_ar is True:
        ar = torch.zeros((n_walkers*2, n_batch), device=device)

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

        # Update acceptance rate if we're past burnin
        if epoch >= n_burnin and save_ar is True:
            ar[:n_walkers] = ar[:n_walkers] + accept1/(n_steps - n_burnin)
            ar[n_walkers:] = ar[n_walkers:] + accept2/(n_steps - n_burnin)

    # Stack up the chain and return
    if save_lp is True:
        if save_ar is True:
            return chain, lpchain, ar
        else:
            return chain, lpchain
    else:
        if save_ar is True:
            return chain, ar
        else:
            return chain
