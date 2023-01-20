import jax
import jax.numpy as jnp
import jax.random as jr 
from tqdm import trange

def sample(rng, log_prob, n_steps, current_state, progressbar=True):
    
    # Split the current state (should be of shape (n_walkers, 2))
    current_state1, current_state2 = current_state
    
    # Pull out the number of parameters and walkers
    n_walkers, n_params = current_state1.shape

    # Initial target log prob for the walkers (and set any nans to -inf)...
    logp_current1 = log_prob(current_state1)
    logp_current2 = log_prob(current_state2)

    logp_current1 = jnp.where(
        jnp.isnan(logp_current1),
        jnp.ones_like(logp_current1) * jnp.log(0.), # = -inf
        logp_current1)
    logp_current2 = jnp.where(
        jnp.isnan(logp_current2), 
        jnp.ones_like(logp_current2) * jnp.log(0.), 
        logp_current2)

    # Holder for the whole chain
    chain = [jnp.concatenate([current_state1, current_state2])[jnp.newaxis, ...]]
    
    uniform_kwargs = {"minval" : 0., "maxval" : 1.}
    randint_kwargs = {"minval" : 0, "maxval" : n_walkers}
    
    # Progress bar?
    loop = trange if progressbar else range

    # MCMC loop
    for _ in loop(1, n_steps):

        # Split keys for separate random processes in each epoch
        rng, rng_p1, rng_z1, rng_a1, rng_p2, rng_z2, rng_a2 = jr.split(rng, 7)

        # FIRST SET OF WALKERS
        # Proposals
        ix = jr.randint(rng_p1, shape=(n_walkers,), **randint_kwargs)
        partners1 = current_state2[ix]
        z1 = 0.5 * (jr.uniform(rng_z1, shape=(n_walkers,), **uniform_kwargs) + 1.) ** 2.
        proposed_state1 = partners1 + (z1 * (current_state1 - partners1).T).T

        # Target log prob at proposed points
        logp_proposed1 = log_prob(proposed_state1)#, *args)
        logp_proposed1 = jnp.where(
            jnp.isnan(logp_proposed1), 
            jnp.ones_like(logp_proposed1) * jnp.log(0.), 
            logp_proposed1)

        # Acceptance probability
        p_accept1 = jnp.minimum(
            jnp.ones(n_walkers), 
            z1 ** (n_params - 1.) * jnp.exp(logp_proposed1 - logp_current1))

        # Accept or not
        accept1 = (jr.uniform(rng_a1, shape=(n_walkers,), **uniform_kwargs) <= p_accept1)

        # Update the state
        current_state1 = (current_state1.T * (1. - accept1) + proposed_state1.T * accept1).T
        logp_current1 = jnp.where(accept1, logp_proposed1, logp_current1)

        # SECOND SET OF WALKERS

        # Proposals
        ix = jr.randint(rng_p2, shape=(n_walkers,), **randint_kwargs)
        partners2 = current_state1[ix]
        z2 = 0.5 * (jr.uniform(rng_z2, shape=(n_walkers,), **uniform_kwargs) + 1.) ** 2.
        proposed_state2 = partners2 + (z2 * (current_state2 - partners2).T).T

        # Target log prob at proposed points
        logp_proposed2 = log_prob(proposed_state2)#, *args)
        logp_proposed2 = jnp.where(
            jnp.isnan(logp_proposed2), 
            jnp.ones_like(logp_proposed2) * jnp.log(0.), 
            logp_proposed2)

        # Acceptance probability
        p_accept2 = jnp.minimum(
            jnp.ones(n_walkers), 
            z2 ** (n_params - 1.) * jnp.exp(logp_proposed2 - logp_current2))

        # Accept or not
        accept2 = (jr.uniform(rng_a2, (n_walkers,), **uniform_kwargs) <= p_accept2)

        # Update the state
        current_state2 = (current_state2.T * (1. - accept2) + proposed_state2.T * accept2).T
        logp_current2 = jnp.where(accept2, logp_proposed2, logp_current2)

        # Append to chain
        chain.append(jnp.concatenate([current_state1, current_state2])[jnp.newaxis, ...])

    # Stack up the chain and return    
    return jnp.concatenate(chain, axis=0)