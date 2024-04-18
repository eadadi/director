import re
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
import jaxutils
import ninjax as nj
cast = jaxutils.cast_to_compute

class RSSM_PROTO(nj.Module):
    """ The prototype for RSSM base class """
    def inirtial(self, bs):
        return NotImplemented
    def _cell(self, x, prev_state):
        return NotImplemented
    def observe(self, embed, action, is_first, state=None):
        return NotImplemented
    def _cell_scan(self, x, state, first, zero_state):
        return NotImplemented
    def imagine(self, action, state=None):
        return NotImplemented
    def get_dist(self, state, argmax=False):
        return NotImplemented
    def obs_step(self, prev_state, prev_action, embed, is_first):
        return NotImplemented
    def img_step(self, prev_state, prev_action):
        return NotImplemented
    def get_stoch(self, deter):
        return NotImplemented
    def dyn_loss(self, post, prior, impl=None, free=1.0):
        return NotImplemented
    def rep_loss(self, post, prior, impl=None, free=1.0):
        return NotImplemented
    def _stats(self, name, x):
        return NotImplemented
    def _mask(self, value, mask):
        return NotImplemented
    def _mask_step(self, value, mask):
        return NotImplemented

class Linear(nj.Module):

  def __init__(
      self, units, act='none', norm='none', bias=True, outscale=1.0,
      outnorm=False, winit='uniform', fan='avg'):
    self._units = tuple(units) if hasattr(units, '__len__') else (units,)
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    self._outscale = outscale
    self._outnorm = outnorm
    self._winit = winit
    self._fan = fan

  def __call__(self, x):
    shape = (x.shape[-1], np.prod(self._units))
    kernel = self.get('kernel', Initializer(
        self._winit, self._outscale, fan=self._fan), shape)
    kernel = jaxutils.cast_to_compute(kernel)
    x = x @ kernel
    if self._bias:
      bias = self.get('bias', jnp.zeros, np.prod(self._units), np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    if len(self._units) > 1:
      x = x.reshape(x.shape[:-1] + self._units)
    x = self.get('norm', Norm, self._norm)(x)
    x = self._act(x)
    return x


class RSSM(RSSM_PROTO):
  """ A base class with SSM api """
  
  def initial(self, bs):
    return NotImplemented
  
  def _cell(self, x, prev_state):
    return NotImplemented

  def observe(self, embed, action, is_first, state=None):
    if self._nonrecurrent_enc:
      return self.observe_nonrecurrent_enc(embed, action, is_first, state)
    else:
      return self.observe_recurrent_enc(embed, action, is_first, state)

  def observe_recurrent_enc(self, embed, action, is_first, state=None):
    """
    embed.shape (b, t, dim)
    action.shape (b, t, adim)
    is_first.shape (b, t)
    state {
      'deter'.shape (b, deter)
      'hidden'.shape (b, deter)
      'logit'.shape (b, classes, stoch)
      'stoch'.shape (b, classes, stoch)
    }
    """

    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = swap(action), swap(embed), swap(is_first)
    start = state, state
    post, prior = jaxutils.scan(step, inputs, start, self._unroll)
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior
  
  def observe_nonrecurrent_enc(self, embed, action, is_first, state=None):
    """
    action.shape (b, t, adim)
    is_first.shape (b, t,)
    state { t=-1
      'deter'.shape (b, deter,)
      'hidden'.shape (b, deter,)
      'logit'.shape (b, classes, stoch)
      'stoch'.shape (b, classes, stoch)
    }
    """
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    zero_state = self.initial(len(is_first))
    if state is None:
      state = zero_state
    is_first = cast(is_first)
    action = cast(action)
    if self._action_clip > 0.0:
      action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(action)))
    # we assume the non-recurrent enc. formulation here
    # do all observation steps in parallel
    x = self.get('obs_out', Linear, **self._kw)(embed)
    # x.shape (b, t, units)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    post = {'stoch': stoch, **stats} # output posterior samples
    # end obs line
    
    # imagination teacher (== posterior) forcing
    post_stoch = jnp.concatenate([
      state['stoch'][:, None], post['stoch'][:, :-1]], 1)
    if self._classes:
      shape = post_stoch.shape[:-2] + (self._stoch * self._classes,)
      post_stoch = post_stoch.reshape(shape)
    action = self._mask_steps(action, 1.0 - is_first)
    state =  tree_map(lambda x: self._mask(x, 1.0 - is_first[:, 0]), state)
    state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first[:, 0]),
        state, zero_state)
    x = jnp.concatenate([post_stoch, action], -1)
    # x.shape (b)
    x = self.get('img_in', Linear, **self._kw)(x)
    # x.shape (b, t, units)
    _, x = self._cell_scan(
      swap(x), 
      {'deter': state['deter'], 'hidden': state['hidden']}, 
      swap(is_first), # we mask inputs of the cell
      {'deter': zero_state['deter'], 'hidden': zero_state['hidden']}
    )
    sequence, x = cast(x)
    x = self.get('img_out', Linear, **self._kw)(x)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, **stats, **sequence}
    post.update(sequence)
    return cast(post), cast(prior)

  def _cell_scan(self, x, state, first, zero_state):
    # x.shape (t, b, d)
    # inputs = (x, is_first)
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    def step(prev, inputs):
      # inputs = (x, is_first)
      x, is_first = inputs
      prev = tree_map(lambda xx: self._mask(xx, 1.0 - is_first), prev)
      prev = tree_map(
        lambda xx, yy: xx + self._mask(yy, is_first),
        prev, zero_state)
      y, new = cast(self._cell(x, prev))
      # return carry state (this one is lost after the scan)
      # and a pair of carry state and the output to accumulate both
      return new, (new, y)
    carry, res = jaxutils.scan2(step, (x, first), state, modify=False, unroll=self._unroll)
    return carry, tree_map(swap, res)

  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_dist(self, state, argmax=False):
    if self._classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfd.MultivariateNormalDiag(mean, std)

  def obs_step(self, prev_state, prev_action, embed, is_first):
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))
    prior = self.img_step(prev_state, prev_action)
    if self._nonrecurrent_enc:
      x = embed
    else:
      x = jnp.concatenate([prior['deter'], embed], -1)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    post = {'stoch': stoch, 'deter': prior['deter'], 'hidden': prior['hidden'], **stats}
    return cast(post), cast(prior)

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    x, outs = self._cell(x, prev_state)
    x = self.get('img_out', Linear, **self._kw)(x)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, **outs, **stats}
    return cast(prior)

  def get_stoch(self, deter):
    x = self.get('img_out', Linear, **self._kw)(deter)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    return cast(dist.mode())

  def dyn_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
    elif impl == 'logprob':
      loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss

  def rep_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
    elif impl == 'uniform':
      uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
      loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
    elif impl == 'entropy':
      loss = -self.get_dist(post).entropy()
    elif impl == 'none':
      loss = jnp.zeros(post['deter'].shape[:-1])
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss
  
  def _stats(self, name, x):
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))
  
  def _mask_steps(self, value, mask):
    return jnp.einsum('bt...,bt->bt...', value, mask.astype(value.dtype))


