import abc
import jax
import jax.core


class Jittable(metaclass=abc.ABCMeta):

  def __new__(cls, *args, **kwargs):
    del args, kwargs
    try:
      registered_cls = jax.tree_util.register_pytree_node_class(cls)
    except ValueError:
      registered_cls = cls
    return object.__new__(registered_cls)

  def tree_flatten(self):
    leaves, treedef = jax.tree_util.tree_flatten(self.__dict__)
    switch = list(map(_is_jax_data, leaves))
    children = [leaf if s else None for leaf, s in zip(leaves, switch)]
    metadata = [None if s else leaf for leaf, s in zip(leaves, switch)]
    return children, (metadata, switch, treedef)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    metadata, switch, treedef = aux_data
    leaves = [j if s else p for j, p, s in zip(children, metadata, switch)]
    obj = object.__new__(cls)
    obj.__dict__ = jax.tree_util.tree_unflatten(treedef, leaves)
    return obj


def _is_jax_data(x):
  if isinstance(x, jax.core.Tracer):
    return True

  if type(x) is object:
    return True

  if isinstance(x, (bool, int, float)) or x is None:
    return False

  return jax.core.valid_jaxtype(x)
