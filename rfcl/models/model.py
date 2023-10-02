"""
Model class
"""
import os
from typing import Any, Callable, Optional

import flax
import flax.linen as nn
import flax.serialization
import optax
from chex import PRNGKey
from flax import struct


Params = flax.core.FrozenDict[str, Any]
# T = TypeVar("T", bound=nn.Module)


@struct.dataclass
class Model:
    """
    Model class that holds the model parameters and training state.
    Provides wrapped functions to execute forward passes in OOP style.
    similar to flax TrainState

    calling Model.create returns the original nn.Module but additional functions

    expects all parameters to optimized by a single optimizer
    """

    model: nn.Module = struct.field(pytree_node=False)
    params: Params
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    step: int = 0

    @classmethod
    def create(
        cls,
        model: nn.Module,
        key: PRNGKey,
        sample_input: Any = [],
        tx: Optional[optax.GradientTransformation] = None,
    ) -> "Model":
        if isinstance(sample_input, list):
            model_vars = model.init(key, *sample_input)
        else:
            model_vars = model.init(key, sample_input)
        opt_state = None
        if tx is not None:
            opt_state = tx.init(model_vars)
        return cls(
            model=model,
            params=model_vars,
            opt_state=opt_state,
            apply_fn=model.apply,
            tx=tx,
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn(self.params, *args, **kwargs)

    def apply_gradients(self, grads):
        updates, updated_opt_state = self.tx.update(grads, self.opt_state, self.params)
        updated_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=updated_params, opt_state=updated_opt_state)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.state_dict()))

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            data = flax.serialization.from_bytes(self.state_dict(), f.read())
        return self.replace(**data)

    def state_dict(self):
        return dict(
            params=self.params,
            opt_state=self.opt_state,
            step=self.step,
        )

    def load_state_dict(self, state_dict):
        return self.replace(**state_dict)

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # if attribute is another module, can we scope it?
            attr = self.model.__getattribute__(name)
            return attr
