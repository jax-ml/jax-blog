from absl import app
from collections.abc import Sequence
import jax
import jax.numpy as jnp
import time

def main(args: Sequence[str]) -> None:
  process_id = int(args[1])
  num_processes = int(args[2])
  jax.distributed.initialize(
      coordinator_address="localhost:9000",
      num_processes=num_processes,
      process_id=process_id,
      local_device_ids=[process_id],
      heartbeat_timeout_seconds=10,
  )

  n, i = num_processes, process_id
  mesh = jax.make_mesh((n,), ("i",), axis_types=(jax.sharding.AxisType.Explicit,))
  jax.set_mesh(mesh)
  x = jax.reshard(jnp.ones(n), jax.P("i"))
  while True:
    print(jnp.sum(x))
    time.sleep(1)

if __name__ == "__main__":
  app.run(main)
