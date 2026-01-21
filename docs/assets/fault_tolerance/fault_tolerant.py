import os
os.environ['XLA_FLAGS'] = ' '.join([
  '--xla_gpu_nccl_terminate_on_error=false',
  '--xla_gpu_nccl_async_execution=true',
  '--xla_gpu_nccl_blocking_communicators=false',
])
os.environ['XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE'] = '1'
os.environ['XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT'] = '1'

from absl import app
from collections.abc import Sequence
from jax.experimental.multihost_utils import live_devices
import jax
import jax.numpy as jnp
import time

def main(args: Sequence[str]) -> None:
  process_id = int(args[1])
  num_processes = int(args[2])
  jax.config.update("jax_enable_recoverability", True)
  jax.distributed.initialize(
      coordinator_address="localhost:9000",
      num_processes=num_processes,
      process_id=process_id,
      local_device_ids=[process_id],
      heartbeat_timeout_seconds=10,
  )

  while True:
    try:
      with live_devices(jax.devices()) as devices:
        print(f'devices = {devices}')
        n, i = len(devices), process_id
        axis_types = (jax.sharding.AxisType.Explicit,)
        mesh = jax.make_mesh((n,), ("i",), devices=devices, axis_types=axis_types)
        jax.set_mesh(mesh)
        x = jax.reshard(jnp.ones(n), jax.P("i"))
        print(jnp.sum(x))
    except Exception as e:
      print('FAIL:', e)
    time.sleep(1)

if __name__ == "__main__":
  app.run(main)
