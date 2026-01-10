---
draft: true
date: 2026-01-10
authors:
  - jax-team
tags:
  - refs
---

# JAX refs: mutable state in a functional world
JAX version 0.8.0 introduced a new feature that the team has been working on for a while: mutable array references, or *refs* for short.
These new ref objects provide for limited-scope mutable state within JAX's otherwise purely functional paradigm, and we believe it's going to unlock a whole new level of flexibility and expressiveness when working in JAX.

<!-- more -->

## What are refs and why do we need them?
JAX's core APIs are built around functional programming, where functions are [pure](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions) and free from side effects.
This design principle is a major strength, enabling powerful transformations like automatic differentiation (`jax.grad`), vectorization (`jax.vmap`), and compilation (`jax.jit`) with efficient and predictable behavior.

However, there are scenarios where managing mutable state can significantly simplify code and improve performance and usability: for example, when writing custom kernels for GPU or TPU with JAX's [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernel language, direct management of memory is essential.

This is where refs come in.

You can think of a ref as a pointer to an array buffer that can be both read from and written to, allowing for controlled mutation within an otherwise purely-functional JAX computation.
Refs are designed to be explicitly managed, meaning you declare and interact with them intentionally, preserving the transparency that makes JAX so robust.

## Mutable state in a functional API: a harmony
You might be wondering: doesn't introducing mutable state go against JAX's functional philosophy?
It's a valid question, and one we've carefully considered.
The answer lies in the careful way that refs are integrated.

Refs are not designed to replace JAX's functional core, but rather to augment it.
Their available APIs are essentially limited to reading the value to the ref, and writing a value to the ref. As such, refs cannot replace familiar `jax.Array` objects in general; rather, they offer an opt-in mechanism for managing state when it’s truly beneficial.

Crucially, transformations like `jit`, `grad`, `vmap`, and `shard_map` have been augmented to work cleanly with refs, allowing you to blend functional purity with explicit, controlled mutability.
We envision refs being used in specific, well-defined contexts where their advantages outweigh the complexity of departing from pure immutability.

We believe that by providing this escape hatch, JAX becomes even more versatile, empowering you to tackle a wider range of problems without sacrificing the performance and composability you've come to expect.

## Refs in action
Here's a quick example of how refs can be used to update a mutable global state within a JIT-compiled function:
```python
>>> import jax

>>> mutable_value = jax.new_ref(jax.numpy.array([0])) #(1)

>>> @jax.jit
... def f():
...   old_val = jax.ref.get(mutable_value)
...   mutable_value[0] += 100  # in-place mutation!
...   return old_val  #(2)

>>> mutable_value
Ref([0], dtype=int32)

>>> f()  # functkon returns old_val
Array([0], dtype=int32)

>>> mutable_value  # global ref is updated as a result.
Ref([100], dtype=int32)
```

1. Refs can be created with the `jax.new_ref` function, by passing a JAX array,
   NumPy array, or Python scalar.
2. Notice that `mutable_value` is not returned from the function as required for
   standard JAX arrays: rather the global value is updated as a side-effect of
   executing the function `f`.

To read more about refs and how you can use them, refer to the [`jax.Ref` documentation](https://docs.jax.dev/en/latest/array_refs.html) in JAX's main docs.

We're excited to see how the community utilizes this powerful new feature!
