---
draft: true  # Note: draft=true means this won't appear in the published site.
date: 2025-11-14
authors:
  - jax-team
---


# Example post

Here is some content above the fold.

<!-- more -->

Here is some more content below the fold.

## Latex math

Here is some inline math: $A = B \times C$.

Here is some block math:
$$
\cos x=\sum_{k=0}^{\infty}\frac{(-1)^k}{(2k)!}x^{2k}
$$

# Code blocks
Here is some unhighlighted inline code: `print('abc')`.

Here is some highlighted inline code: `#!python print('abc')`.

Here is a highlighted code block:
```python
import jax

@jax.jit
def f(x):
  return x + 1

print(f(x))
```

Here is a highlighted code block with a title and an inline annotation:
```python title="script.py"
import jax

@jax.jit
def f(x):
  print(x) #(1)
  return x + 1

print(f(x))
```

1.  Because this is within JIT, the *value* of `x` will not be printed; rather
    this will print the *Tracer* object that represents `x`.
