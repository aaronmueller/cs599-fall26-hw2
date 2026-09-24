# CS599 Fall 2026 Assignment 5: Alignment

This assignment is adapted from the Spring 2026 edition of Stanford CS336 ([original repository](https://github.com/stanford-cs336/assignment5-alignment)). All credit for its development goes to the Stanford course staff. This README and all of the following code are adapted from theirs. **If you are enrolled in this course at BU and have any questions about this assignment, email Aaron Mueller; do not email the Stanford course staff.**

For a full description of the assignment, see the assignment handout at [hw2.pdf](https://aaronmueller.github.io/teaching/cs599b1_fall26/homeworks/hw2/CS599_HW2_Instructions.pdf).

## Setup

As in the previous homework, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird):
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).