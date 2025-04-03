# vsa-graph

`vsa-graph` is a graph compiler interpretation of our VSA programming lanugage
experiments. The rough idea is this: the representation of syntax, and 
the operations performed over them, can be thought of as a computational graph.

Consider the following lisp example:
```lisp
(cons 'foo 'bar)
```

Roughly, the process has two components:
```
┌──────────────────┐           ┌───────────────────┐
│                  │           │                   │
│                  │           │                   │
│                  │           │                   │
│  Syntactic       ├──────────►│  Evaluation       │
│  Analysis        │           │                   │
│                  │           │                   │
│                  │           │                   │
│                  │           │                   │
└──────────────────┘           └───────────────────┘
```

We can of course break down the syntactic analysis into smaller component
parts, but the more interesting vertex is `Evaluation`. In this step,
we perform optimizations, but in order to execute the script we obey an 
evalutation rule.

Following Tomkins-Flanagan & Kelly (2024), we can think that, instead of 
having distinct _operators_ that are applied to arguments, that we instead 
identify the operators with _graph nodes_.

This does not seem to be a novel approach. For example,
[this project](https://github.com/MahmudulAlam/Holographic-Reduced-Representations),
as well as the [Neuromorphic Intermediate Representation](https://neuroir.org/docs/working_with_nir.html).
We will not be using these tools, as they are complex in their own right
and deserve their own research. However, in principle we ought to be
able to take this code and translate it into NIR, as it is also a
graph-based approach.

# Running and hacking

To run, you must install [uv](https://docs.astral.sh/uv/). On Mac and Linux,
it is:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
On Windows it is:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repository, and `cd` into it. In order to run the script,
enter the command:
```sh
uv run ./src/main.py
```

For personal hacking or any commits, any commit must be formatted using
`black` as well as pass `mypy --strict`.


# Logbook

## 2025-04-3: Beginning

Currently the basics of the graph system are laid out, as well as an 
implementation of (1) an abstract base class for VSAs, (2) HRRs. There is
also a rudimentary graph system in `/graph/async_graph.py` and 
`/graph/sync_graph.py`.

In order to truly experiment with the system as a compilation target for a
programming language, I need to implement a translation from some language
to the graph, respecting semantics.

Something to note about the graph is that, as of right now, there are no
constraints on the kind of graph you can make. It's the wild west of freedom,
so there's nothing stopping users from making terrible graphs with terrible
performance. Figuring out how to constrain the user into making interpretable
graphs is a point of research.
