# PointlightModel.jl

This is currently catch-all scratch code for the PointlightModel project.
Expect that it won't be possible to use this without being in touch with the
author(s).

## Installing dependencies

There is one unregistered dependency which must be installed first:
```julia
import Pkg
Pkg.add(url="git@github.com:probcomp/GenGridEnumeration.jl.git", rev="master")

## Or, if you already have GenGridEnumeration cloned locally and want to use
## that:
# Pkg.develop(path="/path/to/GenGridEnumeration")
```

The rest of the dependencies can be installed automatically by the package
manager as follows:
```julia
import Pkg
Pkg.activate("/path/to/PointlightModel")
Pkg.instantiate()
```

## How to develop with it

In your script / REPL / notebook, either use the `PointlightModel` Julia
project directly, or use a different Julia project from which `PointlightModel`
is `Pkg.develop`ed, i.e., `Pkg.develop(path="/path/to/PointlightModel")`.
Then:

```julia
import Revise
import PointlightModel
import PointlightModel: name1, name2, ...
```

or if you prefer,

```julia
import Revise
import PointlightModel
P = PointlightModel

P.foo(P.bar())
```

Note the `import Revise` before the other imports.  This allows
[Revise.jl](https://timholy.github.io/Revise.jl/stable/) to automatically
incorporate any changes you make to the package code during the REPL session.
If Revise encounters errors trying to do this update, it will complain in the
REPL, and its errors can also be viewed by running `Revise.errors()`.  You can
manually reload a tracked module by running e.g.
`Revise.revise(PointlightModel)`.
