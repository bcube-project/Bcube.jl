```@meta
CurrentModule = Bcube
```

# Bcube

## Purpose of Bcube

Bcube is a Julia library providing tools for the spatial discretization of partial differential equation(s) (PDE). The main objectives are:

- to provide a set of tools to quickly assemble an algorithm solving partial differential equation(s) (so the main objective is to help building prototypes without thinking about the numerical core)
- to be completed : efficient/performant PDE resolution?

This documentation is organised as follow. Checkout the tutorials to see what Bcube is capable of and/or quickly learn how to use it. Then, some more elaborated examples are provided to demonstrate the library capabilities. The "Manual" part explains how the core is organized. Finally, the "API" section is the low level code documentation.

## Writing documentation

To write documentation for Bcube, Julia's guidelines should be followed : [https://docs.julialang.org/en/v1/manual/documentation/](https://docs.julialang.org/en/v1/manual/documentation/). Moreover, this project tries to apply the [SciML Style Guide](https://github.com/SciML/SciMLStyle).

## Conventions

This documentation follows the following notation or naming conventions:

- coordinates inside a reference frame are noted $$\hat{x}, \hat{y}$$ or $$\xi, \eta$$ while coordinates in the physical frame are noted $$x,y$$
- when talking about a mapping, $$F$$ or sometimes $$F_{rp}$$ designates the mapping from the reference element to the physical element. On the other side, $$F^{-1}$$ or sometimes $$F_{pr}$$ designates the physical element to the reference element mapping.
- "dof" means "degree of freedom"
