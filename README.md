# Bagpuss

Bagpuss is a lightweight python package for making simulated galaxy catalogues based on simulated universes.
Fake cats created with bagpuss are intended for testing cosmological inference pipelines, and aren't directly designed to be realistic.

## What's in a name?

> Wake up, be bright, be golden and light

Bagpuss is the name of a beloved toy cat from a classic 1970s British children's television show.

## How Bagpuss works

Bagpuss starts by simulating a Universe according to both a cosmological model, a galaxy formation model, and a luminosity model.
Galaxies are then drawn from the distribution defined by that simulation to create a set of galaxies.

The observed galaxy catalogue is then created by applying a suitable selection function to that set of galaxies.
This provides a fake catalogue of galaxies.

We can then create a fake gravitational-wave catalogue for our universe.
To do this we simulate the population of binary black holes (BBHs) to the total galaxy set.
We perform injections of the gravitational-wave signals from these BBHs into noise (either real or simulated) to determine which of these events would be observed by a given gravitational-wave detector network.
We then need to perform a selection on these events to determine the observed galaxy catalogue.

We then perform parameter estimation on the final set of super-threshold observed events to produce posterior probability distributions for each event, and skymaps.

These data products can then be ingested by an inference pipeline to perform cosmological inference.

## How can I learn more?

You can read more about how bagpuss works in its documentation.

## How can I use it?

Bagpuss is licensed under the MIT License, and is available on PyPI.

You can install bagpuss with pip:

```console
$ pip install bagpuss
```

## Who made this?

Bagpuss was created by researchers at the Institute for Gravitational Research at the University of Glasgow.
The development of Bagpuss is led by [Daniel Williams](https://www.github.com/transientlunatic).
For a full list of contributors please see the [CONTRIBUTORS](CONTRIBUTORS.md) file.