Constructive Layout Synthesis via Coactive Learning
===================================================

A python3 implementation of constructive layout synthesis.

Please see our paper:

    Paolo Dragone, Luca Erculiani, Maria Teresa Chietera, Stefano Teso, Andrea Passerini.
    "Constructive Layout Synthesis via Coactive Learning",
    CML workshop at NIPS'16, 2016.

## Requirements

The following packages are required:

- [numpy](http://www.numpy.org/)
- [pymzn](https://github.com/paolodragone/pymzn), tested with version 0.10.8
- [minizinc](https://minizinc.org), tested with version 2.0.13
- [opturion](http://www.opturion.com/), tested with version 1.0.2

## Usage

Type:
```
 $ ./nips16.py --help
```
to get the full list of options.

To run an experiment, type:
```
 $ ./nips16.py exp furn -W users/rand_users.pickle -O "result.pickle" -n 0.1 -S ${size} -t ${tables}
```
where `${size}` is the size of the canvas and `${tables}` is the number of tables.

## Funding

The project is supported by the CARITRO Foundation through grant 2014.0372.

