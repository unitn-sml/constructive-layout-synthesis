Automating Layout Synthesis with Costructive Preference Eliciation
==================================================================

A python3 implementation of constructive layout synthesis.

Code for the ECML Data Science 2018 submission.

## Requirements

The following packages are required:

- [numpy](http://www.numpy.org/)
- [pymzn](https://github.com/paolodragone/pymzn), tested with version 0.12.2
- [minizinc](https://minizinc.org), tested with version 2.1.4
- [opturion](http://www.opturion.com/), tested with version 1.0.2

## Usage

Type:
```
 $ ./main.py --help
```
to get the full list of options.

To run the experiments for table arrangements:
```
 $ ./run_tables.sh
```

To run the experiments for room partitioning:
```
 $ ./run_rooms.sh
```

