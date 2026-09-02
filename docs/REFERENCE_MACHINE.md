# Reference machine

The reference machine provides a reproducible coordinate, not a universal ban
on other machines.

## Authoritative coordinate

- implementation: CPython
- version: 3.13
- operating-system family: Linux
- dependencies: Python standard library only
- proof command: `PYTHONPATH=src python -m unittest discover -s tests -v`

## Compatibility transitions

CPython 3.11 and 3.12 are exercised by CI using the same test suite. A platform
or interpreter is added to the compatibility matrix only after its actual
results pass. Platform labels do not substitute for evidence.

Termux is a planned native target. It should receive its own smoke proof rather
than being inferred from generic Linux CI.
