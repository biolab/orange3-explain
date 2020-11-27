Orange3 Explain
===============

Orange3 Explain is an add-on for the [Orange3](http://orange.biolab.si) 
data mining suite. It provides extensions for explanatory AI.

Installation
------------
Install from Orange add-on installer through Options - Add-ons.

To install the add-on from source run

    pip install .

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    pip install -e .

Usage
-----

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python -m Orange.canvas

The new widget appears in the toolbox bar under the section Example.
