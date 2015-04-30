#!/usr/bin/env python

# Check for setuptools package:

try:
    from setuptools import setup
except ImportError:
    print "no setup tools"
    setuptools = False
    from distutils.core import setup
else:
    setuptools = True

setup_kwargs = {}

# The advantage of setuptools is that EXE wrappers are created on Windows,
# which allows Tab-completion for the script names from the system Scripts
# folder.


# But many people will not have setuptools installed, so we need to handle
# the default Python installation, which only has Distutils:



setup_kwargs['packages'] = ['rnpy',
                            'rnpy.core',
                            'rnpy.imaging',
                            'rnpy.functions',
							'rnpy.legacy']

	

setup(name = "rnpy", 
		version = '0.0.1',
		description = ("Collection of python tools for building and running a resistor network"),
		license = "GNU GENERAL PUBLIC LICENSE v3",
		**setup_kwargs)
