Installing Protons
==================

Follow these instructions to install ``protons``, and its dependencies.
If you run into any issues, please `raise an issue on our Github page`_.

Install using conda
-------------------

The recommended way to install ``protons`` will be through conda.
You can obtain conda by installing the Anaconda_ python distribution.
It will provide a wide suite of scientific python packages, including some of the dependencies in ``protons``.
Additionally it provides the very powerful ``conda`` python environment and package manager.
For instructions on how to use conda, please `see this page`_ to get you started.

As part of the first release, we will start building conda packages for ``protons``.
You will be able to install ``protons`` using the following conda command.

.. code-block:: bash

  conda install -c omnia protons

This will install the latest version of ``protons`` from the Omnia_ channel.

.. _Omnia: http://www.omnia.md/
.. _Anaconda: https://www.continuum.io/why-anaconda
.. _see this page: http://conda.pydata.org/docs/get-started.html
.. Note::

   Currently, no conda packages are being built.
   You can install dependencies through conda, but the package itself needs to be installed using ``setup.py`` .
   See below for instructions.

Install using setup.py
----------------------

Currently, the only way to install protons is by using the ``setup.py`` script that is provided with the package.

Download a copy of the repository, and cd into it.

.. code-block:: bash

    git clone https://github.com/choderalab/constph-openmm.git
    cd constph-openmm

Them, use the command

.. code-block:: bash

    python setup.py install

to install the package. The ``setup.py`` installation does not automatically check for requirements.
Make sure to install the dependencies separately using ``pip``, or ``conda``. See below for instructions.

Requirements
------------

The ``protons`` package has the following requirements:

    - python 2.7, 3.4 or 3.5
    - openmm_ >=7.0.1
    - numpy_ >=1.10
    - scipy_ >=0.17.0
    - openmmtools_ >=0.7.5
    - pymbar_
    - openmoltools_ >=0.7.0
    - ambermini_
    - joblib_
    - lxml_

These should be available from the `omnia` conda channel.
The following example commands create a new conda environment called "protons", with all dependencies installed.

.. code-block:: bash

    # Point conda to the omnia channel as package source
    conda config --add channels omnia
    # Create a new environment called "protons", with all dependencies included
    conda create --name protons openmm=7.0.1 numpy=1.10 scipy=0.17.0 openmmtools=0.7.5 pymbar openmoltools=0.7.0 ambermini joblib lxml
    # Switch to the new environment
    source activate protons



.. _ambermini: https://github.com/choderalab/ambermini
.. _joblib: https://pythonhosted.org/joblib/
.. _lxml: http://lxml.de/
.. _numpy: http://www.numpy.org/
.. _openmm: http://openmm.org/
.. _openmmtools: https://github.com/choderalab/openmmtools
.. _openmoltools: https://github.com/choderalab/openmoltools
.. _pymbar: https://github.com/choderalab/pymbar
.. _scipy: http://www.scipy.org/


Testing your installation
-------------------------

Our library comes with an extensive test suite, designed to detect potential problems with your installation.
After you've installed ``protons``, it is recommended to verify that the code is working as expected.

To test the installation, run

.. code-block:: bash

    py.test --pyargs protons

This requires that you have installed py.test_ in your python environment.
The output of this command will tell you if any parts of the library are not working correctly.
Note that it may take some time to complete the tests.

If a test fails, please try and verify whether your installation was successful.
You may want to try reinstalling the library in a clean python environment and then testing it.
If your tests still fail, please `raise an issue on our Github page`_.

.. _py.test: http://docs.pytest.org/en/latest/
.. _raise an issue on our Github page: https://github.com/choderalab/constph-openmm/issues/new