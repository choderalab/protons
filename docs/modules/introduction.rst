Introduction
============

.. toctree::
   :name: introductiontoc
   :maxdepth: 3
   :numbered:

   introduction

What is protons?
----------------

Protons is a python module that implements a constant-pH molecular dynamics scheme for sampling protonation states
and tautomers of amino acids and small molecules in OpenMM.

The codebase is currently under development by the `Chodera lab`_ at Memorial Sloan Kettering Cancer center.
There is no official release of the code yet.

If you are interested in the development of this code, you can watch the `repository on Github`_.
For the planned features of the code, see :ref:`roadmap`.

.. _Chodera lab: http://www.choderalab.org
.. _repository on Github: https://github.com/choderalab/protons

Installing Protons
------------------

Follow these instructions to install ``protons``, and its dependencies.

Install using conda
^^^^^^^^^^^^^^^^^^^

The recommended way to install ``protons`` is through conda.
You can obtain conda by installing the Anaconda_ python distribution.
For instructions on how to use conda, please `see this page`_ to get you started.

As part of the first release, we will start building conda packages for ``protons``.
You will be able to install ``protons`` using the following conda command.

.. code-block:: bash

  conda install -c omnia protons

This will install the latest version of ``protons`` from the Omnia_ channel.

.. _Omnia: http://www.omnia.md/
.. _Anaconda: https://www.continuum.io/why-anaconda
.. _see this page: https://conda.io/docs/user-guide/getting-started.html
.. Note::

   Currently, no official release conda packages are being built.
   A development release can be installed using the ``-c omnia/label/dev`` channel instead.
   Note that the ``dev`` releases may include insufficiently tested features.
   

Install latest development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Note:: This is only recommended for those developing the codebase.

You can install the latest development version using the conda recipe present in the Github _repository. 
First, install conda, and acquire conda-build_, which is necessary to compile a conda package.

Download a copy of the repository, and cd into it.

.. code-block:: bash

    git clone https://github.com/choderalab/protons.git
    cd protons

.. _conda-build: https://conda.io/docs/user-guide/tasks/build-packages/install-conda-build.html

If you are using bash, you can then use the following commands

.. code-block:: bash

   # Create an environment for development
  - conda create --yes -n protons-development python=3.6
  - source activate protons-development
  - conda config --add channels omnia
  # Add dev channels
  - conda config --add channels omnia/label/dev
  - conda build devtools/conda-recipe
  - conda install --yes --use-local protons-dev

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

.. _py.test: http://docs.pytest.org/en/latest/


