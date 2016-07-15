How to set up the documentation
-------------------------------

Part one of this procedure sets up a documentation framework using Sphinx. Part two explains how you can use readthedocs.org to build your documentation automatically, and how to use conda to install the dependencies on the readthedocs server.

Set up sphinx
=============

1. Run ``sphinx-quickstart`` and follow instructions to set up your documentation. If you run this in a ``docs`` subdirectory, you can prevent it from cluttering up your code directories. Autodoc is recommended for python projects. You can also enable mathjax for displaying math. Be sure to generate a MakeFile and a make batch file for windows, since you'll need these to build your documentation. 

2. Modify ``conf.py`` to add your library to the path of the script, for instance if your documentation lives in a ``docs`` subdirectory of the project.

Example::

  # If extensions (or modules to document with autodoc) are in another directory,``
  # add these directories to sys.path here. If the directory is relative to the
  # documentation root, use os.path.abspath to make it absolute, like shown here.
    sys.path.insert(0, os.path.abspath('..'))

3. Adjust themes, and other options that you like in ``conf.py``.


4. You can add documentation by editing ``index.rst``, which uses reST_. Please read the sphinx documentation for further instructions on how to write your docs.

.. _reST: http://www.sphinx-doc.org/en/stable/rest.html

5. Test your docs using ``make html`` as a command. Depending on what build directory you specified (default ``_build``), you will find the html documentation there. There is no need to commit this. You may wish to add the directory to your ``.gitignore`` file.


6. Fix any errors or warnings that may pop up. Common errors are docstrings including ``rst`` syntax errors, and missing python dependencies.


Read the docs!
==============

Next, the documentation needs to be built on www.readthedocs.org. This next section is important if you use conda packages to run your code and build your docs.

Enable a conda environment for ReadTheDocs_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

7. Describe the necessary conda environment in a yml file, including the channels from which to download packages. 

Example::

  name: constph
  channels:
      - omnia
  dependencies:
      - python
      - openmm ==7.0rc1
      - numpy >=1.10
      - scipy >=0.17.0
      - openmmtools ==0.7.5
      - pymbar
      - openmoltools ==0.7.0
      - ambermini
      - joblib
      - lxml

8. In the top level directory of the repository, place a file called ``readthedocs.yml`` that links to it.

Example::
    
  conda:
      file: docs/environment.yml
     
     
Website settings
~~~~~~~~~~~~~~~~

9. Enable the repository on ReadTheDocs_. Now, everytime code is pushed to master, or a release is made, new docs are automatically built. 

10. While writing documentation, you can use ``make html`` to build it locally to see a preview. If all goes well, your docs should look exactly the same on ReadTheDocs_.

.. _ReadTheDocs: https://www.readthedocs.org 



Disclaimer
~~~~~~~~~~

Note that conda is currently a beta feature of readthedocs. The exact way this will work might change.

Happy documenting!
==================

Author: Bas Rustenburg
