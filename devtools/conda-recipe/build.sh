#!/bin/bash

install_dir=$(${PYTHON} setup.py install | awk '/Installed/{print $2}')
rm ${install_dir}/constph/diagnostics.py*

