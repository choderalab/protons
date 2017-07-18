#!/usr/bin/env python

"""
Render a protons trajectory with PyMOL
"""

#=============================================================================================
# IMPORTS
#=============================================================================================

import numpy as np
#import Scientific.IO.NetCDF
import netCDF4 as NetCDF
import os, shutil
import os.path
from pymol import cmd
from pymol import util
import mdtraj

# try to keep PyMOL quiet
#cmd.set('internal_gui', 0)
#cmd.feedback("disable","all","actions")
#cmd.feedback("disable","all","results")

#=============================================================================================
# PARAMETERS
#=============================================================================================

trajectory_filename = 'NCMC_trajectory.dcd'
reference_pdb_filename = 'imatinib-solvated-for-calibration.pdb'
atom_name = 'HN48'
png_dir = 'png'
nframes = 10001

if os.path.exists(png_dir):
    shutil.rmtree(png_dir)
os.makedirs(png_dir)

#=============================================================================================
# SUBROUTINES
#=============================================================================================

#=============================================================================================
# MAIN
#=============================================================================================

# Read PDB file into MDTraj
print('Reading trajectory into mdtraj...')
traj = mdtraj.load(reference_pdb_filename)
topology = traj.topology

# Find the proton
atom_index = None
for atom in topology.atoms:
    if atom.name == atom_name:
        atom_index = atom.index
print(atom_index)

# Reset
cmd.rewind()
cmd.delete('all')
cmd.reset()

# Load PDB file into PyMOL
cmd.load(reference_pdb_filename, 'system')
cmd.hide('all')
cmd.select('water', 'resn HOH or resn WAT or resn NA or resn CL')
cmd.select('solute', 'not water and not name HN51')
cmd.select('proton', 'name %s' % atom_name)
cmd.select('nearby_water', 'water within 12 of proton')
cmd.deselect()

#cmd.show('cartoon', 'solute')
#cmd.color('white', 'solute')

# speed up builds
cmd.set('defer_builds_mode', 3)
cmd.set('cache_frames', 0)
cmd.cache('disable')
cmd.set('async_builds', 1)

cmd.set('ray_transparency_contrast', 3.0)
cmd.set('ray_transparency_shadows', 0)

cmd.viewport(640,480)
#niterations = 10 # DEBUG

# Load trajectory
cmd.load_traj(trajectory_filename, object='system')

# Align all states
cmd.intra_fit('solute') # protein

# Zoom viewport
cmd.orient('solute')
cmd.zoom('proton', +5)

# Create one-to-one mapping between states and frames.
cmd.mset("1 -%d" % nframes)


# Delete first frame
cmd.mdelete("1")

#cmd.orient('solute')
#cmd.show('sticks', 'protein')
cmd.show('sticks', 'nearby_water')
#cmd.show('sticks', 'water')
cmd.show('spheres', 'solute')
util.cbay('solute')

# Render movie
#cmd.set('ray_trace_frames', 1)
npause = 30
output_frame = 0

def set_sphere_transparency(selection, value):
    for atom_index in selection:
        cmd.set('sphere_transparency', value, 'id %d' % (atom_index+1))

output_frame = 0
#nframes = 90
cmd.viewport(640,480)
for frame in range(0,nframes,10):
    print "rendering frame %04d / %04d" % (frame+1, nframes)
    cmd.frame(frame+1) # update frames
    # Determine proton indices that should be on and off
    active_protons = list()
    inactive_protons = list()

    value = float(frame+1) / float(nframes)
    set_sphere_transparency([atom_index], value)

    filename = os.path.join(png_dir, 'frame%05d.png' % output_frame)
    cmd.png(filename)
    output_frame += 1
