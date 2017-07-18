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

system = 'imatinib' # 'imatinib'
system = 'abl-imatinib-full' # 'imatinib'
png_dir = 'png' # BE CAREFUL: This directory will be removed every time you run this

if system == 'imatinib':
    prefix = 'GBAOAB/Imatinib-solvent'
    netcdf_filename = '%s/Imatinib-solvent.nc' % prefix
    trajectory_filename = '%s/Imatinib-solvent.dcd' % prefix
    reference_pdb_filename = '%s/imatinib-solvated-for-calibration.pdb' % prefix
elif system == 'abl-imatinib-full':
    prefix = 'GBAOAB/Abl-Imatinib-full'
    netcdf_filename = '%s/Abl-Imatinib-full.nc' % prefix
    trajectory_filename = '%s/Abl-Imatinib-full.dcd' % prefix
    reference_pdb_filename = '%s/2HYY-H.pdb' % prefix
else:
    raise Exception('System %s unrecognized' % system)

if os.path.exists(png_dir):
    shutil.rmtree(png_dir)
os.makedirs(png_dir)

#=============================================================================================
# SUBROUTINES
#=============================================================================================

#=============================================================================================
# MAIN
#=============================================================================================

#import __main__
#__main__.pymol_argv = [ 'pymol', '-qc']
#import pymol
#pymol.finish_launching()

# Read chemical identities
print("Reading identities from '%s'..." % netcdf_filename)
ncfile = NetCDF.Dataset(netcdf_filename, 'r')
# atom_index[residue,atom] is the 0-indexed index of atom 'atom' in residue 'residue'
# this is a masked array
atom_index = ncfile['/Protons/Metadata/atom_index'][:,:]
# atom_status[iteration,residue,atom] is the status of atom (residue,atom) at time 'iteration'
# 0 - deprotonated
# 1 - protonated
# 255 - this atom does not exist in atom_index[residue,atom] so don't look it up since it will be masked
atom_status = ncfile['/Protons/Titration/atom_status'][:,:,:]
[nframes, nresidues, natoms] = atom_status.shape
print('There are %d frames for %d titratable residues' % (nframes, nresidues))
ncfile.close()

# Read PDB file into MDTraj
print('Reading trajectory into mdtraj...')
traj = mdtraj.load(reference_pdb_filename)
topology = traj.topology

# Find all atom indices bonded to protons in titratable sites
print('Building table of atoms bonded to protons...')
proton_indices = set() # MDTraj zero-indexed proton indices
for residue in range(nresidues):
    for atom in atom_index[residue,:].compressed():
        proton_indices.add(atom)
atoms = [atom for atom in topology.atoms]
bonded_atoms = dict()
for (atom1, atom2) in topology.bonds:
    if (atom1.element == mdtraj.element.hydrogen):
        bonded_atoms[atom1.index] = atom2.index
    if (atom2.element == mdtraj.element.hydrogen):
        bonded_atoms[atom2.index] = atom1.index

titratable_residues = set()
for residue in range(nresidues):
    for atom in atom_index[residue,:].compressed():
        titratable_residues.add(atoms[atom].residue.resSeq)

# Reset
cmd.rewind()
cmd.delete('all')
cmd.reset()

# Load PDB file into PyMOL
cmd.set('suspend_updates', 'on')
cmd.load(reference_pdb_filename, 'system')
cmd.hide('all')
cmd.select('water', 'resn HOH or resn WAT or resn NA or resn CL')
cmd.select('solute', 'not water')
cmd.select('ligand', 'resn STI')
cmd.select('protein', 'solute and not resn STI')
cmd.select('titratable', 'resi ' + '+'.join([str(resSeq+1) for resSeq in titratable_residues]))
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

model = cmd.get_model('system')
#for atom in model.atom:
#    print "%8d %4s %3s %5d %8.3f %8.3f %8.3f" % (atom.index, atom.name, atom.resn, int(atom.resi), atom.coord[0], atom.coord[1], atom.coord[2])

#pymol.finish_launching()

cmd.viewport(640,480)
#niterations = 10 # DEBUG

# Load trajectory
cmd.load_traj(trajectory_filename, object='system')

# Delete waters
cmd.remove('water')

# Align all states
cmd.intra_fit('solute') # protein

# Zoom viewport
cmd.orient('ligand')
cmd.zoom('ligand')

# Create one-to-one mapping between states and frames.
cmd.mset("1 -%d" % nframes)


# Delete first frame
cmd.mdelete("1")

#cmd.orient('solute')
#cmd.show('sticks', 'protein')
cmd.show('sticks', 'titratable')
cmd.show('spheres', 'ligand')
util.cbag('protein')
util.cbay('ligand')

# Render movie
#cmd.set('ray_trace_frames', 1)
#nframes = 100
npause = 30
output_frame = 0
cmd.set('suspend_updates', 'off')
last_active_protons = list()
last_inactive_protons = list()

def set_sphere_transparency(selection, value):
    for atom_index in selection:
        if atoms[atom_index].element == mdtraj.element.hydrogen:
            cmd.set('sphere_transparency', value, 'id %d' % (atom_index+1))

def set_stick_transparency(selection, value):
    for atom1_index in selection:
        if atoms[atom1_index].element == mdtraj.element.hydrogen:
            atom2_index = bonded_atoms[atom1_index]
            cmd.set_bond('stick_transparency', value, 'id %d' % (atom1_index+1), 'id %d' % (atom2_index+1))

#nframes = 30
for frame in range(nframes-1):
    print "rendering frame %04d / %04d" % (frame+1, nframes)
    cmd.frame(frame+1) # update frames
    # Determine proton indices that should be on and off
    active_protons = list()
    inactive_protons = list()

    #cmd.show('spheres', 'solute')
    #cmd.set('sphere_transparency', 0.0)
    for residue in range(nresidues):
        for atom in range(natoms):
            if atom_status[frame,residue,atom] == 0:
                inactive_protons.append(atom_index[residue,atom])
            elif atom_status[frame,residue,atom] == 1:
                active_protons.append(atom_index[residue,atom])

    appear_set = set(active_protons).intersection(last_inactive_protons)
    disappear_set = set(inactive_protons).intersection(last_active_protons)

    if frame == 0:
        set_sphere_transparency(active_protons, 0)
        set_sphere_transparency(inactive_protons, 1)
        set_stick_transparency(active_protons, 0)
        set_stick_transparency(inactive_protons, 1)

    if len(disappear_set) > 0:
        print('disappear: %s' % disappear_set)
        for pause_frame in range(npause):
            value = float(pause_frame+1)/float(npause)
            set_sphere_transparency(disappear_set, value)
            set_stick_transparency(disappear_set, value)
            filename = os.path.join(png_dir, 'frame%05d.png' % output_frame)
            cmd.png(filename)
            output_frame += 1

    if len(appear_set) > 0:
        print('appear: %s' % appear_set)
        for pause_frame in range(npause):
            value = 1.0 - float(pause_frame+1)/float(npause)
            set_sphere_transparency(appear_set, value)
            set_stick_transparency(appear_set, value)
            filename = os.path.join(png_dir, 'frame%05d.png' % output_frame)
            cmd.png(filename)
            output_frame += 1

    last_inactive_protons = inactive_protons
    last_active_protons = active_protons

    if len(appear_set) + len(disappear_set) == 0:
        filename = os.path.join(png_dir, 'frame%05d.png' % output_frame)
        #cmd.set('suspend_updates', 'off')
        cmd.png(filename)
        output_frame += 1
