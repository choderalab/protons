# Create a movie from a protons trajectory

Create a conda environment
```bash
conda create -n pymol -c mw pymol netcdf4 mdtraj
```

Launch pymol:
```bash
pymol
```

Run `render_trajectory.py` within `pymol`:
```bash
run render_trajectory.py
```
or, to generate a movie from a single NCMC switch within `pymol`,
```bash
run render_ncmc_switching_trajectory.py
```
Note that you will need to edit the parameters at the top of the script, such as:
```
prefix = 'GBAOAB/Imatinib-solvent'
netcdf_filename = '%s/Imatinib-solvent.nc' % prefix
trajectory_filename = '%s/Imatinib-solvent.dcd' % prefix
reference_pdb_filename = '%s/imatinib-solvated-for-calibration.pdb' % prefix
```
or use the presets provided.

Render the frames into a movie:
```bash
ffmpeg -r 30 -i png/frame%05d.png -r 15 -b:v 5000000 -f mp4 -vcodec h264 -pix_fmt yuv420p -y movie.mp4
```
If this complains that your movie is not 640x480, something went wrong during the rendering and the viewport dimensions changed away from 640x480. Reset this in `pymol` within
```
cmd.viewport(640,480)
```
and try again.
