# Create a movie from a protons trajectory

Create a conda environment
```bash
conda create -n pymol -c mw pymol netcdf4 mdtraj
```

Launch pymol:
```bash
pymol
```

Run `render_trajectory.py` within pymol:
```bash
run render_trajectory.py
```

Render the frames into a movie:
```bash
ffmpeg -r 30 -i png/frame%05d.png -r 15 -b:v 5000000 -f mp4 -vcodec h264 -pix_fmt yuv420p -y movie.mp4
```