#!/usr/bin/env python3

import subprocess, json
import numpy as np
import os
from matplotlib import pyplot as plt

datadir = os.environ["HOME"] + '/Daten'

schnell = subprocess.Popen(
    [
        'builddir/schnell',
        datadir + '/wasserkiste/no-waves-pattern/0_left.png',
        datadir + '/wasserkiste/no-waves-pattern/0_right.png',
        datadir + '/Kalibrierungen/DISCO3D/Luft/left.yaml',
        datadir + '/Kalibrierungen/DISCO3D/Luft/right.yaml',
        'april'
    ],
    stdout = subprocess.PIPE
)

stdout, stderr = schnell.communicate()

if schnell.wait() != 0:
    print("child died:", stderr.decode().strip())
    exit(schnell.returncode)

j = json.loads(stdout)

baseline   = np.array(j["baseline"   ])
restimates = np.array(j["restimates" ])
lestimates = np.array(j["lestimates" ])
plane_pt   = np.array(j["someplane"  ]["pt"])
plane_abcd = np.array(j["someplane"  ]["abcd"])
scenepts   = np.array(j["scenepoints"])

def meshgrid(pt, abcd, xlim, ylim):
    (xlo, xhi), (ylo, yhi) = xlim, ylim
    xx, yy = np.mgrid[0:101, 0:101] / 100

    xx = xx * (xhi - xlo) / (xx.max() - xx.min()) + xlo
    yy = yy * (yhi - ylo) / (yy.max() - yy.min()) + ylo
        
    a, b, c, _ = abcd
    x0, y0, z0 = pt
        
    z = a*(xx - x0) + b*(yy - y0) / -c + z0
            
    return xx, yy, z

fig = plt.figure()
ax  = fig.add_subplot(projection='3d')

ax.plot_surface(*meshgrid(plane_pt, plane_abcd, (-0.2, 0.5), (-0.2, 0.2)), alpha = 0.2, label='water surface')

ax.scatter(scenepts[:, 0], scenepts[:, 1], scenepts[:, 2], marker='.', color='green', label='correspondence')
ax.scatter(0, 0, 0, marker='o', color='black', label='left cam')
ax.scatter(*baseline, marker='o', color='red', label='right cam')

ax.scatter(lestimates[:, 0], lestimates[:, 1], lestimates[:, 2], marker='.', color='blue', label='estimates, left')
ax.scatter(restimates[:, 0], restimates[:, 1], restimates[:, 2], marker='.', color='cyan', label='estimates, right')

plt.legend()
plt.show()