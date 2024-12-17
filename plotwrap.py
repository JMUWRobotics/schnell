#!/usr/bin/env python3

import subprocess, json
import numpy as np
from matplotlib import pyplot as plt

schnell = subprocess.Popen(
    [
        'builddir/schnell',
        '../wasserkiste/no-waves/0_left.png',
        '../wasserkiste/no-waves/0_right.png',
        '../Kalibrierungen/DISCO3D/Luft/left.yaml',
        '../Kalibrierungen/DISCO3D/Luft/right.yaml',
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

print(baseline)

fig = plt.figure()
ax  = fig.add_subplot(projection='3d')

ax.scatter(scenepts[:, 0], scenepts[:, 1], scenepts[:, 2], marker='.', color='green', label='correspondence')
ax.scatter(0, 0, 0, marker='o', color='black', label='left cam')
ax.scatter(*baseline, marker='o', color='red', label='right cam')

ax.scatter(lestimates[:, 0], lestimates[:, 1], lestimates[:, 2], marker='.', color='blue', label='estimates, left')
ax.scatter(restimates[:, 0], restimates[:, 1], restimates[:, 2], marker='.', color='cyan', label='estimates, right')

plt.legend()
plt.show()