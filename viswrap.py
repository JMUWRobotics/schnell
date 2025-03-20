#!/usr/bin/env python3

import numpy as np
import vispy as vp
from vispy import app, scene
import subprocess, json
import numpy as np
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'

datadir = os.environ["HOME"] + '/Daten/wasserkiste/foureyes/april'

schnell = subprocess.Popen(
    [
        'builddir/schnell',
        datadir,
        #'--abcd=1,-0.2319539051171811,-0.8465498185840091,0.32044448656555136'
        '--solve'
    ],
    stdout = subprocess.PIPE
)

stdout, stderr = schnell.communicate()

if schnell.wait() != 0:
    print("child died:", stderr.decode().strip() if stderr else "empty stderr")
    exit(schnell.returncode)

stdout, jsondump = stdout.decode().strip().split('DELIMITER')

print(stdout)
j = json.loads(jsondump)

plane_pt   = np.array(j["someplane"]["pt"])
plane_abcd = np.array(j["someplane"]["abcd"])

def meshgrid(pt, abcd, xlim, ylim):
    (xlo, xhi), (ylo, yhi) = xlim, ylim
    xx, yy = np.mgrid[0:101, 0:101] / 100

    xx = xx * (xhi - xlo) / (xx.max() - xx.min()) + xlo
    yy = yy * (yhi - ylo) / (yy.max() - yy.min()) + ylo
        
    a, b, c, _ = abcd
    x0, y0, z0 = pt
        
    z = a*(xx - x0) + b*(yy - y0) / -c + z0
            
    return xx, yy, z

###

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = scene.ArcballCamera(fov=0)
axis = scene.visuals.XYZAxis(
    parent=view.scene,
    pos=(
        (-1,-1,-1), (-0.5,-1,-1),
        (-1,-1,-1), (-1,-0.5,-1),
        (-1,-1,-1), (-1,-1,-0.5)
    ),
    color=(
        (1, 0, 0, 0.8), (1, 0, 0, 0.8),
        (0, 1, 0, 0.8), (0, 1, 0, 0.8),
        (0, 0, 1, 0.8), (0, 0, 1, 0.8)
    )
)

drawn_cams = set()
cam_colors = [
    'k', 'm', 'y', 'c'
]

xlim, ylim = np.array([np.inf, -np.inf]), np.array([np.inf, -np.inf])
plane_n = plane_abcd[:-1]

for i, pair in enumerate(j["stereopairs"]):

    idx1, idx2 = pair["idxs"]
    tag = f"({idx1} $\\rightarrow$ {idx2})"

    T0 = np.array(pair["T0"])
    T1 = np.array(pair["T1"])

    lbackrefr  = np.array(pair["lbackrefr"  ])
    rbackrefr  = np.array(pair["rbackrefr"  ])
    lisects    = np.array(pair["lisects"    ])
    risects    = np.array(pair["risects"    ])

    triangulations = np.array(pair["scenepoints"])

    # xlim[0] = min(np.min(triangulations[:,0]), xlim[0])
    # xlim[1] = max(np.max(triangulations[:,0]), xlim[1])
    # ylim[0] = min(np.min(triangulations[:,1]), ylim[0])
    # ylim[1] = max(np.max(triangulations[:,1]), ylim[1])

    xlim[0] = min(min((T0 + plane_n)[0], (T1 + plane_n)[0]), xlim[0])
    xlim[1] = max(max((T0 - plane_n)[0], (T1 - plane_n)[0]), xlim[1])
    ylim[0] = min(min((T0 + plane_n)[1], (T1 + plane_n)[1]), ylim[0])
    ylim[1] = max(max((T0 - plane_n)[1], (T1 - plane_n)[1]), ylim[1])

    scenepoints = scene.visuals.Markers(
        pos=triangulations,
        parent=view.scene
    )
    restimates = scene.visuals.Markers(
        pos=np.array(pair["restimates"]),
        parent=view.scene,
        face_color='blue'
    )
    lestimates = scene.visuals.Markers(
        pos=np.array(pair["lestimates"]),
        parent=view.scene,
        face_color='blue'
    )
    lback = scene.visuals.Markers(
        pos=-np.array(pair["lback"]) + T0,
        parent=view.scene,
        size=3.5,
        edge_width_rel=0.5,
        edge_color=cam_colors[idx1]
    )
    rback = scene.visuals.Markers(
        pos=-np.array(pair["rback"]) + T1,
        parent=view.scene,
        size=3.5,
        edge_width_rel=0.5,
        edge_color=cam_colors[idx2]
    )
    # for lbr, rbr, li, ri in zip(lbackrefr, rbackrefr, lisects, risects):
    #     scene.visuals.Line(
    #         pos=(li, li + lbr),
    #         color=(0.5,0.5,0.5,0.75),
    #         parent=view.scene
    #     )
    #     scene.visuals.Line(
    #         pos=(ri, ri + rbr),
    #         color=(0.5,0.5,0.5,0.75),
    #         parent=view.scene
    #     )
    if idx1 not in drawn_cams:
        cam1 = scene.visuals.Markers(
            pos=T0.reshape(1, -1),
            parent=view.scene,
            face_color=cam_colors[idx1],
            edge_color='white'
        )
        drawn_cams.add(idx1)
    if idx2 not in drawn_cams:
        cam2 = scene.visuals.Markers(
            pos=T1.reshape(1, -1),
            parent=view.scene,
            face_color=cam_colors[idx2],
            edge_color='white'
        )
        drawn_cams.add(idx2)

normal = scene.visuals.Arrow(
    pos=(
        plane_pt,
        plane_pt + plane_n / 8
    ),
    parent=view.scene
)
plane = scene.visuals.SurfacePlot(
    *meshgrid(plane_pt, plane_abcd, xlim, ylim),
    parent=view.scene
)
plane.attach(vp.visuals.filters.Alpha(0.5))

if __name__ == '__main__':
    app.run()
