#!/usr/bin/env python3

import numpy as np
import vispy as vp
from vispy import app, scene
import subprocess, json
import numpy as np
import os
import sys

os.environ['QT_QPA_PLATFORM'] = 'xcb'

datadir = os.environ["HOME"] + '/Daten/wasserkiste/foureyes/april'

schnell = subprocess.Popen(
    ['builddir/schnell', datadir] + sys.argv,
    stdout = subprocess.PIPE
)

stdout, stderr = schnell.communicate()

if schnell.wait() != 0:
    print("child died:", stderr.decode().strip() if stderr else "empty stderr")
    exit(schnell.returncode)

stdout, jsondump = stdout.decode().strip().split('DELIMITER')

print(stdout)
data = json.loads(jsondump)
steps = data["steps"]

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

def intersect(abcd, p, n):
    n = n / np.linalg.norm(n)
    ax, ay, az = p + n
    bx, by, bz = p
    a = np.array([ax, ay, az, 1]).reshape(-1, 1)
    b = np.array([bx, by, bz, 1]).reshape(-1, 1)
    pluecker = a @ b.T - b @ a.T
    x, y, z, w = pluecker.T @ abcd

    if np.isclose(w, 0):
        return None

    return np.array([x / w, y / w, z / w])

drawn_cams = set()
cam_colors = [
    'k', 'm', 'y', 'c'
]

draw_idx = 0
curscene = []

def draw(_):
    global draw_idx
    global curscene
    j = steps[draw_idx]
    plane_pt   = np.array(j["plane"]["pt"])
    plane_abcd = np.array(j["plane"]["abcd"])

    print(draw_idx)

    draw_idx += 1

    xlim, ylim = np.array([np.inf, -np.inf]), np.array([np.inf, -np.inf])
    plane_n = plane_abcd[:-1]

    for elem in curscene:
        elem.parent = None

    curscene.clear()

    normals = []
    means = []

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
        restimates: np.ndarray = np.array(pair["restimates"])
        lestimates: np.ndarray = np.array(pair["lestimates"])

        evals, evecs = np.linalg.eig(np.cov(lestimates.T))
        lnormal = evecs[:, np.argmin(evals)]
        evals, evecs = np.linalg.eig(np.cov(restimates.T))
        rnormal = evecs[:, np.argmin(evals)]

        normals.append((lnormal + rnormal) / 2)
        means.append( (restimates.mean(axis=0) + lestimates.mean(axis=0)) / 2 )

        # xlim[0] = min(np.min(triangulations[:,0]), xlim[0])
        # xlim[1] = max(np.max(triangulations[:,0]), xlim[1])
        # ylim[0] = min(np.min(triangulations[:,1]), ylim[0])
        # ylim[1] = max(np.max(triangulations[:,1]), ylim[1])

        xlim[0] = min(min((T0 + plane_n)[0], (T1 + plane_n)[0]), xlim[0])
        xlim[1] = max(max((T0 - plane_n)[0], (T1 - plane_n)[0]), xlim[1])
        ylim[0] = min(min((T0 + plane_n)[1], (T1 + plane_n)[1]), ylim[0])
        ylim[1] = max(max((T0 - plane_n)[1], (T1 - plane_n)[1]), ylim[1])

        curscene.append(scene.visuals.Markers(
            pos=triangulations,
            parent=view.scene
        ))
        curscene.append(scene.visuals.Markers(
            pos=restimates,
            parent=view.scene,
            face_color='blue'
        ))
        curscene.append(scene.visuals.Markers(
            pos=lestimates,
            parent=view.scene,
            face_color='blue'
        ))
        curscene.append(scene.visuals.Markers(
            pos=-np.array(pair["lback"]) + T0,
            parent=view.scene,
            size=3.5,
            edge_width_rel=0.5,
            edge_color=cam_colors[idx1]
        ))
        curscene.append(scene.visuals.Markers(
            pos=-np.array(pair["rback"]) + T1,
            parent=view.scene,
            size=3.5,
            edge_width_rel=0.5,
            edge_color=cam_colors[idx2]
        ))
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

    normal = np.sum(normals, axis=0)
    normal /= np.linalg.norm(normal)
    mean = np.mean(means, axis=0)
    isect = intersect(plane_abcd, mean, normal)

    if isect is not None:
        print(np.linalg.norm( mean - isect ) * 100, "cm")

    curscene.append(scene.visuals.Arrow(
        pos=(
            plane_pt,
            plane_pt + plane_n / 8
        ),
        parent=view.scene
    ))
    plane = scene.visuals.SurfacePlot(
        *meshgrid(plane_pt, plane_abcd, xlim, ylim),
        parent=view.scene
    )
    plane.attach(vp.visuals.filters.Alpha(0.5))
    curscene.append(plane)

my_app = app.use_app()
timer = app.Timer(connect=draw, app=my_app)
canvas = scene.SceneCanvas(keys='interactive', show=True, app=my_app)
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

@canvas.events.key_press.connect
def on_key_press(event: vp.app.KeyEvent):
    match event.text:
        case 'r':
            timer.stop()
            global draw_idx
            draw_idx = 0
            timer.start(iterations=len(steps))
            print('restart')
        case 's':
            if timer.running:
                timer.stop()
                print('stop')
            else:
                timer.start(timer.interval, timer.max_iterations - timer.iter_count)
                print('start')

if __name__ == '__main__':
    timer.start(interval=0.5, iterations=len(steps))
    my_app.run()
