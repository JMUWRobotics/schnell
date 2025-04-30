#!/usr/bin/env python3

import numpy as np
import vispy as vp
from vispy import app, scene
import subprocess, json
import numpy as np
import os
import sys
import math

from scipy.optimize import least_squares

os.environ['QT_QPA_PLATFORM'] = 'xcb'

datadir = os.environ["HOME"] + '/Documents/schnell/Daten/wasserkiste/foureyes/april'

def parser():

    sin = False

    if len(sys.argv) < 2:
        
        sin = False

    else:
       
       # get the arguments to the wrigth variabales
        for elements in sys.argv[1:]:

            print(elements)

            if(elements == "--sin" ):
                sin = True

    return sin

SIN = parser()

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

output_path ="output.json"
with open(output_path, "w") as f:
    f.write(jsondump)
    
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

def circgrid(pt, abcd, scale = 1):
    randomv = np.random.rand(3)
    n = abcd[:-1]
    print(randomv, n)
    v = np.cross(n, randomv)
    v /= np.linalg.norm(v)

    xs, ys, zs = [], [], []
    
    for i in np.linspace(0, np.pi, 34):
        rv = v * np.cos(i) + np.cross(n, v) * np.sin(i) + n * ( n.T @ v ) * (1 - np.cos(i))
        rv *= scale
        x, y, z = pt + rv
        xs.append(x)
        ys.append(y)
        zs.append(z)
        print(x, y, z)
    
    return np.array(xs), np.array(ys), np.array(zs)

# https://math.stackexchange.com/a/897677
def vector_align(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    d = np.dot(a, b)
    c = np.linalg.norm(np.cross(a, b))
    G = np.array([
        [d, -c, 0],
        [c,  d, 0],
        [0,  0, 1]
    ])
    u = np.dot(a, b)*a
    v = b - u
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    F_1 = np.hstack((u.reshape(3, 1), v.reshape(3, 1), w.reshape(3, 1)))
    return F_1 @ G @ np.linalg.inv(F_1)

def intersect(abcd, p, n):
    n = n / np.linalg.norm(n)
    ax, ay, az = p + n
    bx, by, bz = p
    a = np.append(p + n, 1).reshape(-1, 1)
    b = np.append(p, 1).reshape(-1, 1)
    
    # pluecker = a @ b.T - b @ a.T
    # x, y, z, w = pluecker.T @ abcd

    assert a.shape == b.shape == (4,1), f"{a.shape} {b.shape} not (4,1)"

    pluecker = a @ b.T - b @ a.T
    isect = pluecker.T @ abcd

    # if np.isclose(w, 0):
    #     return None

    # return np.array([x / w, y / w, z / w])
    return isect[:-1] / isect[-1]

drawn_cams = set()
cam_colors = [
    'k', 'm', 'y', 'c'
]

draw_idx = 0
curscene = []

# to get the surface of the sin wave
def evaluate(x, y, plane_param):

    origin = np.array(plane_param[:3])
    u = np.array(plane_param[3:6])
    v = np.array(plane_param[6:9])
    normal = np.array(plane_param[9:12])
    amplitude = plane_param[12]
    frequency = plane_param[13]
    phase = plane_param[14]

    base = origin + x * u + y * v
    wave = amplitude * np.sin(frequency * x + phase)
    return base + wave * normal

def intersect_line_with_surface(plane_param, line_origin, line_dir, initial_guess=(0.0, 0.0)):
    line_origin = np.array(line_origin)
    line_dir = np.array(line_dir)
    line_dir = line_dir / np.linalg.norm(line_dir)  # normalize

    def residual(xy):
        x, y = xy
        surface_point = evaluate(x, y, plane_param)
        o = line_origin
        d = line_dir

        # Project surface point onto line
        t = np.dot(surface_point - o, d)
        closest_point = o + t * d

        diff = surface_point - closest_point
        return diff  # 3 residuals: (dx, dy, dz)

    result = least_squares(residual, initial_guess)

    if result.success:
        x_opt, y_opt = result.x
        intersection_point = evaluate(x_opt, y_opt, plane_param)
        return intersection_point
    else:
        raise RuntimeError("Intersection optimization failed!")

def draw(_):

    global draw_idx
    global curscene
    j = steps[draw_idx]

    if SIN:

        plane_pt   = np.array(j["plane"][:3])
        plane_param = np.array(j["plane"])
        plane_n = np.array(plane_param[9:12])

        print("plane_param", plane_param)

    else:
        plane_pt   = np.array(j["plane"]["pt"])
        plane_abcd = np.array(j["plane"]["abcd"])
        plane_n = plane_abcd[:-1]


    print(draw_idx)

    draw_idx += 1

    xlim, ylim = np.array([np.inf, -np.inf]), np.array([np.inf, -np.inf])

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

        normals.append(lnormal)
        normals.append(rnormal)
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

        filtered = [vec for vec in pair["lback"] if all((x is not None) and (not math.isnan(x)) for x in vec)]
        curscene.append(scene.visuals.Markers(
            pos= -np.array(filtered) + T0,
            parent=view.scene,
            size=3.5,
            edge_width_rel=0.5,
            edge_color=cam_colors[idx1]
        ))

        filtered = [vec for vec in pair["rback"] if all((x is not None) and (not math.isnan(x)) for x in vec)]
        curscene.append(scene.visuals.Markers(
            pos = -np.array(filtered) + T1,
            # pos= -np.array(pair["rback"]) + T1,
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

    # turn the normal vector arround if in the other direction -> check if this allways works
    norm_nomals= []
    for n in normals:
        n /= np.linalg.norm(n)
        if n[2] > 0:
            n *= -1
        norm_nomals.append(n)
    normal = np.sum(norm_nomals, axis=0)
    normal /= np.linalg.norm(normal)
    mean = np.mean(means, axis=0)

    isect = None

    if SIN:
        isect = intersect_line_with_surface(plane_param, mean, normal)
    else:    
        isect = intersect(plane_abcd, mean, normal)

    # black camera is np.zero(3)
    # print("normal", normal)
    # print("guess plane", plane_abcd)
    # print("mean", mean)

    if isect is not None:
        print("vec", (mean.reshape(1, -1) - isect.reshape(1, -1))/ np.linalg.norm((mean.reshape(1, -1) - isect.reshape(1, -1))), "water height", np.linalg.norm(mean.reshape(1, -1) - isect.reshape(1, -1)), "m")
    
    for n in norm_nomals:
        dist = np.vstack([mean.reshape(1, -1), mean.reshape(1, -1) + n.reshape(1, -1)])
        curscene.append(scene.visuals.Line(
            pos=dist,
            color='green',
            width=3,
            # method='gl',  # GPU-accelerated line rendering
            parent=view.scene
        ))
    dist = np.vstack([mean.reshape(1, -1), isect.reshape(1, -1)])
    curscene.append(scene.visuals.Line(
        pos=dist,
        color='red',
        width=3,
        # method='gl',  # GPU-accelerated line rendering
        parent=view.scene
    ))
    curscene.append(scene.visuals.Markers(
        pos=isect.reshape(1, -1),
        parent=view.scene,
        face_color='red'
    ))
    curscene.append(scene.visuals.Markers(
        pos=mean.reshape(1, -1),
        parent=view.scene,
        face_color='orange'
    ))
    curscene.append(scene.visuals.Arrow(
        pos=(
            plane_pt,
            plane_pt + plane_n / 8
        ),
        parent=view.scene
    ))
    # xx, yy, zz = circgrid(plane_pt, plane_abcd)
    # plane = scene.visuals.SurfacePlot(
    #     x=xx, y=yy, z=zz,
    #     parent=view.scene
    # )

    plane = None

    if SIN:

        res = 100
        x_vals = np.linspace(-1, 1, res)
        y_vals = np.linspace(-1, 1, res)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)

        # Compute 3D surface points
        points = np.zeros((res, res, 3), dtype=np.float32)
        for i in range(res):
            for k in range(res):
                points[i, k] = evaluate(X[i, k], Y[i, k], plane_param)

        # Flatten the grid for VisPy
        vertices = points.reshape(-1, 3)

        # Build face indices
        faces = []
        for i in range(res - 1):
            for k in range(res - 1):
                idx = i * res + k
                faces.append([idx, idx + 1, idx + res])
                faces.append([idx + 1, idx + res + 1, idx + res])
        faces = np.array(faces)

        plane = scene.visuals.Mesh(vertices=vertices, faces=faces, color=(0.5, 0.7, 1, 1), shading='smooth', parent=view.scene)

    else:
        plane = scene.visuals.Plane(
            direction='+x',
            parent=view.scene
        )
        plane.transform = vp.scene.ChainTransform(
            vp.scene.STTransform(translate=plane_pt),
            vp.scene.MatrixTransform(np.vstack((np.hstack((vector_align([1, 0, 0], plane_n), np.zeros((3,1)))), np.array([0, 0, 0, 1]))))
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
