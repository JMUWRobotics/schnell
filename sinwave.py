import numpy as np
from vispy import scene, app
from vispy.geometry import MeshData
from scipy.optimize import least_squares

# === Define the sinusoidal surface parameters ===

# origin = np.array([0.0, 0.0, 1.0])
# u = np.array([1.0, 0.0, 0.0])
# v = np.array([0.0, 1.0, 0.0])
# normal = np.array([0.0, 0.0, 1.0])
# amplitude = 0.5
# frequency = 5
# phase = np.pi / 2

plane_param = np.array([0.1522135796926206, 0.07386223835536908, 0.2721447693021134, -0.8779893174123757, 0.11769968478070637, 0.46398442076461244, -0.007492214882922366, 0.9658229358896685, -0.25909442916745534, 0.47911459271736, 0.23195390399035876, 0.8465498174761541, 0, 0, 0])
xy_guess = np.array([0.17852059308677284, 0.-0.6040395325557387]) 


origin = np.array(plane_param[:3])
u = np.array(plane_param[3:6])
v = np.array(plane_param[6:9])
normal = np.array(plane_param[9:12])
amplitude = plane_param[12]
frequency = plane_param[13]
phase = plane_param[14]

line_pt = np.array([0,0,1])
line_vec = np.array([0,1,1])

# === Function to evaluate the surface at (x, y) ===
def evaluate(x, y):
    base = origin + x * u + y * v
    wave = amplitude * np.sin(frequency * x + phase)
    return base + wave * normal

def intersect_line_with_surface(initial_guess=(0.0, 0.0)):

    global line_pt, line_vec

    line_pt = np.array(line_pt)
    line_vec = np.array(line_vec)
    line_vec = line_vec / np.linalg.norm(line_vec)  # normalize

    def residual(xy):
        x, y = xy
        surface_point = evaluate(x, y)
        o = line_pt
        d = line_vec

        # Project surface point onto line
        t = np.dot(surface_point - o, d)
        closest_point = o + t * d

        diff = surface_point - closest_point

        # print(f"Surface point: {surface_point}, Closest point: {closest_point}, Diff: {diff}")

        return diff  # 3 residuals: (dx, dy, dz)

    result = least_squares(residual, initial_guess)

    if result.success:
        x_opt, y_opt = result.x
        intersection_point = evaluate(x_opt, y_opt)
        return intersection_point
    else:
        raise RuntimeError("Intersection optimization failed!")

# === Generate a grid of points ===
res = 100
x_vals = np.linspace(-1, 1, res)
y_vals = np.linspace(-1, 1, res)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

# Compute 3D surface points
points = np.zeros((res, res, 3), dtype=np.float32)
for i in range(res):
    for j in range(res):
        points[i, j] = evaluate(X[i, j], Y[i, j])

# Flatten the grid for VisPy
vertices = points.reshape(-1, 3)

# Build face indices
faces = []
for i in range(res - 1):
    for j in range(res - 1):
        idx = i * res + j
        faces.append([idx, idx + 1, idx + res])
        faces.append([idx + 1, idx + res + 1, idx + res])
faces = np.array(faces)

# === Setup VisPy canvas ===
canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

isec = np.array(intersect_line_with_surface())
print("isec = ", isec)
marker = scene.visuals.Markers(parent=view.scene)
marker.set_data(
    pos=np.array([isec]),  # Marker position
    face_color='orange',
    size=10
)

# Your line data (two points stacked into shape (2, 3))
dist = np.vstack([line_pt, line_pt + line_vec])
# Create and configure the line
line = scene.visuals.Line(
    pos=dist,
    color='red',
    width=3,
    parent=view.scene,
    # method='gl'  # Optional: 'gl' for smooth GPU rendering
)

# print("guess: ", evaluate(xy_guess[0], xy_guess[1]), "diff =", evaluate(xy_guess[0], xy_guess[1]) - isec)
# marker = scene.visuals.Markers(parent=view.scene)
# marker.set_data(
#     pos=np.array([evaluate(xy_guess[0], xy_guess[1])]),  # Marker position
#     face_color='green',
#     size=10
# )

# print("guess: ", evaluate(xy_guess[0], xy_guess[1]), "diff =", evaluate(xy_guess[0], xy_guess[1]) - isec)
marker = scene.visuals.Markers(parent=view.scene)
marker.set_data(
    pos=np.array([[0, 1, 0.013050340134658]]),  # Marker position
    face_color='green',
    size=10
)

# === Create mesh visual ===
mesh = scene.visuals.Mesh(vertices=vertices, faces=faces, color=(0.5, 0.7, 1, 1), shading='smooth')
view.add(mesh)

# Add axis for orientation
axis = scene.visuals.XYZAxis(parent=view.scene)

# Run the app
app.run()
