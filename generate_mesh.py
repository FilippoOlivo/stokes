import gmsh

def generate_geo_file(r, cx, cy, template, output_file):
    with open(template, "r") as f:
        content = f.read()
    content = content.replace("[RADIUS]", f"{r:.4f}")
    content = content.replace("[CENTER_X]", f"{cx:.4f}")
    content = content.replace("[CENTER_Y]", f"{cy:.4f}")
    with open(output_file, "w") as f:
        f.write(content)

cy = [.5]
cx = [1.5]
r = [0.1]

template = "mesh_template.geo"
for i, (x, y, radius) in enumerate(zip(cx, cy, r)):
    output_file = f"mesh_{i}.geo"
    generate_geo_file(radius, x, y, template, output_file)
    gmsh.initialize()
    gmsh.open(output_file)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_file.replace(".geo", ".msh"))