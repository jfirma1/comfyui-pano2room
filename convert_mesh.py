import argparse
import trimesh
import numpy as np


def convert(obj_path: str, out_path: str, scale: float = 100.0):
    print(f"Loading: {obj_path}")
    mesh = trimesh.load(obj_path, process=False)
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    rotation = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh.apply_transform(rotation)
    mesh.apply_scale(scale)

    mesh.export(out_path)
    print(f"Exported: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a proxy mesh OBJ to FBX (or other format).")
    parser.add_argument("input",  help="Path to the input .obj file")
    parser.add_argument("output", help="Path for the output file (e.g. proxy_mesh.fbx)")
    parser.add_argument("--scale", type=float, default=100.0,
                        help="Uniform scale factor applied after rotation (default: 100)")
    args = parser.parse_args()

    convert(args.input, args.output, args.scale)
