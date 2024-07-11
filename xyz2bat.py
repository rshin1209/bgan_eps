import os
import numpy as np
import networkx as nx
import subprocess
import argparse
import mdtraj as md


def get_connectivity(X, nb1, nb2):
    graph_dict = {}
    for atom_pair in X:
        a1, a2 = atom_pair
        index1 = a1.index
        index2 = a2.index

        # Add the indices to each other's neighbor list
        if index1 not in graph_dict:
            graph_dict[index1] = []
        if index2 not in graph_dict:
            graph_dict[index2] = []

        graph_dict[index1].append(index2)
        graph_dict[index2].append(index1)
    graph_dict[nb1 - 1].append(nb2 - 1)
    graph_dict[nb2 - 1].append(nb1 - 1)
    return graph_dict


def get_topology(graph, output_dir):
    G = nx.Graph(graph)
    atom_num = len(graph)
    total_paths = []
    for i in range(atom_num):
        for j in range(atom_num):
            for path in nx.all_simple_paths(G, source=i, target=j):
                if len(path) in [2, 3, 4]:
                    total_paths.append(path)

    unique = []
    for items in total_paths:
        if tuple(items) not in unique and tuple(reversed(items)) not in unique:
            unique.append(tuple(items))
    unique.sort()
    unique.sort(key=len)

    with open(f"./{output_dir}/topology.txt", "w") as fw:
        fw.write(str(atom_num) + "\n")
        for t in unique:
            fw.write(" ".join(str(s + 1) for s in t) + "\n")

    return atom_num, unique


def xyz2bat(output_dir, args):
    traj = f"./dataset/{args.reaction}.xyz"
    command = f"obabel {traj} -O ./{output_dir}/{args.reaction}.pdb"
    subprocess.run(command, shell=True)
    trajectory = md.load(f"./{output_dir}/{args.reaction}.pdb")

    reference = md.load(f"./dataset/{args.ts}.pdb")
    graph = get_connectivity(reference.top.bonds, args.nb1, args.nb2)
    atom_num, unique = get_topology(graph, output_dir)
    unique.insert(0, (args.atom1 - 1, args.atom2 - 1))

    bond_list = [items for items in unique if len(items) == 2]
    angle_list = [items for items in unique if len(items) == 3]
    torsion_list = [items for items in unique if len(items) == 4]

    b = md.compute_distances(trajectory, bond_list) * 10.0  # nm2ang
    a = md.compute_angles(trajectory, angle_list)
    t = md.compute_dihedrals(trajectory, torsion_list)

    snapshot = np.hstack((b, a, t))
    os.remove(f"./{output_dir}/{args.reaction}.pdb")
    np.save(f"./{output_dir}/dof.npy", snapshot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--atom1", type=int, default=0, help="first atom number in reaction coordinate (e.g., bond 2 or bond 3)")
    parser.add_argument("--atom2", type=int, default=0, help="second atom number in reaction coordinate (e.g., bond 2 or bond 3)")
    parser.add_argument("--reaction", type=str, default="", help="Name of the reaction file without format tag")
    parser.add_argument("--ts", type=str, default="", help="Name of the optimized transition state structure file (pdb) without format tag")
    parser.add_argument("--nb1", type=int, default=0, help="first atom number in bond 1")
    parser.add_argument("--nb2", type=int, default=0, help="second atom number in bond 1")

    args = parser.parse_args()
    if args.atom1 == None or args.atom2 == None:
        print("Error: Atom numbers of reacting bond are not provided.")
        sys.exit()
    if args.reaction == "":
        print("Error: Reaction Name is not provided.")
        sys.exit()

    output_dir = "./log/%s" % args.reaction
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xyz2bat(output_dir, args)
