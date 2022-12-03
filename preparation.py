import os
import math
import networkx as nx
import numpy as np
import argparse
from pymol import cmd

def get_con(xyzfile, atom1, atom2, atom3, atom4):
    filename = xyzfile[:-3]

    pymol_argv = ['pymol','-qc']
    cmd.load('../dataset/'+filename+'xyz')
    cmd.save('./temporary/'+filename+'pdb', state = 1)

    lst = []
    with open('./temporary/'+filename+'pdb', 'r') as fr:
        lines_list = fr.readlines()
        for lines in lines_list:
            if lines.startswith('CONECT'):
                lst.append(lines)

    graph = dict()
    for items in lst:
        atom = items.split()
        atom.pop(0)
        if int(atom[0]) not in graph:
            graph[int(atom[0])] = []
        for i in range(1,len(atom)):
            graph[int(atom[0])].append(int(atom[i]))

    if atom3 != 0 and atom4 != 0:
        graph[atom3].append(atom4)
        graph[atom4].append(atom3)

    atom_num = len(graph)
    paths, total_paths, unique = [], [], []

    G = nx.Graph(graph)
    for i in range(1,atom_num+1):
        for j in range(1,atom_num+1):
            for path in nx.all_simple_paths(G, source=i, target=j):
                if len(path) == 2 or len(path) == 3 or len(path) == 4:
                    total_paths.append(path)
    for items in total_paths:
        if items not in unique and list(reversed(items)) not in unique:
            unique.append(items)
    unique.sort(key=len)
    f = open('./temporary/topology.txt', 'w')
    f.write(str(atom_num) + '\n')
    for t in unique:
        f.write(' '.join(str(s) for s in t) + '\n')
    f = open('./temporary/topology_%d_%d.txt'%(atom1,atom2), 'w')
    f.write(str(atom_num) + '\n')
    f.write(str(atom1) + ' ' + str(atom2) + '\n')
    for t in unique:
        f.write(' '.join(str(s) for s in t) + '\n')

class Topology:
    def __init__(self, top_input_file):
        self.topology = []
        with open(top_input_file, 'r') as f:
            for line in f:
                if len(line.split()) == 2:
                    atom1, atom2 = line.split()
                    self.topology.append([eval(atom1),eval(atom2)])
                if len(line.split()) == 3:
                    atom1, atom2, atom3 = line.split()
                    self.topology.append([eval(atom1),eval(atom2),eval(atom3)])
                if len(line.split()) == 4:
                    atom1, atom2, atom3, atom4 = line.split()
                    self.topology.append([eval(atom1),eval(atom2),eval(atom3),eval(atom4)])

def xyz2bat(xyzfile, atom1, atom2):
    top = Topology('./temporary/topology_%d_%d.txt'%(atom1,atom2))
    pymol_argv = [ 'pymol', '-qc']
    cmd.load('../dataset/'+xyzfile)
    snapshots = []
    for num in range(1,cmd.count_states(selection='all')+1):
        snapshot = []
        for items in top.topology:
            if len(items) == 2:
                snapshot.append(cmd.get_distance(atom1="resi %d"%items[0],atom2="resi %d"%items[1],state=num))
            if len(items) == 3:
                snapshot.append(math.radians(cmd.get_angle(atom1="resi %d"%items[0],atom2="resi %d"%items[1],atom3="resi %d"%items[2],state=num)))
            if len(items) == 4:
                snapshot.append(math.radians(cmd.get_dihedral(atom1="resi %d"%items[0],atom2="resi %d"%items[1],atom3="resi %d"%items[2],atom4="resi %d"%items[3],state=num)))
        snapshots.append(snapshot)
    np.save('./temporary/'+xyzfile[:-3]+'npy',np.array(snapshots))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--filename', type=str, default='data.xyz', help='name of the input file in ../dataset')
    parser.add_argument('--atom1', type=int, default=1, help='main reacting bond atom 1')
    parser.add_argument('--atom2', type=int, default=2, help='main reacting bond atom 2')
    parser.add_argument('--atom3', type=int, default=0, help='1st reacting bond atom 3')
    parser.add_argument('--atom4', type=int, default=0, help='1st reacting bond atom 4')

    args = parser.parse_args()
    xyz_file = args.filename
    atom1 = args.atom1
    atom2 = args.atom2
    atom3 = args.atom3
    atom4 = args.atom4

    get_con(xyz_file, atom1, atom2, atom3, atom4)


    xyz2bat(xyz_file, atom1, atom2)
