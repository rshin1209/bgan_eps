import os
import math
import numpy as np
import argparse
import importlib
import torch
import joblib
import mdtraj as md
from torch.utils.data import Dataset, DataLoader
from model import Generator, Discriminator
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool


class BGANEPS(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generator_loss(self, z, x, epoch, epochs):
        x_ = self.g_net(z)
        z_ = self.e_net(x)

        x__ = self.g_net(z_)
        z__ = self.e_net(x_)

        if epoch + 1 > epochs - 30:
            self.dofs = np.vstack((self.dofs, self.reconstruct(x__)))

        dz_ = self.dz_net(z_)
        dx_ = self.dx_net(x_)

        l2_loss_z = self.mse_loss(z, z__)
        l2_loss_x = self.mse_loss(x, x__)

        g_loss_adv = self.mse_loss(torch.ones_like(dz_), dz_)
        e_loss_adv = self.mse_loss(torch.ones_like(dx_), dx_)

        g_loss = l2_loss_x
        e_loss = l2_loss_z
        g_e_loss = g_loss_adv + e_loss_adv + 10.0 * l2_loss_z + 10.0 * l2_loss_x
        return g_loss, e_loss, g_e_loss

    def discriminator_loss(self, z, x):
        x_ = self.g_net(z)
        z_ = self.e_net(x)

        dz = self.dz_net(z)
        dx = self.dx_net(x)

        dz_ = self.dz_net(z_)
        dx_ = self.dx_net(x_)

        dz_loss = (
            self.mse_loss(torch.ones_like(dz), dz)
            + self.mse_loss(torch.zeros_like(dz_), dz_)
        ) / 2.0
        dx_loss = (
            self.mse_loss(torch.ones_like(dx), dx)
            + self.mse_loss(torch.zeros_like(dx_), dx_)
        ) / 2.0

        d_loss = dz_loss + dx_loss
        return dz_loss, dx_loss, d_loss

    def reconstruct(self, X):
        X = X.detach().cpu().numpy()
        X = self.scaler.inverse_transform(X)
        X = X[(X[:, 0] < self.bondmax) & (X[:, 0] > self.bondmin)]
        return X

    def run(self, args, X, dim, scaler, loop):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.g_net = Generator(input_dim=dim, output_dim=dim, nb_units=dim).to(
            self.device
        )
        self.e_net = Generator(input_dim=dim, output_dim=dim, nb_units=dim).to(
            self.device
        )
        self.dz_net = Discriminator(input_dim=dim, nb_units=dim // 4).to(self.device)
        self.dx_net = Discriminator(input_dim=dim, nb_units=dim // 4).to(self.device)
        self.mse_loss = torch.nn.MSELoss()
        self.scaler = scaler
        self.bondmax = args.bondmax
        self.bondmin = args.bondmin

        TRAIN = torch.from_numpy(X).float()
        self.dofs = self.reconstruct(TRAIN)
        train_set = DataLoader(
            TRAIN, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        self.g_e_optimizer = torch.optim.Adam(
            list(self.g_net.parameters()) + list(self.e_net.parameters()),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
        self.d_optimizer = torch.optim.Adam(
            list(self.dz_net.parameters()) + list(self.dx_net.parameters()),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
        for epoch in range(args.epochs):
            for i, x in enumerate(train_set):
                z = torch.randn_like(x) / 2.0
                z, x = z.to(self.device), x.to(self.device)

                # Generator / Encoder Training
                self.g_e_optimizer.zero_grad()
                g_loss, e_loss, g_e_loss = self.generator_loss(z, x, epoch, args.epochs)
                g_e_loss.backward()
                self.g_e_optimizer.step()

                self.d_optimizer.zero_grad()
                dz_loss, dx_loss, d_loss = self.discriminator_loss(z, x)
                d_loss.backward()
                self.d_optimizer.step()
            print(
                "Epoch [%d] g_loss [%.4f] e_loss [%.4f] g_e_loss [%.4f] dz_loss [%.4f] dx_loss [%.4f] d_loss [%.4f]"
                % (epoch + 1, g_loss, e_loss, g_e_loss, dz_loss, dx_loss, d_loss)
            )

        self.dofs = self.dofs.T
        self.EPS(args, loop)

    def get_jacobian(self, X_bin_edge, xjtype, Y_bin_edge, yjtype):
        if Y_bin_edge is None and yjtype is None:
            if xjtype == "bond":
                return X_bin_edge**2
            elif xjtype == "torsion":
                return 1
            else:
                return math.sin(X_bin_edge)
        else:
            return_dict = {
                ("bond", "bond"): X_bin_edge**2 * Y_bin_edge**2,
                ("bond", "angle"): X_bin_edge**2 * math.sin(Y_bin_edge),
                ("bond", "torsion"): X_bin_edge**2,
                ("angle", "angle"): math.sin(X_bin_edge) * math.sin(Y_bin_edge),
                ("angle", "torsion"): math.sin(X_bin_edge),
                ("torsion", "bond"): Y_bin_edge**2,
                ("torsion", "angle"): math.sin(Y_bin_edge),
            }
            return return_dict.get((xjtype, yjtype), 1)

    def entropy_1D(self, args):
        num, jtype = args
        counts, bin_edges = np.histogram(self.dofs[num][self.indices], 50)
        sample_size = np.sum(counts)
        prob_den = counts / sample_size
        dx = bin_edges[1] - bin_edges[0]
        bin_edges += dx / 2

        entropy_sum = -np.sum(
            [
                prob_den[i]
                * math.log(
                    prob_den[i]
                    / (self.get_jacobian(bin_edges[i], jtype, None, None) * dx)
                )
                for i in range(50)
                if prob_den[i] != 0
            ]
        )

        return entropy_sum + (np.count_nonzero(prob_den) - 1) / (2 * sample_size)

    def entropy_2D(self, args):
        xnum, ynum, xjtype, yjtype = args
        H, X_bin_edges, Y_bin_edges = np.histogram2d(
            self.dofs[xnum][self.indices], self.dofs[ynum][self.indices], 50
        )
        dx, dy = X_bin_edges[1] - X_bin_edges[0], Y_bin_edges[1] - Y_bin_edges[0]
        X_bin_edges += dx / 2
        Y_bin_edges += dy / 2
        sample_size = np.sum(H)
        H = H / sample_size

        entropy_sum = -np.sum(
            [
                H[row][col]
                * math.log(
                    H[row][col]
                    / (
                        self.get_jacobian(
                            X_bin_edges[row], xjtype, Y_bin_edges[col], yjtype
                        )
                        * dx
                        * dy
                    )
                )
                for row in range(50)
                for col in range(50)
                if H[row][col] != 0
            ]
        )

        return entropy_sum + (np.count_nonzero(H) - 1) / (2 * sample_size)

    def get_sorted_indices(self, entropy_list, num_indices):
        return sorted(range(len(entropy_list)), key=lambda i: entropy_list[i])[
            -num_indices:
        ]

    def EPS(self, args, loop):
        s_profile, sample_size = [], []
        conversion_factor = -1.987204259e-3 * args.temperature  # kcal/mol
        ensemble = args.ensemble + 1
        x = np.linspace(args.bondmax, args.bondmin, ensemble)
        dx = x[0] - x[1]
        runavg = np.linspace(x[0], x[-2], 30)
        bat = {2: "bond", 3: "angle", 4: "torsion"}
        eps_output_dir = f"./{self.output_dir}/bgan{loop+1}"
        if not os.path.exists(eps_output_dir):
            os.makedirs(eps_output_dir)

        dof_list = [
            line.split()
            for line in open(f"./{self.output_dir}/topology.txt", "r").readlines()
            if len(line.split()) != 1
        ]
        atom_num = eval(open(f"./{self.output_dir}/topology.txt", "r").readlines()[0])
        bond_num = len([items for items in dof_list if len(items) == 2])
        angle_num = len([items for items in dof_list if len(items) == 3])
        torsion_num = len([items for items in dof_list if len(items) == 4])

        react = self.dofs[0]
        self.dofs = self.dofs[1:]

        if args.eps_type == "max":
            bond_entropy, angle_entropy, torsion_entropy = [], [], []
            for i in range(len(dofs)):
                label = bat[len(dof_list[i])]
                entropy = entropy_1D([dofs[i], label])
                if label == "bond":
                    bond_entropy.append(entropy)
                elif label == "angle":
                    angle_entropy.append(entropy)
                else:
                    torsion_entropy.append(entropy)
            bond_mea_indices = self.get_sorted_indices(bond_entropy, atom_num - 1)
            angle_mea_indices = self.get_sorted_indices(angle_entropy, atom_num - 2)
            torsion_mea_indices = self.get_sorted_indices(torsion_entropy, atom_num - 3)
            angle_mea_indices = [x + bond_num for x in angle_mea_indices]
            torsion_mea_indices = [
                x + bond_num + angle_num for x in torsion_mea_indices
            ]
            mea_indices = sorted(
                bond_mea_indices + angle_mea_indices + torsion_mea_indices
            )
            dofs = dofs[mea_indices]
            dof_list = [dof_list[i] for i in mea_indices]

        # for e in range(len(x) - 1):
        #    eps_file_path = f'{eps_output_dir}/eps_{x[e]:.3f}_{(x[e+1]):.3f}.log'
        #    with open(eps_file_path, "w") as f_eps:
        #        self.indices = np.argwhere((react >= (x[e+1])) & (react < x[e])).flatten()
        for e in range(30):
            eps_file_path = (
                f"{eps_output_dir}/eps_{runavg[e]:.3f}_{(runavg[e]-dx):.3f}.log"
            )
            with open(eps_file_path, "w") as f_eps:
                self.indices = np.argwhere(
                    (react > (runavg[e] - dx)) & (react < runavg[e])
                ).flatten()
                sample_size.append(len(self.indices))
                # with Pool() as p:
                #    entropy1d = p.map(self.entropy_1D, [(i, bat[len(dof_list[i])]) for i in range(len(dof_list))])
                #    entropy2d = p.map(self.entropy_2D, [(i, j, bat[len(dof_list[i])], bat[len(dof_list[j])]) for i in range(len(dof_list) - 1) for j in range(i + 1, len(dof_list))])
                entropy1d = [
                    self.entropy_1D((i, bat[len(dof_list[i])]))
                    for i in range(len(dof_list))
                ]
                entropy2d = [
                    self.entropy_2D(
                        (i, j, bat[len(dof_list[i])], bat[len(dof_list[j])])
                    )
                    for i in range(len(dof_list) - 1)
                    for j in range(i + 1, len(dof_list))
                ]
                for i in range(len(dof_list)):
                    f_eps.write(
                        f'{bat[len(dof_list[i])]} {" ".join(str(x) for x in dof_list[i])}\t{entropy1d[i]}\n'
                    )

                MI_list, MIST_list = [], []
                index = 0
                MI = 0.0
                for i in range(len(dof_list) - 1):
                    dummy = []
                    for j in range(i + 1, len(dof_list)):
                        dummy.append(entropy1d[i] + entropy1d[j] - entropy2d[index])
                        f_eps.write(
                            "%s %s\t%s %s\t\t%f\t%f\n"
                            % (
                                bat[len(dof_list[i])],
                                " ".join(str(x) for x in dof_list[i]),
                                bat[len(dof_list[j])],
                                " ".join(str(x) for x in dof_list[j]),
                                entropy2d[index],
                                dummy[-1],
                            )
                        )
                        index += 1
                    MI_list.append(dummy)
                    MIST_list.append(max(dummy))
                    MI += sum(dummy)

                if args.eps_type == "max":
                    f_eps.write("Total MIE: %f\n" % MI)
                    f_eps.write("Total MIST: %f\n" % sum(MIST_list))
                    f_eps.write("MIE Entropy: %f\n" % (sum(entropy1d) - MI))
                    f_eps.write(
                        "MIST Entropy: %f\n" % (sum(entropy1d) - sum(MIST_list))
                    )
                    f_eps.write(
                        "MIE Entropy (kcal/mol): %f\n"
                        % ((sum(entropy1d) - MI) * conversion_factor)
                    )
                    f_eps.write(
                        "MIST Entropy (kcal/mol): %f\n"
                        % ((sum(entropy1d) - sum(MIST_list)) * conversion_factor)
                    )
                    s_profile.append(
                        (sum(entropy1d) - sum(MIST_list)) * conversion_factor
                    )

                else:
                    bond_entropy = sum(entropy1d[:bond_num]) * (atom_num - 1) / bond_num
                    angle_entropy = (
                        sum(entropy1d[bond_num : bond_num + angle_num])
                        * (atom_num - 2)
                        / angle_num
                    )
                    torsion_entropy = (
                        sum(entropy1d[bond_num + angle_num :])
                        * (atom_num - 3)
                        / torsion_num
                    )
                    bond_mist = (
                        sum(MIST_list[:bond_num]) * (atom_num - 2) / (bond_num - 1)
                    )
                    angle_mist = (
                        sum(MIST_list[bond_num : bond_num + angle_num])
                        * (atom_num - 3)
                        / (angle_num - 1)
                    )
                    torsion_mist = (
                        sum(MIST_list[bond_num + angle_num :])
                        * (atom_num - 4)
                        / (torsion_num - 1)
                    )
                    total = (
                        bond_entropy
                        + angle_entropy
                        + torsion_entropy
                        - bond_mist
                        - angle_mist
                        - torsion_mist
                    )
                    f_eps.write("Bond Entropy: %f\n" % bond_entropy)
                    f_eps.write("Angle Entropy: %f\n" % angle_entropy)
                    f_eps.write("Torsion Entropy: %f\n" % torsion_entropy)
                    f_eps.write("MIST: %f\n" % (bond_mist + angle_mist + torsion_mist))
                    f_eps.write(
                        "MIST Entropy (kcal/mol): %f\n" % (conversion_factor * total)
                    )
                    s_profile.append(conversion_factor * total)

        np.save(os.path.join(eps_output_dir, "eps.npy"), np.array(s_profile))
        with open(os.path.join(eps_output_dir, "eps.log"), "w") as fw:
            fw.write(
                "Ensemble No. \t Bond Length Range \t Entropy(kcal/mol) \t Snapshots\n"
            )
            for i in range(len(s_profile)):
                # fw.write(f'{x[i]:.4f} \t {x[i+1]:.4f} \t {s_profile[i]:.4f} \t {(s_profile[i] - s_profile[0]):.4f} \t {sample_size[i]}\n')
                fw.write(
                    f"{i+1} \t {runavg[i]:.4f} \t {(runavg[i]-dx):.4f} \t {s_profile[i]:.4f} \t {(s_profile[i] - s_profile[0]):.4f} \t {sample_size[i]}\n"
                )
