# Bidirectional Generative Adversarial Network - Entropic Path Sampling

The repository documents how to perform bidirectional generative adversarial network - entropic path sampling ([BGAN-EPS](10.26434/chemrxiv-2022-lcfbq)) method to accelerate entropic path sampling ([EPS](https://doi.org/10.1021/acs.jpclett.1c03116)) by integrating EPS with deep generative models.

![Overview](https://user-images.githubusercontent.com/25111091/205412819-def87e10-1ced-44b5-a279-a2c26eb8b133.jpg)

## Bidirectional Generative Adversarial Network (BGAN)

![BGAN](https://user-images.githubusercontent.com/25111091/205412357-c7548b3e-6161-42f6-9c06-3f204374ae7f.jpg)

The bidirectional generative adversarial network (BGAN) model is designed to enhance the estimation of probability density function of molecular configurations. The BGAN model consists of two pairs of generative adversarial networks (GANs): one is used to generate pseudo-molecular coordinates and the other to generate pseudo-latent variables.

## Module Requirements
- Numpy
- Argparse
- Importlib
- Pytorch
- Sklearn
- Scipy
- Networkx
- Pymol
- GPU Access

## Step 1: Prepare post-transition-state (post-TS) trajectories and place a single combined file (all post-TS trajectories) in the folder named "dataset" (bond formation cutoff: 1.6 Å for the C-C bond formation).

## Step 2: Prepare topology file and convert trajectories from the Cartesian coordinate to the internal coordinate by running the command below.
`python preparation.py --filename ./dataset/ngnd_64_adduct_postTS.xyz --atom1 5 --atom2 14 --atom3 8 --atom4 9`
`python preparation.py --filename ./dataset/ngnd_64_adduct_postTS.xyz --atom1 7 --atom2 12 --atom3 8 --atom4 9`

The topology file is prepared by representing Cartesian coordinate of reactive species in the graph structure based on the bonding atoms. The connectivity script computes all possible bond, angle, and torsion angle via path finding algorithm and outputs redundant internal coordinates (more than 3N-6) as the connectivity file. Additionally, the user must define the main reacting bond and the first reacting bond. atom1 and atom2 are the atoms involved in the main reacting bond and atom3 and atom4 are the atoms involved in the first reacting bond. If the reaction involves a single bond formation, atom3 and atom4 can be ignored.

## Step 3: Run bgan_eps.py to evaluate the entropic profiles by running the command below.

`python bgan_eps.py --filename ./temporary/ngnd_64_adduct_postTS.npy --epochs 200 --ensemble 9`

## BGAN-EPS
`Epoch [185] Time [423.1538] g_loss [3.4851] h_loss [3.6374] g_h_loss [3.9927] dx_loss [0.1359] dy_loss [0.2211] d_loss [0.3570]
Epoch [186] Time [424.9112] g_loss [3.4672] h_loss [3.5512] g_h_loss [3.9423] dx_loss [0.1767] dy_loss [0.1973] d_loss [0.3740]
Epoch [187] Time [426.6559] g_loss [3.4324] h_loss [3.5499] g_h_loss [3.9113] dx_loss [0.1711] dy_loss [0.2036] d_loss [0.3747]
Epoch [188] Time [428.4182] g_loss [3.4060] h_loss [3.5051] g_h_loss [3.8595] dx_loss [0.2042] dy_loss [0.1813] d_loss [0.3855]
Epoch [189] Time [430.1732] g_loss [3.4479] h_loss [3.5435] g_h_loss [3.9227] dx_loss [0.1751] dy_loss [0.1746] d_loss [0.3497]
Epoch [190] Time [431.9690] g_loss [3.4166] h_loss [3.5404] g_h_loss [3.9172] dx_loss [0.1434] dy_loss [0.1806] d_loss [0.3240]
Epoch [191] Time [433.7298] g_loss [3.4237] h_loss [3.5307] g_h_loss [3.8826] dx_loss [0.1672] dy_loss [0.1950] d_loss [0.3623]
Epoch [192] Time [435.4906] g_loss [3.4092] h_loss [3.4609] g_h_loss [3.8532] dx_loss [0.1573] dy_loss [0.1811] d_loss [0.3384]
Epoch [193] Time [437.2610] g_loss [3.4128] h_loss [3.5358] g_h_loss [3.9005] dx_loss [0.1424] dy_loss [0.2074] d_loss [0.3498]
Epoch [194] Time [439.0479] g_loss [3.4679] h_loss [3.5331] g_h_loss [3.9367] dx_loss [0.1613] dy_loss [0.1763] d_loss [0.3377]
Epoch [195] Time [440.7975] g_loss [3.5147] h_loss [3.5943] g_h_loss [3.9927] dx_loss [0.1718] dy_loss [0.1807] d_loss [0.3525]
Epoch [196] Time [442.5460] g_loss [3.3874] h_loss [3.5183] g_h_loss [3.8644] dx_loss [0.1873] dy_loss [0.2055] d_loss [0.3928]
Epoch [197] Time [444.3238] g_loss [3.3821] h_loss [3.4684] g_h_loss [3.8319] dx_loss [0.1660] dy_loss [0.2070] d_loss [0.3730]
Epoch [198] Time [446.0905] g_loss [3.3916] h_loss [3.4319] g_h_loss [3.8191] dx_loss [0.1839] dy_loss [0.1788] d_loss [0.3627]
Epoch [199] Time [447.8589] g_loss [3.3698] h_loss [3.4197] g_h_loss [3.7789] dx_loss [0.2072] dy_loss [0.1925] d_loss [0.3997]
[2.8801820405583136, 2.7586166619128925, 2.6229387346696513, 2.4697832319802027, 2.3132930670672986, 2.1556491449613544, 1.9972661171376582, 1.8513560794384587, 1.7299191231963564]
[137.55619703486667, 129.01194725070548, 127.87302142008913, 124.71919021784925, 124.50773301633102, 125.00839534670577, 125.96119163643706, 126.82494799247753, 132.31009158720974]`
