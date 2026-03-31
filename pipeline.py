#lets get started
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from scipy.linalg import eigh #used as eigen value solver. as far as i read similar to numpy but used for heavy stuff, and eigh = eigen decomposition for hermitian and skew hermitian
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import networkx as nx #used to create manipulate and analyze graph

from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors 

def mol_to_graph(smiles):
    """Convert SMILES to adjacency matrix and atom features"""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol) #add hydrogens for richer topology
    n = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            int(atom.GetIsAromatic()),
            atom.GetFormalCharge(),
        ])
    X = np.array(features, dtype=np.float32)

    A = np.zeros((n,n), dtype = np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i,j] = A[j,i] = 1.0

    return X, A, mol 

def compute_laplacian(A):
    """ normalized laplacian is L = I - [D(raised to -1/2)*A*D(raised to 1/2)]"""
    D = np.diag(A.sum(axis=1))
    D_invsqrt = np.diag(1.0/ np.sqrt(A.sum(axis=1).clip(1e-8)))
    L = np.eye(len(A)) - D_invsqrt @ A @ D_invsqrt
    return L 

def spectral_analysis(L):
    """Eigendecomposition of L. eigenvalues is resonant frequency of the molecule, eigen vectors is the new coordinate system"""
    eigenvalues, eigenvectors = eigh(L) #eigh is for symmetric matrices more stable than eig
    fingerprint = np.round(eigenvalues, 4).toList()
    #we round it to 4 decimals to catch isospectral graphs despite noise

    #spectral gap : diff betn lambda 1 and lambda 2, 
    #high gap is well connected rigid, low gap is loossely connected more reactive
    spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
    return eigenvalues, eigenvectors, spectral_gap


def spectral_embedding(eigenvectors):
    """Use eigen vectors 1,2,3 as (x,y,z ) coordinates. this is laplacian eigenmap. it places atoms so that bonded atoms are close in spectral space. rotation and translation invariant"""
    return eigenvectors[:, 1:4]

#chebnet

class ChebConv(nn.Module):
    """Spectral graph conv using chebyshev polynomials."""
    def __init__(self, in_ch, out_ch, K=3):
        super().__init__()
        self.K = K 
        #one weight matrix per polynomial order 
        self.weight = nn.Parameter(torch.randn(K, in_ch, out_ch) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x, L_tilde):
        """x is node features, L_tilde is scaled Laplacian"""
        Tx = [x, L_tilde @ x]
        for k in range(2, self.K):
            Tx.append(2 * L_tilde @ Tx[-1] - Tx[-2])

        out = sum(Tx[k] @ self.weight[k] for k in range(self.K))
        return out + self.bias 

class SpectralGNN(nn.Module):
    """full model is 2 chebconv layer and then we do global mean pool to mlp to property prediction"""
    def __init__(self, in_ch, hidden=32, out_ch=1, K=3):
        super().__init__()
        self.conv1 = ChebConv(in_ch, hidden, K)
        self.conv2 = ChebConv(hidden, hidden, K)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, out_ch)
        )
    
    def forward(self, x, L_tilde):
        x = F.relu(self.conv1(x, L_tilde))
        x = F.relu(self.conv2(x, L_tilde))
        x = x.mean(dim=0)
        return self.head(x)

def make_L_tilde(L):
    """Scale L eigen values betwen [-1, 1] for Chebyshev stability."""
    lambda_max = 2.0  # normalized Laplacian always has λ_max ≤ 2
    L_tilde = (2.0 / lambda_max) * L - np.eye(len(L))
    return torch.tensor(L_tilde, dtype=torch.float32)

def plot_dashboard(smiles, name):
    X, A, mol = mol_to_graph(smiles)
    L = compute_laplacian(A)
    eigenvalues, eigenvectors, gap = spectral_analysis(L)
    embedding = spectral_embedding(eigenvectors)
    fiedler_vector = eigenvectors[:,1]

    nodal_colors = ['#ff7f0e', if val > 0 else '#1f77b4' for val in fiedler_vector]

    print(f"\n{'─'*40}")
    print(f"Molecule : {name}")
    print(f"Atoms    : {len(X)}")
    print(f"Spectral Gap: {gap:.4f}  ({'rigid/aromatic' if gap > 0.3 else 'flexible/reactive'})")

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Harmonic Manifold Analysis: {name}", fontsize=14, fontweight='bold')

    #Eigenspectrum
    ax1 = fig.add_subplot(131)
    ax1.bar(range(len(eigenvalues)), eigenvalues, color='steelblue', alpha=0.8)
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.5, label=f'Spectral gap: {gap:.3f}')
    ax1.set_xlabel('Eigenvalue index')
    ax1.set_ylabel('λ (frequency)')
    ax1.set_title('Eigenspectrum\n(Vibrational Signature)')
    ax1.legend(fontsize=8)

    # 3D Spectral Embedding ──
    ax2 = fig.add_subplot(132, projection='3d')
    atomic_nums = X[:, 0]
    colors = plt.cm.plasma(atomic_nums / atomic_nums.max())
    ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=nodal_colors, s=100, edgecolors = 'white', alpha=0.9)
    # Draw bonds in spectral space
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[i, j] > 0:
                ax2.plot([embedding[i,0], embedding[j,0]],
                         [embedding[i,1], embedding[j,1]],
                         [embedding[i,2], embedding[j,2]],
                         'gray', alpha=0.3, linewidth=0.8)
    ax2.set_title('3D Harmonic Manifold\n(Spectral Embedding)')
    ax2.set_xlabel('v₁'); ax2.set_ylabel('v₂'); ax2.set_zlabel('v₃')

    # graph structures
    ax3 = fig.add_subplot(133)
    G = nx.from_numpy_array(A)
    labels = {i: mol.GetAtomWithIdx(i).GetSymbol() for i in range(len(X))}
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, labels=labels, ax=ax3,
            node_color='lightblue', node_size=300,
            font_size=7, edge_color='gray', width=1.5)
    ax3.set_title('Molecular Graph\n(Bond Topology)')

    plt.tight_layout()
    plt.savefig(f'{name.lower()}_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {name.lower()}_dashboard.png")

    return X, L, gap

molecules = {
    "Caffeine":  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Aspirin":   "CC(=O)Oc1ccccc1C(=O)O",
    "Benzene":   "c1ccccc1",
    "Dopamine":  "NCCc1ccc(O)c(O)c1",
    "LSD": "CCN(CC)C(=O)[C@H]1CN(C)[C@@H]2Cc3c[nH]c4cccc(C2=C1)c34",
    "Psilocybin": "CN(C)CCc1c[nH]c2cccc(OP(=O)(O)O)c12"
}

model = SpectralGNN(in_ch=4, hidden=32, out_ch=1, K=3)
print("Model parameters:", sum(p.numel() for p in model.parameters()))

results = {}
for name, smiles in molecules.items():
    X, L, gap = plot_dashboard(smiles, name)

    # Quick forward pass (untrained — just verifying the pipeline works)
    L_tilde = make_L_tilde(L)
    x_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x_tensor, L_tilde)
    print(f"Model output (untrained): {pred.item():.4f}")
    results[name] = {"spectral_gap": gap, "n_atoms": len(X)}

print("\n── Summary ──")
for name, r in results.items():
    print(f"{name:12s} | gap={r['spectral_gap']:.4f} | atoms={r['n_atoms']}")