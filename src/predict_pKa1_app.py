import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import requests
import csv
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch

st.set_page_config(page_title='First Acid Dissociation Constant Prediction', page_icon=':atom:', layout='wide')

st.title('Predicting First Acid Dissociation Constants')

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Molecule Visualization", "Estimation pKa1 for your molecule", "Data", "Statistics of the model"])
smiles = ''
st.sidebar.header('Input')
smiles = st.sidebar.text_input('Input the molecule (SMILES):')

st.markdown(
    """    
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f0f0; /* Change background color to your desired color */
        padding: 5px 0;
        text-align: center;
        z-index: 9999; /* Set a high z-index value */
    }
    .footer a {
        margin: 0 5px;
        color: #0367d7;
        text-decoration: none;
    }
    </style>
    <div class="footer">
        <p>Cristian Crișan</p>
        <a href="https://github.com/ccrisan7" target="_blank">GitHub</a>
        <a href="https://www.linkedin.com/in/cristian-crisan" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)


if page == "Molecule Visualization":
    

    def smiles_to_name(smiles):
        if not smiles:
            return None

        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
            response = requests.get(url)
            response.raise_for_status()
            cid = response.json()["IdentifierList"]["CID"][0]
            
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
            response = requests.get(url)
            response.raise_for_status()
            synonyms = response.json()["InformationList"]["Information"][0]["Synonym"]
            return synonyms[0]
        except KeyError:
            st.warning("No official name found for the given SMILES.")
            return None
        except IndexError:
            st.warning("No official name found for the given SMILES.")
            return None
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 404:
                st.warning("Resource not found for the given SMILES.")
            return None

    official_name = smiles_to_name(smiles)
    if official_name:
        st.sidebar.write("Substance name:", official_name)

    # Convert SMILES to RDKit molecule object
    if smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.sidebar.success("Molecule successfully loaded.")
                expander_structure = st.expander("Molecule Structure", expanded=True)
                expander_graph = st.expander("Graph Representation", expanded=True)

                with expander_structure:
                    # Resize the molecule image to fit better
                    img = Draw.MolToImage(mol, size=(400, 250))
                    st.image(img, caption='Molecule Structure')

                with expander_graph:
                    # Generate the graph of the molecule
                    mol_graph = nx.Graph()
                    for atom in mol.GetAtoms():
                        mol_graph.add_node(atom.GetIdx(), atomic_symbol=atom.GetSymbol())
                    for bond in mol.GetBonds():
                        mol_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=str(bond.GetBondType()))

                    # Plot the graph
                    plt.figure(figsize=(8, 6))
                    pos = nx.spring_layout(mol_graph)  # Layout for better visualization
                    nx.draw(mol_graph, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10)
                    st.pyplot(plt)

            else:
                st.error("Failed to load molecule. Please input a valid SMILES.")
        except ValueError:
            st.error("Failed to load molecule. Please input a valid SMILES.")

elif page == "Data":
    st.subheader("Initial Data")
    data = pd.read_csv("dataset_IUPAC.csv")
    st.write(data)

    initial_dataset = "dataset_IUPAC.csv"
    used_dataset = 'used_dataset.csv'

    unique_strings = set()  # To store unique strings encountered in the first column

    with open(initial_dataset, 'r', newline='') as f_input, open(used_dataset, 'w', newline='') as f_output:
        csv_reader = csv.reader(f_input)
        csv_writer = csv.writer(f_output)
        csv_writer.writerow(['No.', 'SMILES', 'pKa1'])
        
        index = 0  # Index starts from 0
        
        for row in csv_reader:
            # Check if the row has at least 5 columns and meets the criteria
            if len(row) >= 5 and row[2] == "pK1" and row[4].replace('.', '', 1).isdigit() and 20 <= float(row[4]) <= 25:
                # Check if the string in the first column is unique
                if row[0] not in unique_strings:
                    # Check if the fourth column does not contain unwanted strings
                    unwanted_strings = ["~0.8", "~1.2", "<1.5", "<2", ">11", "4.40-4.80", "7.7-7.8", "ca. 3"]
                    if not any(val in row[3] for val in unwanted_strings):
                        csv_writer.writerow([index, row[1], row[3]])  # Write index, second column, and fourth column
                        unique_strings.add(row[0])
                        index += 1  # Increment index for the next row

    st.subheader("Processed Data")
    data = pd.read_csv("used_dataset.csv")
    st.write(data)


elif page == "Statistics of the model":
    st.image("output2.png")
    st.image("output3.png")
    st.image("output4.png")


elif page == "Estimation pKa1 for your molecule":
    PAULING_ELECTRONEGATIVITY = {
        "C": 2.55,
        "Si": 1.90,
        "P": 2.19,
        "I": 2.66,
        "As": 2.18,
        "Cl": 3.16,
        "B": 2.04,
        "Se": 2.55,
        "S": 2.58,
        "Br": 2.96,
        "N": 3.04,
        "F": 3.98,
        "Ge": 2.01,
        "O": 3.44
    }

    class Molecule:
        def __init__(self, smiles):
            self.smiles = smiles
            self.graph, self.adj_matrix = self._generate_graph()
            self.pk1 = 7
            self.num_node_features = 9
            self.num_bond_features = 5
        
        def _generate_graph(self):
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is not None:
                G = nx.Graph()
                for atom in mol.GetAtoms():
                    atom_attrs = {
                        "formal_charge": atom.GetFormalCharge(),
                        "num_implicit_hs": atom.GetNumImplicitHs(),
                        "is_aromatic": atom.GetIsAromatic(),
                        "mass": atom.GetMass(),
                        "degree": atom.GetDegree(),
                        "hybridization": atom.GetHybridization(),
                        "num_radical_electrons": atom.GetNumRadicalElectrons(),
                        "is_in_ring": atom.IsInRing(),
                        "pauling_electronegativity": self.get_pauling_electronegativity(atom.GetSymbol())
                    }
                    G.add_node(atom.GetIdx(), **atom_attrs)
                for bond in mol.GetBonds():
                    bond_attrs = {
                        "bond_type": bond.GetBondTypeAsDouble(),
                        "is_conjugated": bond.GetIsConjugated(),
                        "is_aromatic": bond.GetIsAromatic(),
                        "stereo": bond.GetStereo(),
                        "is_in_ring": bond.IsInRing()
                    }
                    G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **bond_attrs)
                adj_matrix = nx.adjacency_matrix(G).todense()
                adj_matrix = adj_matrix[:50, :50]  # Take only the first 50x50 submatrix
                adj_matrix = np.pad(adj_matrix, ((0, 50 - adj_matrix.shape[0]), (0, 50 - adj_matrix.shape[1])), mode='constant')
                return G, adj_matrix
            return None, None
        
        def get_pauling_electronegativity(self, symbol):
            return PAULING_ELECTRONEGATIVITY.get(symbol, None)

    if smiles:
        molecule = Molecule(smiles)
        if molecule.graph:
            st.write("## Molecule Details:")
            st.write("SMILES:", molecule.smiles)
            st.write("Number of Nodes (Atoms):", molecule.graph.number_of_nodes())
            st.write("Number of Edges (Bonds):", molecule.graph.number_of_edges())
            
            atom_data = []
            for atom_idx, atom_attrs in molecule.graph.nodes(data=True):
                atom_info = {"Atom Index": atom_idx}
                for attr_name, attr_value in atom_attrs.items():
                    atom_info[attr_name] = attr_value
                atom_data.append(atom_info)
            
            bond_data = []
            for atom1, atom2, bond_attrs in molecule.graph.edges(data=True):
                bond_info = {"Atom 1": atom1, "Atom 2": atom2}
                for attr_name, attr_value in bond_attrs.items():
                    bond_info[attr_name] = attr_value
                bond_data.append(bond_info)
            
            st.write("\nAtom Details:")
            st.table(atom_data)
            
            st.write("\nBond Details:")
            st.table(bond_data)
        else:
            st.write("Invalid SMILES string. Cannot generate molecule details.")


        class GNNModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.35):
                super(GNNModel, self).__init__()

                # Node feature encoder
                self.node_encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim)
                )

                # Convolutional layers
                self.convs = nn.ModuleList()
                for _ in range(num_layers):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))

                # Dropout
                self.dropout = nn.Dropout(dropout)

                # Final regression layer
                self.regression = nn.Linear(hidden_dim, output_dim)

                # Fitting parameters: weight matrices and bias vectors
                self.W1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
                self.b1 = nn.Parameter(torch.randn(hidden_dim))
                self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim))
                self.b2 = nn.Parameter(torch.randn(output_dim))

            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch

                # Embed raw features
                x = self.node_encoder(x)

                for conv in self.convs:
                    x = F.relu(conv(x, edge_index))
                    x = self.dropout(x)

                # Global pooling
                x = global_mean_pool(x, batch)

                # Regression
                x = F.relu(torch.matmul(x, self.W1) + self.b1)
                out = torch.matmul(x, self.W2) + self.b2

                return out
        class MoleculeDataset(torch.utils.data.Dataset):
            def __init__(self, molecules):
                self.molecules = molecules
            
            def __len__(self):
                return len(self.molecules)
            
            def __getitem__(self, idx):
                molecule = self.molecules[idx]
                graph = molecule.graph
                adj_matrix = molecule.adj_matrix
                # Convert networkx graph to PyTorch Geometric Data object
                x = []  # Node features
                edge_index = []  # Edge connectivity
                edge_attr = []  # Edge features
                y = molecule.pk1  # PKa1 value
                
                for node_idx, attrs in graph.nodes(data=True):
                    # Node features
                    node_feats = [attrs["formal_charge"], attrs["num_implicit_hs"], attrs["is_aromatic"], attrs["mass"],
                                attrs["degree"], attrs["hybridization"], attrs["num_radical_electrons"], attrs["is_in_ring"], attrs["pauling_electronegativity"]]
                    x.append(node_feats)
                
                for u, v, attrs in graph.edges(data=True):
                    # Edge connectivity
                    edge_index.append([u, v])
                    # Edge features
                    edge_feats = [attrs["bond_type"], attrs["is_conjugated"], attrs["is_aromatic"], attrs["stereo"], attrs["is_in_ring"]]
                    edge_attr.append(edge_feats)
                
                x = torch.tensor(x, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                
                return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([y], dtype=torch.float), adj_matrix=torch.tensor(adj_matrix, dtype=torch.float))

        molecule_list = []
        molecule_list.append(molecule)
        molecule_list.append(molecule)
        dataset = MoleculeDataset(molecule_list)

        def custom_collate(batch):
            return Batch.from_data_list(batch)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

        input_dim = 9  # Assuming 9-dimensional input features
        hidden_dim = 48
        output_dim = 1  # Assuming single-dimensional regression output
        num_layers = 2

        model = GNNModel(input_dim, hidden_dim, output_dim, num_layers)
        model.load_state_dict(torch.load("model.pth"))

        predicted_values = []
        true_values = []

        # Evaluation loop using the test loader
        model.eval()
        with torch.no_grad():
            for data in loader:
                outputs = model(data)
                predicted_values.extend(outputs.squeeze().tolist())
                true_values.extend(data.y.squeeze().tolist())

        # Convert lists to numpy arrays for easier manipulation if needed
        predicted_values = np.array(predicted_values)
        true_values = np.array(true_values)

        predicted_pka1 = predicted_values[0]

        superscript_digits = {
        '0': '⁰',
        '1': '¹',
        '2': '²',
        '3': '³',
        '4': '⁴',
        '5': '⁵',
        '6': '⁶',
        '7': '⁷',
        '8': '⁸',
        '9': '⁹',
        }

        def format_power_of_10(value):
            exponent = -math.floor(math.log10(abs(value)))
            coefficient = value * (10 ** exponent)
            
            # Convert exponent to superscript
            exponent_str = ''.join(superscript_digits[d] for d in str(exponent))
            
            formatted_result = "{:.2f} × 10⁻{}".format(coefficient, exponent_str)
            return formatted_result

        st.write(f"### The predicted value of pKa₁ for the molecule is: {predicted_pka1:.2f}")
        st.write("### The predicted value of Ka₁ for the molecule is: ", format_power_of_10(10 ** -predicted_pka1))
