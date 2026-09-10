import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings("ignore")

# ── Load Excel ────────────────────────────────────────────────────────────────
df = pd.read_excel(
    "data/Surface_Tension_Aim3_Full.xlsx",
    sheet_name="Surface_Tension_Dataset",
    header=3
)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Column names from the sheet ───────────────────────────────────────────────
CAT_SMILES = "Cation SMILES"
AN_SMILES  = "Anion SMILES"
TARGET     = "Surface\nTension (mN/m)"

# ── RDKit descriptor function ─────────────────────────────────────────────────
def get_descriptors(smiles, prefix):
    result = {}
    invalid = ["", "—", "nan", "None", "FILL FROM PUBCHEM",
               "populate from PubChem", "pending synthesis"]

    if pd.isna(smiles) or str(smiles).strip() in invalid:
        for key in ["MW","F_count","HBD","HBA","TPSA","RotBonds","RingCount","LogP"]:
            result[f"{prefix}_{key}"] = np.nan
        return result

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        print(f"  WARNING: could not parse: {smiles}")
        for key in ["MW","F_count","HBD","HBA","TPSA","RotBonds","RingCount","LogP"]:
            result[f"{prefix}_{key}"] = np.nan
        return result

    result[f"{prefix}_MW"]       = round(Descriptors.MolWt(mol), 2)
    result[f"{prefix}_F_count"]  = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    result[f"{prefix}_HBD"]      = rdMolDescriptors.CalcNumHBD(mol)
    result[f"{prefix}_HBA"]      = rdMolDescriptors.CalcNumHBA(mol)
    result[f"{prefix}_TPSA"]     = round(Descriptors.TPSA(mol), 2)
    result[f"{prefix}_RotBonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    result[f"{prefix}_RingCount"]= rdMolDescriptors.CalcNumRings(mol)
    result[f"{prefix}_LogP"]     = round(Descriptors.MolLogP(mol), 3)
    return result

# ── Compute descriptors for each row ─────────────────────────────────────────
print("\nComputing RDKit descriptors...")
rows = []
for idx, row in df.iterrows():
    cat = get_descriptors(row[CAT_SMILES], "CAT")
    an  = get_descriptors(row[AN_SMILES],  "AN")
    rows.append({**cat, **an})

df_rdkit = pd.DataFrame(rows)

# ── Derived features ──────────────────────────────────────────────────────────
df_rdkit["F_Total"]  = df_rdkit["CAT_F_count"].fillna(0) + df_rdkit["AN_F_count"].fillna(0)
df_rdkit["MW_Total"] = df_rdkit["CAT_MW"].fillna(0) + df_rdkit["AN_MW"].fillna(0)
df_rdkit["HBD_Total"]= df_rdkit["CAT_HBD"].fillna(0) + df_rdkit["AN_HBD"].fillna(0)
df_rdkit["HBA_Total"]= df_rdkit["CAT_HBA"].fillna(0) + df_rdkit["AN_HBA"].fillna(0)
df_rdkit["TPSA_Total"]= df_rdkit["CAT_TPSA"].fillna(0) + df_rdkit["AN_TPSA"].fillna(0)
df_rdkit["LogP_Total"]= df_rdkit["CAT_LogP"].fillna(0) + df_rdkit["AN_LogP"].fillna(0)

# ── Merge with original ───────────────────────────────────────────────────────
df_final = pd.concat([df.reset_index(drop=True),
                      df_rdkit.reset_index(drop=True)], axis=1)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = "data/surface_tension_with_rdkit.csv"
df_final.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(f"Shape: {df_final.shape}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nRDKit features computed:")
rdkit_cols = [c for c in df_rdkit.columns]
for col in rdkit_cols:
    n = df_rdkit[col].notna().sum()
    print(f"  {col:20s}  {n}/{len(df)} rows filled")