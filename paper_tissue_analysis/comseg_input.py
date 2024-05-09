


import pandas as pd



dico_bc_gene0 = {
    'r1': "Rtkn2",  # bc1
    'r2': "Lamp3",  # bc3
    'r3': "Pecam1",  # bc4
    'r4': "Ptprb",  # bc5
    'r5': "Pdgfra",  # bc6
    'r6': "Chil3",  # bc7
    'r7': "unknow",  # bc1
    'r8': "Rtkn2",  # bc3
    'r9': "Lamp3",  # bc4
    'r10': "Pecam1",  # bc7
}


df = pd.read_csv("/media/tom/Transcend/2023-10-06_LUSTRA/input_comseg/img0.csv")

##keep only the gene  round 1 ro 6

df = df[df["round_name"].isin(["r1", "r2", "r3", "r4", "r5", "r6"])]

## rename the gene name

list_gene = list(df["gene"])
list_gene = [dico_bc_gene0[gene] for gene in list_gene]
df["gene"] = list_gene

df.to_csv("/media/tom/Transcend/2023-10-06_LUSTRA/input_comseg/csv/img0.csv")