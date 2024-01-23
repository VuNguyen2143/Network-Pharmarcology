#import required packages
#from operator import pos
import networkx as nx
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#import pygrank as pg

#Retrieve data from STRING DB
protein_list = ['ABI1', 'ABL1', 'ABL2', 'ACKR3', 'ACVR1', 'ACVR1B', 'ACVR2A', 'AFDN', 'AFF1', 'AFF3', 'AFF4', 'AKT1', 'AKT2', 'ALK', 'AMER1', 'APC', 'AR', 'ARHGAP26', 'ARHGAP35', 'ARHGEF12', 'ARID1A', 'ARID1B', 'ARID2', 'ARNT', 'ASPSCR1', 'ASXL1', 'ATF1', 'ATIC', 'ATM', 'ATP1A1', 'ATP2B3', 'ATR', 'ATRX', 'AXIN1', 'AXIN2', 'B2M', 'BAP1', 'BARD1', 'BAX', 'BCL10', 'BCL11A', 'BCL11B', 'BCL2', 'BCL3', 'BCL6', 'BCL7A', 'BCL9', 'BCL9L', 'BCOR', 'BCR', 'BIRC3', 'BLM', 'BMPR1A', 'BRAF', 'BRCA1', 'BRCA2', 'BRD3', 'BRD4', 'BRIP1', 'BTG1', 'BTK', 'BUB1B', 'CACNA1D', 'CALR', 'CANT1', 'CARD11', 'CARS1', 'CASP8', 'CBFA2T3', 'CBFB', 'CBL', 'CBLB', 'CBLC', 'CCDC6', 'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CD274', 'CD74', 'CD79A', 'CD79B', 'CDC73', 'CDH1', 'CDH11', 'CDK12', 'CDK4', 'CDK6', 'CDKN1B', 'CDKN2A', 'CDKN2C', 'CEBPA', 'CENPU', 'CHD4', 'CHEK2', 'CIITA', 'CLIP1', 'CLTC', 'CLTCL1', 'CNOT3', 'CNTRL', 'COL1A1', 'COL2A1', 'CREB1', 'CREB3L1', 'CREB3L2', 'CREBBP', 'CRLF2', 'CRTC1', 'CSF3R', 'CTCF', 'CTNNB1', 'CUX1', 'CXCR4', 'CYLD', 'DAXX', 'DCTN1', 'DDB2', 'DDIT3', 'DDX10', 'DDX3X', 'DDX5', 'DDX6', 'DEK', 'DICER1', 'DNAJB1', 'DNM2', 'DNMT3A', 'DROSHA', 'EBF1', 'EGFR', 'EIF3E', 'ELF4', 'ELK4', 'ELL', 'EML4', 'EP300', 'EPAS1', 'EPS15', 'ERBB2', 'ERBB3', 'ERBB4', 'ERC1', 'ERCC2', 'ERCC3', 'ERCC4', 'ERCC5', 'ERG', 'ESR1', 'ETV4', 'ETV6', 'EWSR1', 'EZH2', 'EZR', 'FANCA', 'FANCC', 'FANCD2', 'FANCE', 'FANCF', 'FANCG', 'FAS', 'FBXW7', 'FCGR2B', 'FES', 'FGFR1', 'FGFR1OP', 'FGFR2', 'FGFR3', 'FGFR4', 'FIP1L1', 'FLI1', 'FLT3', 'FLT4', 'FOXA1', 'FOXO1', 'FOXO3', 'FOXO4', 'FOXP1', 'FUS', 'GAS7', 'GATA1', 'GATA2', 'GATA3', 'GNA11', 'GNAQ', 'GNAS', 'GPC3', 'GRIN2A', 'H3-3B', 'H3C12', 'H4C6', 'HEY1', 'HGF', 'HIF1A', 'HIP1', 'HLA-A', 'HMGA1', 'HMGA2', 'HNRNPA2B1', 'HOXA11', 'HOXA9', 'HRAS', 'HSP90AA1', 'HSP90AB1', 'IDH1', 'IGLL5', 'IKBKB', 'IKZF1', 'IL2', 'IL21R', 'IL6ST', 'IL7R', 'IRF4', 'IRS4', 'ITK', 'JAK1', 'JAK2', 'JAK3', 'JUN', 'KAT6A', 'KAT6B', 'KCNJ5', 'KDM5A', 'KDM5C', 'KDM6A', 'KDR', 'KEAP1', 'KIF5B', 'KIT', 'KLF4', 'KMT2A', 'KMT2C', 'KMT2D', 'KNL1', 'KRAS', 'KTN1', 'LATS1', 'LATS2', 'LCK', 'LEF1', 'LIFR', 'LMNA', 'LMO1', 'LMO2', 'LOC102723407', 'LYL1', 'LZTR1', 'MAF', 'MAFB', 'MALT1', 'MAML2', 'MAP2K1', 'MAP2K2', 'MAP2K4', 'MAP3K1', 'MAPK1', 'MAX', 'MDM2', 'MDM4', 'MECOM', 'MED12', 'MEN1', 'MET', 'MITF', 'MLF1', 'MLH1', 'MLLT1', 'MLLT10', 'MLLT3', 'MPL', 'MRTFA', 'MSH2', 'MSH6', 'MSN', 'MTCP1', 'MTOR', 'MUC1', 'MYB', 'MYC', 'MYCL', 'MYD88', 'MYH11', 'MYH9', 'MYOD1', 'NBN', 'NCOA1', 'NCOA2', 'NCOA4', 'NCOR1', 'NCOR2', 'NDRG1', 'NF1', 'NF2', 'NFATC2', 'NFE2L2', 'NFIB', 'NFKB2', 'NFKBIE', 'NIN', 'NONO', 'NOTCH1', 'NPM1', 'NR4A3', 'NRAS', 'NRG1', 'NSD2', 'NTRK1', 'NTRK2', 'NTRK3', 'NUMA1', 'NUP214', 'NUP98', 'PALB2', 'PATZ1', 'PAX3', 'PAX5', 'PAX8', 'PBRM1', 'PBX1', 'PCM1', 'PDCD1LG2', 'PDE4DIP', 'PDGFB', 'PDGFRA', 'PDGFRB', 'PICALM', 'PIK3CA', 'PIK3CB', 'PIK3R1', 'PIM1', 'PLCG1', 'PML', 'PMS2', 'POLD1', 'POLE', 'POLQ', 'POT1', 'POU5F1', 'PPARG', 'PPFIBP1', 'PPM1D', 'PPP2R1A', 'PPP6C', 'PRCC', 'PRDM1', 'PREX2', 'PRKACA', 'PRKAR1A', 'PTCH1', 'PTEN', 'PTK6', 'PTPN11', 'PTPN13', 'QKI', 'RAC1', 'RAD21', 'RAD50', 'RAD51B', 'RAF1', 'RANBP2', 'RAP1GDS1', 'RARA', 'RB1', 'RBM10', 'RECQL4', 'RET', 'RHOA', 'RMI2', 'RNF43', 'RPL10', 'RPL22', 'RPL5', 'RSPO2', 'RSPO3', 'RUNX1', 'RUNX1T1', 'SALL4', 'SET', 'SF3B1', 'SFPQ', 'SMAD2', 'SMAD3', 'SMAD4', 'SMARCA4', 'SMARCB1', 'SMARCE1', 'SMO', 'SND1', 'SOCS1', 'SOX2', 'SPOP', 'SRC', 'SRSF2', 'SRSF3', 'SSX1', 'STAT3', 'STAT5B', 'STAT6', 'STK11', 'STRN', 'SUZ12', 'TAF15', 'TAL1', 'TAL2', 'TCEA1', 'TCF12', 'TCF7L2', 'TFE3', 'TFEB', 'TFG', 'TGFBR2', 'TNFAIP3', 'TOP1', 'TP53', 'TP63', 'TPM3', 'TPM4', 'TPR', 'TRIM24', 'TRIP11', 'TSC1', 'TSC2', 'U2AF1', 'USP8', 'WWTR1', 'XPA'
]
proteins = '%0d'.join(protein_list)
url = 'https://string-db.org/api/tsv/network?identifiers=' + proteins + '&species=9606'+'&required_score=400'
r = requests.get(url)


lines = r.text.split('\n') # pull the text from the response object and split based on new lines
data = [l.split('\t') for l in lines] # split each line into its components based on tabs
# convert to dataframe using the first row as the column names; drop empty, final row
df = pd.DataFrame(data[1:-1], columns = data[0]) 
#print(df.head(5))
df['escore'] = df['escore'].astype(float)
df['dscore'] = df['dscore'].astype(float)
# dataframe with the preferred names of the two proteins and the score of the interaction
df_ex_da = df[(df['escore']>=0.4) | (df['dscore']>=0.4)]
interactions_ex_da=df_ex_da[['preferredName_A', 'preferredName_B', 'score']]  

# Calculate node degrees

    
df_dti = pd.read_csv('drug.target.interaction.tsv.gz', compression='gzip', sep='\t')
#print(df_dti.head(3))
df_dti_human=df_dti[df_dti['ORGANISM']=='Homo sapiens']
df_dti_human_tiger1=df_dti_human[df_dti_human['GENE'].isin(protein_list)]
df_dti_human_tiger1=df_dti_human_tiger1.copy()
df_dti_human_tiger1.rename(columns={'GENE':'preferredName_A'},inplace=True)
df_drug_pro=pd.merge(df_dti_human_tiger1,df_ex_da,on='preferredName_A',how='inner')
#df_drug_pro.to_excel('D:\Coding\Datasets\OncologyTT-Datasets\OncologyTT-Datasets-main\GNN\drug_protein_interactions.xlsx', index=False)

interactions_drug_pro=df_drug_pro[['DRUG_NAME', 'preferredName_A', 'ACT_VALUE']]  
interactions_drug_pro=interactions_drug_pro.dropna()

#interactions = df[['preferredName_A', 'preferredName_B', 'score']]  
#print(interactions_ex_da.head(5))
#print(interactions.head(5))
#interactions_ex_da.to_excel('D:\Coding\Datasets\OncologyTT-Datasets\OncologyTT-Datasets-main\GNN\interactions.xlsx', index=False)
#Build Graph
G=nx.Graph(name='Protein Interaction Graph')
interactions = np.array(interactions_ex_da)
for i in range(len(interactions)):
    interaction = interactions[i]
    a = interaction[0] # protein a node
    b = interaction[1] # protein b node
    w = float(interaction[2]) # score as weighted edge where high scores = low weight
    G.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph 
    
dp_G=nx.Graph(name='Drug Protein Interaction Graph')
interactions = np.array(interactions_drug_pro)
for i in range(len(interactions)):
    interaction = interactions[i]
    a = interaction[0] # protein a node
    b = interaction[1] # protein b node
    w = float(interaction[2]) # score as weighted edge where high scores = low weight
    dp_G.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph 

degree_centrality = nx.degree_centrality(G)

# Find the top 5 nodes with the highest degree centrality
top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# Print the top 5 nodes
print("Top 5 nodes based on Degree Centrality:")
for i, (node, centrality) in enumerate(top_nodes, start=1):
    print(f"{i}. Node: {node}, Degree Centrality: {centrality}")
    
betweenness_centrality = nx.betweenness_centrality(G)
top_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# Print the top 5 nodes
print("Top 5 nodes based on Betweenness Centrality:")
for i, (node, centrality) in enumerate(top_nodes, start=1):
    print(f"{i}. Node: {node}, Degree Centrality: {centrality}")

closeness_centrality = nx.closeness_centrality(G)
top_nodes = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# Print the top 5 nodes
print("Top 5 nodes based on Closeness Centrality:")
for i, (node, centrality) in enumerate(top_nodes, start=1):
    print(f"{i}. Node: {node}, Degree Centrality: {centrality}")
    
eigenvector_centrality = nx.eigenvector_centrality(G)
top_nodes = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# Print the top 5 nodes
print("Top 5 nodes based on Eigenvector Centrality:")
for i, (node, centrality) in enumerate(top_nodes, start=1):
    print(f"{i}. Node: {node}, Degree Centrality: {centrality}")
    
pagerank = nx.pagerank(G)
top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]

# Print the top 5 nodes
print("Top 5 nodes based on Pagerank Centrality:")
for i, (node, centrality) in enumerate(top_nodes, start=1):
    print(f"{i}. Node: {node}, Degree Centrality: {centrality}")

    """
    katz_centrality = nx.katz_centrality(G)
top_nodes = sorted(katz_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# Print the top 5 nodes
print("Top 5 nodes based on Katz Centrality:")
for i, (node, centrality) in enumerate(top_nodes, start=1):
    print(f"{i}. Node: {node}, Degree Centrality: {centrality}")
    """

    
harmonic_centrality = nx.harmonic_centrality(G)
top_nodes = sorted(harmonic_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# Print the top 5 nodes
print("Top 5 nodes based on Harmonic Centrality:")
for i, (node, centrality) in enumerate(top_nodes, start=1):
    print(f"{i}. Node: {node}, Degree Centrality: {centrality}")
    
load_centrality = nx.load_centrality(G)
top_nodes = sorted(load_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# Print the top 5 nodes
print("Top 5 nodes based on Load Centrality:")
for i, (node, centrality) in enumerate(top_nodes, start=1):
    print(f"{i}. Node: {node}, Degree Centrality: {centrality}")