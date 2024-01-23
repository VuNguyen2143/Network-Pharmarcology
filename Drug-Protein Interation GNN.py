import networkx as nx
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#Download drug-target interaction flat file from DrugCentral
df_dti = pd.read_csv('drug.target.interaction.tsv.gz', compression='gzip', sep='\t')
#print(df_dti.head(3))
df_dti_human=df_dti[df_dti['ORGANISM']=='Homo sapiens']
tiger1_gene_list = ["ABI1", "ABL1", "ABL2", "ACKR3", "ACSL3", "ACVR1", "ACVR1B", "ACVR2A", "AFDN", "AFF1", "AFF3", "AFF4", "AKT1", "AKT2", "ALK", "AMER1", "APC", "APOBEC3B", "AR", "ARHGAP26", "ARHGAP35", "ARHGEF12", "ARID1A", "ARID1B", "ARID2", "ARNT", "ASPSCR1", "ASXL1", "ATF1", "ATIC", "ATM", "ATP1A1", "ATP2B3", "ATR", "ATRX", "AXIN1", "AXIN2", "B2M", "BAP1", "BARD1", "BAX", "BCL10", "BCL11A", "BCL11B", "BCL2", "BCL3", "BCL6", "BCL7A", "BCL9", "BCL9L", "BCOR", "BCORL1", "BCR", "BIRC3", "BLM", "BMPR1A", "BRAF", "BRCA1", "BRCA2", "BRD3", "BRD4", "BRIP1", "BTG1", "BTK", "BUB1B", "CACNA1D", "CALR", "CAMTA1", "CANT1", "CARD11", "CARS", "CASP8", "CBFA2T3", "CBFB", "CBL", "CBLB", "CBLC", "CCDC6", "CCNB1IP1", "CCND1", "CCND2", "CCND3", "CCNE1", "CD274", "CD74", "CD79A", "CD79B", "CDC73", "CDH1", "CDH11", "CDK12", "CDK4", "CDK6", "CDKN1B", "CDKN2A", "CDKN2C", "CDX2", "CEBPA", "CHCHD7", "CHD4", "CHEK2", "CIC", "CIITA", "CLIP1", "CLTC", "CLTCL1", "CNBP", "CNOT3", "CNTRL", "COL1A1", "COL2A1", "CREB1", "CREB3L1", "CREB3L2", "CREBBP", "CRLF2", "CRTC1", "CRTC3", "CSF3R", "CTCF", "CTNNB1", "CUX1", "CXCR4", "CYLD", "DAXX", "DCTN1", "DDB2", "DDIT3", "DDR2", "DDX10", "DDX3X", "DDX5", "DDX6", "DEK", "DICER1", "DNAJB1", "DNM2", "DNMT3A", "DROSHA", "EBF1", "EGFR", "EIF3E", "EIF4A2", "ELF4", "ELK4", "ELL", "EML4", "EP300", "EPAS1", "EPS15", "ERBB2", "ERBB3", "ERBB4", "ERC1", "ERCC2", "ERCC3", "ERCC4", "ERCC5", "ERG", "ESR1", "ETNK1", "ETV1", "ETV4", "ETV5", "ETV6", "EWSR1", "EXT1", "EXT2", "EZH2", "EZR", "FANCA", "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG", "FAS", "FAT1", "FAT4", "FBXO11", "FBXW7", "FCGR2B", "FCRL4", "FES", "FEV", "FGFR1", "FGFR1OP", "FGFR2", "FGFR3", "FGFR4", "FH", "FHIT", "FIP1L1", "FLCN", "FLI1", "FLT3", "FLT4", "FOXA1", "FOXL2", "FOXO1", "FOXO3", "FOXO4", "FOXP1", "FSTL3", "FUBP1", "FUS", "GAS7", "GATA1", "GATA2", "GATA3", "GNA11", "GNAQ", "GNAS", "GOLGA5", "GOPC", "GPC3", "GPHN", "GRIN2A", "H3F3A", "H3F3B", "HERPUD1", "HEY1", "HGF", "HIF1A", "HIP1", "HIST1H3B", "HIST1H4I", "HLA-A", "HLF", "HMGA1", "HMGA2", "HNF1A", "HNRNPA2B1", "HOOK3", "HOXA11", "HOXA13", "HOXA9", "HOXC11", "HOXC13", "HOXD11", "HOXD13", "HRAS", "HSP90AA1", "HSP90AB1", "IDH1", "IDH2", "IGH", "IGK", "IGL", "IKBKB", "IKZF1", "IKZF3", "IL2", "IL21R", "IL6ST", "IL7R", "IRF4", "IRS4", "ITK", "JAK1", "JAK2", "JAK3", "JUN", "KAT6A", "KAT6B", "KCNJ5", "KDM5A", "KDM5C", "KDM6A", "KDR", "KDSR", "KEAP1", "KIF5B", "KIT", "KLF4", "KLF6", "KLK2", "KMT2A", "KMT2C", "KMT2D", "KNL1", "KRAS", "KTN1", "LASP1", "LATS1", "LATS2", "LCK", "LEF1", "LIFR", "LMNA", "LMO1", "LMO2", "LPP", "LRIG3", "LRP1B", "LYL1", "LZTR1", "MAF", "MAFB", "MALT1", "MAML2", "MAP2K1", "MAP2K2", "MAP2K4", "MAP3K1", "MAP3K13", "MAPK1", "MAX", "MDM2", "MDM4", "MECOM", "MED12", "MEN1", "MET", "MITF", "MLF1", "MLH1", "MLLT1", "MLLT10", "MLLT11", "MLLT3", "MLLT6", "MN1", "MPL", "MRTFA", "MSH2", "MSH6", "MSI2", "MSN", "MTCP1", "MTOR", "MUC1", "MUTYH", "MYB", "MYC", "MYCL", "MYCN", "MYD88", "MYH11", "MYH9", "MYO5A", "MYOD1", "NAB2", "NBN", "NCOA1", "NCOA2", "NCOA4", "NCOR1", "NCOR2", "NDRG1", "NF1", "NF2", "NFATC2", "NFE2L2", "NFIB", "NFKB2", "NFKBIE", "NIN", "NKX2-1", "NONO", "NOTCH1", "NOTCH2", "NPM1", "NR4A3", "NRAS", "NRG1", "NSD1", "NSD2", "NSD3", "NT5C2", "NTRK1", "NTRK2", "NTRK3", "NUMA1", "NUP214", "NUP98", "NUTM1", "NUTM2B", "NUTM2D", "OLIG2", "P2RY8", "PAFAH1B2", "PALB2", "PATZ1", "PAX3", "PAX5", "PAX7", "PAX8", "PBRM1", "PBX1", "PCM1", "PDCD1LG2", "PDE4DIP", "PDGFB", "PDGFRA", "PDGFRB", "PER1", "PHF6", "PHOX2B", "PICALM", "PIK3CA", "PIK3CB", "PIK3R1", "PIM1", "PLAG1", "PLCG1", "PML", "PMS2", "POLD1", "POLE", "POLQ", "POT1", "POU2AF1", "POU5F1", "PPARG", "PPFIBP1", "PPM1D", "PPP2R1A", "PPP6C", "PRCC", "PRDM1", "PRDM16", "PREX2", "PRF1", "PRKACA", "PRKAR1A", "PRRX1", "PSIP1", "PTCH1", "PTEN", "PTK6", "PTPN11", "PTPN13", "PTPRB", "PTPRC", "PTPRK", "PTPRT", "QKI", "RABEP1", "RAC1", "RAD21", "RAD50", "RAD51B", "RAF1", "RANBP2", "RAP1GDS1", "RARA", "RB1", "RBM10", "RBM15", "RECQL4", "REL", "RET", "RHOA", "RHOH", "RMI2", "RNF213", "RNF43", "ROS1", "RPL10", "RPL22", "RPL5", "RPN1", "RRAS2", "RSPO2", "RSPO3", "RUNX1", "RUNX1T1", "SALL4", "SBDS", "SDC4", "SDHA", "SDHAF2", "SDHB", "SDHC", "SDHD", "SET", "SETBP1", "SETD2", "SF3B1", "SFPQ", "SFRP4", "SH2B3", "SH3GL1", "SIX1", "SLC34A2", "SLC45A3", "SMAD2", "SMAD3", "SMAD4", "SMARCA4", "SMARCB1", "SMARCE1", "SMO", "SND1", "SOCS1", "SOX2", "SPEN", "SPOP", "SRC", "SRSF2", "SRSF3", "SS18", "SS18L1", "SSX1", "SSX2", "SSX4", "STAG2", "STAT3", "STAT5B", "STAT6", "STIL", "STK11", "STRN", "SUFU", "SUZ12", "SYK", "TAF15", "TAL1", "TAL2", "TBL1XR1", "TBX3", "TCEA1", "TCF12", "TCF3", "TCF7L2", "TCL1A", "TENT5C", "TERT", "TET1", "TET2", "TFE3", "TFEB", "TFG", "TGFBR2", "TLX1", "TLX3", "TMEM127", "TMPRSS2", "TNFAIP3", "TNFRSF14", "TNFRSF17", "TOP1", "TP53", "TP63", "TPM3", "TPM4", "TPR", "TRA", "TRAF7", "TRB", "TRD", "TRIM24", "TRIM27", "TRIM33", "TRIP11", "TRRAP", "TSC1", "TSC2", "U2AF1", "UBR5", "USP6", "USP8", "VHL", "WAS", "WDCP", "WIF1", "WRN", "WT1", "WWTR1", "XPA", "XPC", "XPO1", "YWHAE", "ZBTB16", "ZFHX3", "ZMYM2", "ZNF331", "ZNF384", "ZNF521", "ZRSR2"
]
df_dti_human_tiger1=df_dti_human[df_dti_human['GENE'].isin(tiger1_gene_list)]
#column_names = df_dti.columns.tolist()
#Generate unique ID for drug
#df_dti_human_tiger1['DRUG_ID'], _ = pd.factorize(df_dti_human_tiger1['DRUG_NAME'])
df_dti_human_tiger1.loc[:, 'DRUG_ID'], _ = pd.factorize(df_dti_human_tiger1['DRUG_NAME'])
df_drugs=df_dti_human_tiger1[['DRUG_ID','DRUG_NAME']].copy().drop_duplicates()
print(df_drugs.head(5))
#print(column_names)
columns_to_select = ['DRUG_ID', 'GENE', 'ACT_VALUE']
dr_pr_inter_df=df_dti_human_tiger1[columns_to_select].copy()
dr_pr_inter_df=dr_pr_inter_df.dropna()
#gene_to_remove=['JUN','HRAS','HIF1A','MDM2','IL2','EZH2','HSP90AA1','HLA-A','AKT1','BRD4']
print(dr_pr_inter_df.head(3))
#print(dr_pr_inter_df.shape)

#dr_pr_inter_df=dr_pr_inter_df[~dr_pr_inter_df['GENE'].isin(gene_to_remove)]


drug_protein_graph = nx.Graph(name='Drug-Protein Interaction Graph')

# Add nodes for drugs and proteins
interactions = np.array(dr_pr_inter_df)
for i in range(len(interactions)):
    interaction = interactions[i]
    a = interaction[0] # 
    b = interaction[1] # 
    w = float(interaction[2]) # score as weighted edge where high scores = low weight
    drug_protein_graph.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph
    []
# Calculate node degrees
node_degrees = dict(drug_protein_graph.degree())
drug_node_degrees = {key: value for key, value in node_degrees.items() if isinstance(key, (int, float))}
#print(node_degrees)
#print(drug_node_degrees)
# Find the top 5 nodes with the highest degrees
top_nodes = sorted(drug_node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
# Extract node names from the top nodes
top_node_names = [node[0] for node in top_nodes]


print("Top 10 nodes with the most interactions:")
for node in top_nodes:
    print(f"Node: {node[0]}, Drug name: {df_drugs.loc[df_drugs['DRUG_ID']==node[0],'DRUG_NAME'].iloc[0]}, Degree: {node[1]}")
    
def rescale(l,newmin,newmax):
    arr = list(l)
    return [(x-min(arr))/(max(arr)-min(arr))*(newmax-newmin)+newmin for x in arr]
# use the matplotlib plasma colormap
graph_colormap = cm.get_cmap('plasma')
# node color varies with Degree
c = rescale([drug_protein_graph.degree(v) for v in drug_protein_graph],0.0,0.9) 
c = [graph_colormap(i) for i in c]
# node size varies with betweeness centrality - map to range [10,100] 
bc = nx.betweenness_centrality(drug_protein_graph) # betweeness centrality
s =  rescale([v for v in bc.values()],250,1000)
# edge width shows 1-weight to convert cost back to strength of interaction 
ew = rescale([float(drug_protein_graph[u][v]['weight']) for u,v in drug_protein_graph.edges],0.1,0.5)
# edge color also shows weight
ec = rescale([float(drug_protein_graph[u][v]['weight']) for u,v in drug_protein_graph.edges],0.1,0.5)
ec = [graph_colormap(i) for i in ec]

pos = nx.spring_layout(drug_protein_graph)
plt.figure(figsize=(10,10),facecolor='White')
nx.draw_networkx(drug_protein_graph, pos=pos, with_labels=True, node_color=c, node_size=s,edge_color= ec,width=ew,font_color='white',font_weight='bold',font_size='4')
plt.axis('off')
plt.show()
print("Graph Info:")
print("Number of nodes:", drug_protein_graph.number_of_nodes())
print("Number of edges:", drug_protein_graph.number_of_edges())
