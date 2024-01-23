import torch
import pandas as pd

# Load data
player_df = pd.read_csv("tbl_player.csv")
skill_df = pd.read_csv("tbl_player_skill.csv")
team_df = pd.read_csv("tbl_team.csv")

# Extract subsets
player_df = player_df[["int_player_id", "str_player_name", "str_positions", "int_overall_rating", "int_team_id"]]
skill_df = skill_df[["int_player_id", "int_long_passing", "int_ball_control", "int_dribbling"]]
team_df = team_df[["int_team_id", "str_team_name", "int_overall"]]

# Merge data
player_df = player_df.merge(skill_df, on='int_player_id')
fifa_df = player_df.merge(team_df, on='int_team_id')

# Sort dataframe
fifa_df = fifa_df.sort_values(by="int_overall_rating", ascending=False)
print("Players: ", fifa_df.shape[0])
print(fifa_df.head() )
max(fifa_df["int_player_id"].value_counts())

# Sort to define the order of nodes
sorted_df = fifa_df.sort_values(by="int_player_id")
# Select node features
node_features = sorted_df[["str_positions", "int_long_passing", "int_ball_control", "int_dribbling"]]
# Convert non-numeric columns
pd.set_option('mode.chained_assignment', None)
positions = node_features["str_positions"].str.split(",", expand=True)
node_features["first_position"] = positions[0]
# One-hot encoding
node_features = pd.concat([node_features, pd.get_dummies(node_features["first_position"])], axis=1, join='inner')
node_features.drop(["str_positions", "first_position"], axis=1, inplace=True)
node_features.head() 

# Convert to numpy
x = node_features.to_numpy()
x.shape # [num_nodes x num_features]



# Sort to define the order of nodes
sorted_df = fifa_df.sort_values(by="int_player_id")
# Select node features
labels = sorted_df[["int_overall"]]
labels.head()
# Convert to numpy
y = labels.to_numpy()
y.shape
# Remap player IDs
fifa_df["int_player_id"] = fifa_df.reset_index().index
# This tells us how many players per team we have to connect
fifa_df["str_team_name"].value_counts()

import itertools
import numpy as np

teams = fifa_df["str_team_name"].unique()
all_edges = np.array([], dtype=np.int32).reshape((0, 2))
for team in teams:
    team_df = fifa_df[fifa_df["str_team_name"] == team]
    players = team_df["int_player_id"].values
    # Build all combinations, as all players are connected
    permutations = list(itertools.combinations(players, 2))
    edges_source = [e[0] for e in permutations]
    edges_target = [e[1] for e in permutations]
    team_edges = np.column_stack([edges_source, edges_target])
    all_edges = np.vstack([all_edges, team_edges])
# Convert to Pytorch Geometric format
edge_index = all_edges.transpose()
edge_index # [2, num_edges]
e = torch.tensor(edge_index, dtype=torch.long)
print(e)

edge_index1 = e.t().clone().detach()
edge_index1

from torch_geometric.data import Data
data = Data(x=x, edge_index=edge_index1.to().contiguous(), y=y)

from torch_geometric.loader import DataLoader
data_list = [Data(...), ..., Data(...)]
loader = DataLoader(data_list, batch_size=32)
from torch_geometric.data import Data
data = Data(x=x, edge_index=edge_index1.t().contiguous(), y=y)

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
networkX_graph = to_networkx(data)

import networkx as nx
nx.draw(networkX_graph)