import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


'''
This python script is used to generate dissagregated player statistics for each match that can be used to train the models.
'''

WINDOW_SIZE = 39 # How many matches to consider when calculating historic statistics


# Read in the required dataframes
results = pd.read_csv('results.csv') # Contains results for each match
players = pd.read_csv('players.csv') # Contains player statistics for each match


# Divide all match statistics into individual games played
players_bo1_m1 = players[['date','player_name','team','opponent','country','player_id','match_id','event_id','event_name','best_of',
 'map_1','kills','assists','deaths','hs','kast','kddiff','adr','fkdiff','rating']]
players__bo1_m1=players_bo1_m1[players_bo1_m1.best_of == 1]

players_m1 = players[players.best_of.isin([2,3,5])][['date','player_name','team','opponent','country','player_id','match_id','event_id','event_name','best_of',
 'map_1','m1_kills','m1_assists','m1_deaths','m1_hs','m1_kast','m1_kddiff','m1_adr','m1_fkdiff','m1_rating']]

players_m2 = players[players.best_of.isin([2,3,5])][['date','player_name','team','opponent','country','player_id','match_id',
 'event_id','event_name','best_of','map_2','m2_kills','m2_assists','m2_deaths','m2_hs','m2_kast','m2_kddiff','m2_adr','m2_fkdiff',
 'm2_rating']]

players_m3 = players[players.best_of.isin([3,5])][['date','player_name','team','opponent','country','player_id','match_id',
 'event_id','event_name','best_of','map_3','m3_kills','m3_assists','m3_deaths','m3_hs','m3_kast','m3_kddiff','m3_adr','m3_fkdiff',
 'm3_rating']]

# Rename columns to allow joins
new_columns = ['date','player_name','team','opponent','country','player_id','match_id','event_id','event_name','best_of',
 'map','kills','assists','deaths','hs','kast','kddiff','adr','fkdiff','rating']


players__bo1_m1.columns = new_columns
players_m1.columns = new_columns
players_m2.columns = new_columns
players_m3.columns = new_columns


# Combine statistics and drop where map is NaN
players_map = pd.concat([players__bo1_m1,players_m1,players_m2,players_m3])
players_map = players_map[players_map.map.notna()]

# Remove matches with missing players (bo1 == 10 entries, bo2 == 20 entries, bo3 == 30 entries)
players_map.dropna(inplace=True)
players_map = players_map.groupby('match_id').filter(lambda x: len(x)%10 == 0)
players_map.sort_values('date',inplace=True)


# Split dataframe into map statistics for each player
maps_played = players_map.map.unique()
maps_df = {map_name: players_map[players_map.map == map_name] for map_name in maps_played}

# Select only relevant columns from results
selected_results = results[['match_id','_map','team_1','team_2','result_1','result_2','map_winner']]
selected_results.columns = ['match_id','map','team_1','team_2','result_1','result_2','winner']


# Functions used to calculate the summary statistics in some window of time

def calulate_rolling_window_sum(df, col, rolling_period):
    return df.groupby('player_id')[col].rolling(rolling_period, min_periods=1).sum().reset_index(0, drop=True)

def calulate_rolling_window_avg(df, col, rolling_period):
    return df.groupby('player_id')[col].rolling(rolling_period, min_periods=1).mean().reset_index(0, drop=True)

# Columns on which to aggregate
columns_to_sum = ['kills','assists','deaths','hs','kddiff','fkdiff','round_number']
columns_to_avg = ['adr','kast','rating']

window_size = WINDOW_SIZE # How many previous matches should be included in the 'history' of a player


# Calculate cumulative statistics based on games played up to some certain point 
for map_played in maps_played:
    maps_df[map_played] = maps_df[map_played].merge(selected_results, on=['match_id','map'])
    maps_df[map_played]['round_number'] = maps_df[map_played].result_1 + maps_df[map_played].result_2 
    maps_df[map_played].sort_values('date',inplace=True)

    for col in columns_to_sum:
        new_col = 'cum_'+col
        maps_df[map_played][new_col] = calulate_rolling_window_sum(maps_df[map_played], col, window_size)
    for col in columns_to_avg:
        new_col = 'cum_avg_'+col
        maps_df[map_played][new_col] = calulate_rolling_window_avg(maps_df[map_played], col, window_size)


# Shif the statistics by one match down so that the current values for a match are historics values
columns_to_shift = ['cum_kills', 'cum_assists', 'cum_deaths', 'cum_hs', 'cum_kddiff',
       'cum_fkdiff', 'cum_round_number', 'cum_avg_adr', 'cum_avg_kast',
       'cum_avg_rating']

for map_played in maps_played:
    for col in columns_to_shift:
        maps_df[map_played][col] = maps_df[map_played].sort_values(['date','match_id']).groupby('player_id')[col].shift(1)


# Change cumulative statistics to per round statistics
columns = ['cum_kills','cum_assists','cum_deaths','cum_hs','cum_kddiff','cum_fkdiff']
for map_played in maps_played:
    for col in columns:
        maps_df[map_played][col] = maps_df[map_played][col]/maps_df[map_played]['cum_round_number']

# Combine all statistics into single dataframe
all_maps = pd.concat([maps_df[map_played] for map_played in maps_played])

################# The following code fixes a problem, where sometimes the wrong result would be assigned to a match
################# This is due to the way the data of single players is joined together later

w_list = []
all_maps['winner'] = all_maps['winner'] - 1
for row in all_maps.itertuples():

    if(row.team != row.team_1):
        w_list.append(1-row.winner)
    else:
        w_list.append(row.winner)

all_maps['winner'] = np.array(w_list)

#########################################

# Choose only columns relevant in predicting match outcomes

rel_columns = ['date', 'player_name', 'team', 'opponent', 'player_id',
       'match_id','best_of', 'map','cum_kills', 'cum_assists', 'cum_deaths', 'cum_hs', 'cum_kddiff',
       'cum_fkdiff', 'cum_avg_adr', 'cum_avg_kast',
       'cum_avg_rating','winner']
all_maps.columns
all_maps = all_maps[rel_columns]


# Function creating matchups from player data
def create_matchups_players(all_maps):

    from functools import reduce

    all_maps.sort_values(['match_id','map','team','cum_avg_rating'],inplace=True) # Sort players by match, team and map to form groups

    # Choose every nth player where n is the player number in a match (match = 10 players)
    # Team 1
    t1_p1 = all_maps.iloc[::10]
    t1_p2 = all_maps.iloc[1::10]
    t1_p3 = all_maps.iloc[2::10]
    t1_p4 = all_maps.iloc[3::10]
    t1_p5 = all_maps.iloc[4::10]

    # Team 2
    t2_p1 = all_maps.iloc[5::10]
    t2_p2 = all_maps.iloc[6::10]
    t2_p3 = all_maps.iloc[7::10]
    t2_p4 = all_maps.iloc[8::10]
    t2_p5 = all_maps.iloc[9::10]


    player_dfs = [t1_p1,t1_p2,t1_p3,t1_p4,t1_p5,t2_p1,t2_p2,t2_p3,t2_p4,t2_p5]

    for df in player_dfs:
        print(df.shape) # Sanity Check

    # Rename columns to reflect player number
    for i, p in enumerate(player_dfs):
        p.rename(columns={c: c+'_p'+str(i+1) for c in p.columns if c not in ['date','match_id','team','map','best_of','winner','opponent','best_of']},inplace=True)
    

    # Left join all players into teams
    temp_t1 = reduce(lambda left, right: pd.merge(left,right, on=['match_id','team','map','date','winner','opponent','best_of']), [t1_p1,t1_p2,t1_p3,t1_p4,t1_p5])
    temp_t2 = reduce(lambda left, right: pd.merge(left,right, on=['match_id','team','map','date','winner','opponent','best_of']), [t2_p1,t2_p2,t2_p3,t2_p4,t2_p5])


    print(temp_t1.shape,temp_t2.shape) # Second sanitiy check, numbers of rows and collumns should appear logical

    temp = temp_t1.merge(temp_t2, left_on=['date','match_id','map','team','best_of'], right_on=['date','match_id','map','opponent','best_of'])
    print(temp.shape)


    # Create single target variable
    temp['winner'] = temp.winner_x
    temp.drop(['winner_x','winner_y'],inplace=True, axis=1)

    # Select which columns to keep
    columns_to_keep = ['date']
    columns_to_keep.append('match_id')
    columns_to_keep.append('map')
    names = ['cum_kills_p',
    'cum_assists_p',
    'cum_deaths_p',
    'cum_hs_p',
    'cum_kddiff_p',
    'cum_fkdiff_p',
    'cum_avg_adr_p',
    'cum_avg_kast_p',
    'cum_avg_rating_p']

    for i in range(1,11):
        for stat_name in names:
            columns_to_keep.append(stat_name + str(i))

    columns_to_keep.append('winner')

    final_stats = temp[columns_to_keep]

    # Return combined statistics
    return final_stats

# Call the above function
final_stats = create_matchups_players(all_maps)

# Remove all missing values
final_stats.dropna(inplace=True)

final_stats.to_csv('players_stats.csv',index=False)

