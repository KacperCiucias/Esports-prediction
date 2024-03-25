import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

'''
This python script is used to prepare data for model training.
The resulting .csv file contains statistics aggregated on the Team level for each match that was deemed viable.
'''


WINDOWS_SIZE = 39 # The size of the rolling window used to aggreagate statistics
RANK_WINDOW_SIZE = 30 # Size of the rolling widnow used to calculate rank trends
ADD_ECON = False # Add data about economy? reduces the number of matches by around 33% due to data missing before 2018
ADD_ROSTER_CHANGES = False # Add data about roster changes?
ADD_GAME_UPDATES = False # Add data about game updates?



results = pd.read_csv('results.csv') # File containg match results
players = pd.read_csv('players.csv') # File containing player statistics
economy = pd.read_csv('economy.csv') # File containing economy data if needed


def calculate_round_types(match):

    '''
    This function is responsible for transforming the economy data into useful information, about the win-raio
    in specific round types (for example the win ration of each team in eco rounds).
    '''

    match = np.array(match[10:])
    t1_economy = match[:30]
    t2_economy = match[30:60]
    round_winners = match[60:90]

    t1_economy = t1_economy[~np.isnan(t1_economy)]
    t2_economy = t2_economy[~np.isnan(t2_economy)]
    round_winners = round_winners[~np.isnan(round_winners)]

    

    #Bin round economy

    t1_pistol_wins = sum([1 for x in [round_winners[0], round_winners[15]] if x == 1])
    t2_pistol_wins = sum([1 for x in [round_winners[0], round_winners[15]] if x == 2])

    t1_eco_round_wins = sum([1 for i,x in enumerate(t1_economy) if (x <= 8000) & (round_winners[i] == 1)]) - t1_pistol_wins
    t1_total_eco_rounds = sum([1 for i,x in enumerate(t1_economy) if (x <= 8000)]) - t1_pistol_wins

    t2_eco_round_wins = sum([1 for i,x in enumerate(t2_economy) if (x <= 8000) & (round_winners[i] == 2)]) - t2_pistol_wins
    t2_total_eco_rounds = sum([1 for i,x in enumerate(t2_economy) if (x <= 8000)]) - t1_pistol_wins

    t1_force_round_wins = sum([1 for i,x in enumerate(t1_economy) if ((x <= 16000) & (x > 8000)) & (round_winners[i] == 1)])
    t1_total_force_rounds = sum([1 for i,x in enumerate(t1_economy) if ((x <= 16000) & (x > 8000))])

    t2_force_round_wins = sum([1 for i,x in enumerate(t2_economy) if ((x <= 16000) & (x > 8000)) & (round_winners[i] == 2)])
    t2_total_force_rounds = sum([1 for i,x in enumerate(t2_economy) if ((x <= 16000) & (x > 8000))])

    t1_full_round_wins = sum([1 for i,x in enumerate(t1_economy) if (x > 16000) & (round_winners[i] == 1)])
    t1_total_full_rounds = sum([1 for i,x in enumerate(t1_economy) if (x > 16000)])

    t2_full_round_wins = sum([1 for i,x in enumerate(t2_economy) if (x > 16000) & (round_winners[i] == 2)])
    t2_total_full_rounds = sum([1 for i,x in enumerate(t2_economy) if (x > 16000)])




    t1_pistol_wr = t1_pistol_wins/2
    t2_pistol_wr = t2_pistol_wins/2

    if(t1_total_eco_rounds==0):
        t1_eco_wr = 0
    else:
        t1_eco_wr = t1_eco_round_wins/t1_total_eco_rounds

    if(t2_total_eco_rounds==0):
        t2_eco_wr = 0
    else:  
        t2_eco_wr = t2_eco_round_wins/t2_total_eco_rounds

    if(t1_total_force_rounds == 0):
        t1_force_wr = 0
    else:
        t1_force_wr = t1_force_round_wins/t1_total_force_rounds

    if(t2_total_force_rounds == 0):
        t2_force_wr = 0
    else:
        t2_force_wr = t2_force_round_wins/t2_total_force_rounds

    if(t1_total_full_rounds == 0):
        t1_full_wr = 0
    else:
        t1_full_wr = t1_full_round_wins/t1_total_full_rounds
    
    if(t2_total_full_rounds == 0):
        t2_full_wr = 0
    else:
        t2_full_wr = t2_full_round_wins/t2_total_full_rounds



    




    #print(f't1 Eco WR: {t1_eco_wr}, t2 Eco WR: {t2_eco_wr}. t1 Force WR: {t1_force_wr}, t2 Force WR: {t2_force_wr}.\n t1 Full WR: {t1_full_wr}, t2 Full WR: {t2_full_wr}. t1 Pistol WR: {t1_pistol_wr}, t2 Pistol WR: {t2_pistol_wr}.')

    return t1_pistol_wr, t1_eco_wr, t1_force_wr, t1_full_wr, t2_pistol_wr, t2_eco_wr, t2_force_wr, t2_full_wr



if ADD_ECON:

    t1_pistol_wr_array = []
    t2_pistol_wr_array = []

    t1_eco_wr_array = []
    t2_eco_wr_array = []

    t1_force_wr_array = []
    t2_force_wr_array = []

    t1_full_wr_array = []
    t2_full_wr_array = []

    for row in economy.itertuples():

        t1_pistol_wr, t1_eco_wr, t1_force_wr, t1_full_wr, t2_pistol_wr, t2_eco_wr, t2_force_wr, t2_full_wr = calculate_round_types(row) # Get round type win-ratios for each match


        # Aggregate by row
        t1_pistol_wr_array.append(t1_pistol_wr)
        t1_eco_wr_array.append(t1_eco_wr)
        t1_force_wr_array.append(t1_force_wr)
        t1_full_wr_array.append(t1_full_wr)

        t2_pistol_wr_array.append(t2_pistol_wr)
        t2_eco_wr_array.append(t2_eco_wr)
        t2_force_wr_array.append(t2_force_wr)
        t2_full_wr_array.append(t2_full_wr)

    # Add win-ratio for each round type to the economy dataframe
    economy['t1_pistol_wr'] = t1_pistol_wr_array
    economy['t1_eco_wr'] = t1_eco_wr_array
    economy['t1_force_wr'] = t1_force_wr_array
    economy['t1_full_wr'] = t1_full_wr_array

    economy['t2_pistol_wr'] = t2_pistol_wr_array
    economy['t2_eco_wr'] = t2_eco_wr_array
    economy['t2_force_wr'] = t2_force_wr_array
    economy['t2_full_wr'] = t2_full_wr_array

    t1_economy = economy[['date','match_id','team_1','_map','t1_pistol_wr', 't1_eco_wr', 't1_force_wr',
       't1_full_wr']]
    t2_economy = economy[['date','match_id','team_2','_map','t2_pistol_wr', 't2_eco_wr', 't2_force_wr',
        't2_full_wr']]

    t1_economy.columns = ['date','match_id','team','map','pistol_wr', 'eco_wr', 'force_wr',
        'full_wr']
    t2_economy.columns = ['date','match_id','team','map','pistol_wr', 'eco_wr', 'force_wr',
        'full_wr']
    team_economy = pd.concat([t1_economy, t2_economy], axis=0).sort_values('date')

    cumulative_statistics_economy = team_economy.groupby('team')[['pistol_wr', 'eco_wr', 'force_wr',
        'full_wr']].rolling(WINDOWS_SIZE, 1).mean().reset_index()

    cumulative_statistics_economy = team_economy.groupby('team')[['pistol_wr', 'eco_wr', 'force_wr',
        'full_wr']].rolling(WINDOWS_SIZE, 1).mean().reset_index()

    cumulative_statistics_economy.columns = ['team','index_','cum_avg_pistol_wr','cum_avg_eco_wr','cum_avg_force_wr','cum_avg_full_wr']

    team_economy = team_economy.reset_index()

    temp = team_economy.merge(cumulative_statistics_economy, left_on=['index','team'], right_on=['index_','team'])

    temp = temp.drop(['index','index_'], axis=1)

    team_economy_rolling = temp

    team_economy_rolling = team_economy_rolling[['date', 'match_id', 'team', 'map','cum_avg_pistol_wr', 'cum_avg_eco_wr', 'cum_avg_force_wr',
        'cum_avg_full_wr']]

    team_economy_shifted = team_economy_rolling.sort_values(['date','match_id']).groupby(['team','map'])[['cum_avg_pistol_wr', 'cum_avg_eco_wr', 'cum_avg_force_wr',
        'cum_avg_full_wr']].shift(1).reset_index()

    team_economy_rolling = team_economy_rolling.reset_index()

    team_economy_rolling = team_economy_rolling.merge(team_economy_shifted, on=['index']).drop(['cum_avg_pistol_wr_x',
        'cum_avg_eco_wr_x', 'cum_avg_force_wr_x', 'cum_avg_full_wr_x'], axis=1)

    team_economy_rolling.columns = ['index', 'date', 'match_id', 'team', 'map', 'cum_avg_pistol_wr',
        'cum_avg_eco_wr', 'cum_avg_force_wr', 'cum_avg_full_wr']

    team_economy_rolling.drop('index', axis=1, inplace=True)

    team_economy = team_economy_rolling


ranks1=results[['date','team_1','rank_1']]
ranks2 = results[['date','team_2','rank_2']]

ranks1.columns = ['date','team','rank']
ranks2.columns = ['date','team','rank']

ranks = pd.concat([ranks1,ranks2])

ranks.drop_duplicates(inplace=True)
ranks.sort_values(['team','date'],inplace=True)
ranks['rank_before'] = ranks.groupby('team')['rank'].shift(1).fillna(method='backfill')
ranks['rank_diff'] = ranks['rank_before'] - ranks['rank']


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

selected_results = results[['match_id','_map','team_1','team_2','result_1','result_2','map_winner']]
selected_results.columns = ['match_id','map','team_1','team_2','result_1','result_2','winner']
selected_results.head()

def calulate_rolling_window_sum(df, col, rolling_period):
    return df.groupby('player_id')[col].rolling(rolling_period, min_periods=1).sum().reset_index(0, drop=True)

def calulate_rolling_window_avg(df, col, rolling_period):
    return df.groupby('player_id')[col].rolling(rolling_period, min_periods=1).mean().reset_index(0, drop=True)

columns_to_sum = ['kills','assists','deaths','hs','kddiff','fkdiff','round_number']
columns_to_avg = ['adr','kast','rating']

window_size = WINDOWS_SIZE

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


columns_to_shift = ['cum_kills', 'cum_assists', 'cum_deaths', 'cum_hs', 'cum_kddiff',
       'cum_fkdiff', 'cum_round_number', 'cum_avg_adr', 'cum_avg_kast',
       'cum_avg_rating']

for map_played in maps_played:
    for col in columns_to_shift:
        maps_df[map_played][col] = maps_df[map_played].sort_values(['date','match_id']).groupby('player_id')[col].shift(1)

columns = ['cum_kills','cum_assists','cum_deaths','cum_hs','cum_kddiff','cum_fkdiff']

for map_played in maps_played:
    for col in columns:
        maps_df[map_played][col] = maps_df[map_played][col]/maps_df[map_played]['cum_round_number']


# Grouping players into teams by match

all_maps = pd.concat([maps_df[map_played] for map_played in maps_played])
all_maps = all_maps.merge(ranks[['date','team','rank_diff']], on=['date','team'])


results['1st_won'] = np.where(results.map_winner == 1, 1, 0)
results['2nd_won'] = np.where(results.map_winner == 2, 1, 0)

rel_columns = ['date', 'player_name', 'team', 'opponent', 'player_id',
       'match_id','best_of', 'map','cum_kills', 'cum_assists', 'cum_deaths', 'cum_hs', 'cum_kddiff',
       'cum_fkdiff', 'cum_avg_adr', 'cum_avg_kast',
       'cum_avg_rating','rank_diff','winner']
all_maps.columns
all_maps = all_maps[rel_columns].sort_values(['match_id','map','team'])

all_team_maps = all_maps.groupby(['date','match_id','map','team'])[['cum_kills', 'cum_assists', 'cum_deaths', 'cum_hs', 'cum_kddiff',
       'cum_fkdiff', 'cum_avg_adr', 'cum_avg_kast',
       'cum_avg_rating','rank_diff']].mean().reset_index()

t1 = all_team_maps.iloc[::2]
t2 = all_team_maps.iloc[1::2]


all_team_maps = t1.merge(t2, on=['match_id','map'])
all_team_maps_2 = all_team_maps.merge(results, left_on=['match_id','map'],right_on=['match_id','_map'])

team_x = all_team_maps_2.team_x
team_y = all_team_maps_2.team_y

team_1 = all_team_maps_2.team_1
team_2 = all_team_maps_2.team_2

winners = all_team_maps_2.map_winner - 1

for i,x in enumerate(team_x):
    if(team_x[i] != team_1[i]):
        winners[i] = 1-winners[i]
all_team_maps['winner'] = winners


all_team_maps.drop('date_y',axis=1, inplace=True)
all_team_maps.columns = ['date','match_id', 'map', 'team_1', 'cum_kills_1', 'cum_assists_1',
       'cum_deaths_1', 'cum_hs_1', 'cum_kddiff_1', 'cum_fkdiff_1',
       'cum_avg_adr_1', 'cum_avg_kast_1', 'cum_avg_rating_1', 'rank_diff_1',
       'team_2', 'cum_kills_2', 'cum_assists_2', 'cum_deaths_2', 'cum_hs_2',
       'cum_kddiff_2', 'cum_fkdiff_2', 'cum_avg_adr_2', 'cum_avg_kast_2',
       'cum_avg_rating_2', 'rank_diff_2', 'winner']

all_team_maps['rank_diff_1'] = all_team_maps.groupby('team_1')['rank_diff_1'].rolling(30, min_periods=1).sum().reset_index(0, drop=True)
all_team_maps['rank_diff_2'] = all_team_maps.groupby('team_2')['rank_diff_2'].rolling(30, min_periods=1).sum().reset_index(0, drop=True)

if ADD_GAME_UPDATES:
    updates = pd.read_excel("game_updates.xlsx")
    updates = updates[["Update date","Majority"]]
    all_team_maps.sort_values('date', inplace=True)
    updates.sort_values('Update date', inplace=True)
    all_team_maps['date'] = all_team_maps.date.astype('datetime64')
    all_team_with_game_updates = pd.merge_asof(
            all_team_maps,
            updates,
            left_on="date",
            right_on="Update date",
            direction="backward",
        )

    all_team_with_game_updates.drop(['date','team_1','team_2'],axis=1,inplace=True)

    all_team_with_game_updates['recent_update'] = np.where(all_team_with_game_updates.Majority == 0, 0, 1)

    all_team_with_game_updates = all_team_with_game_updates[['match_id', 'map', 'cum_kills_1', 'cum_assists_1', 'cum_deaths_1',
        'cum_hs_1', 'cum_kddiff_1', 'cum_fkdiff_1', 'cum_avg_adr_1',
        'cum_avg_kast_1', 'cum_avg_rating_1', 'rank_diff_1', 'cum_kills_2',
        'cum_assists_2', 'cum_deaths_2', 'cum_hs_2', 'cum_kddiff_2',
        'cum_fkdiff_2', 'cum_avg_adr_2', 'cum_avg_kast_2', 'cum_avg_rating_2',
        'rank_diff_2','recent_update','Majority', 'winner']]
    all_team_maps = all_team_with_game_updates


if ADD_ROSTER_CHANGES:

    team_rosters = players[['date','player_id','team','match_id']].sort_values('date', ascending=True).groupby(['team','date'])['player_id'].sum().reset_index()
    merged_rosters = team_rosters.reset_index().merge(team_rosters.sort_values('date', ascending=True).groupby(['team'])['player_id'].shift(1).reset_index(), on='index')

    merged_rosters.drop('index', axis=1, inplace=True)
    merged_rosters.columns = ['team','date','roster_1','roster_2']
    merged_rosters['roster_2'] = np.where(merged_rosters.roster_2.isna(), merged_rosters.roster_1, merged_rosters.roster_2)
    merged_rosters['roster_change'] = 1 - (merged_rosters.roster_1 == merged_rosters.roster_2).astype(int)
    merged_rosters_rolling = merged_rosters.groupby('team')['roster_change'].rolling(window=5, min_periods=1).sum().reset_index().merge(merged_rosters.reset_index(), left_on='level_1', right_on='index')
    merged_rosters_rolling = merged_rosters_rolling[['date','team_x','roster_change_x']]
    merged_rosters_rolling.columns = ['date','team','roster_change']



    all_team_maps = all_team_maps.merge(merged_rosters_rolling, left_on=['date','team_1'], right_on=['date','team']).drop('team', axis=1)
    all_team_maps = all_team_maps.merge(merged_rosters_rolling, left_on=['date','team_2'], right_on=['date','team']).drop('team', axis=1)
    all_team_maps.rename(columns={'roster_change_x':'roster_change_1','roster_change_y':'roster_change_2'}, inplace=True)
    all_team_maps = all_team_maps[['date', 'match_id', 'map', 'team_1', 'cum_kills_1', 'cum_assists_1',
       'cum_deaths_1', 'cum_hs_1', 'cum_kddiff_1', 'cum_fkdiff_1',
       'cum_avg_adr_1', 'cum_avg_kast_1', 'cum_avg_rating_1', 'rank_diff_1','roster_change_1',
       'team_2', 'cum_kills_2', 'cum_assists_2', 'cum_deaths_2', 'cum_hs_2',
       'cum_kddiff_2', 'cum_fkdiff_2', 'cum_avg_adr_2', 'cum_avg_kast_2',
       'cum_avg_rating_2', 'rank_diff_2',
       'roster_change_2','winner']]


# all_team_maps.drop(['team_1','team_2'],axis=1,inplace=True)




if ADD_ECON:
    temp = all_team_maps.merge(team_economy, left_on=['match_id','map','team_1'], right_on=['match_id','map','team'], suffixes=('','_2')).merge(team_economy, left_on=['match_id','map','team_2'], right_on=['match_id','map','team'], suffixes=('','_2'))
    temp = temp[['match_id', 'map', 'team_1', 'cum_kills_1', 'cum_assists_1',
       'cum_deaths_1', 'cum_hs_1', 'cum_kddiff_1', 'cum_fkdiff_1',
       'cum_avg_adr_1', 'cum_avg_kast_1', 'cum_avg_rating_1', 'rank_diff_1','cum_avg_pistol_wr', 'cum_avg_eco_wr', 'cum_avg_force_wr',
       'cum_avg_full_wr', 'team_2', 'cum_kills_2', 'cum_assists_2', 'cum_deaths_2', 'cum_hs_2',
       'cum_kddiff_2', 'cum_fkdiff_2', 'cum_avg_adr_2', 'cum_avg_kast_2',
       'cum_avg_rating_2', 'rank_diff_2','cum_avg_pistol_wr_2',
       'cum_avg_eco_wr_2', 'cum_avg_force_wr_2', 'cum_avg_full_wr_2', 'winner']]
    temp.columns = ['match_id', 'map', 'team_1', 'cum_kills_1', 'cum_assists_1',
       'cum_deaths_1', 'cum_hs_1', 'cum_kddiff_1', 'cum_fkdiff_1',
       'cum_avg_adr_1', 'cum_avg_kast_1', 'cum_avg_rating_1', 'rank_diff_1',
       'cum_avg_pistol_wr_1', 'cum_avg_eco_wr_1', 'cum_avg_force_wr_1',
       'cum_avg_full_wr_1', 'team_2', 'team_2_2', 'cum_kills_2', 'cum_assists_2',
       'cum_deaths_2', 'cum_hs_2', 'cum_kddiff_2', 'cum_fkdiff_2',
       'cum_avg_adr_2', 'cum_avg_kast_2', 'cum_avg_rating_2', 'rank_diff_2',
       'cum_avg_pistol_wr_2', 'cum_avg_eco_wr_2', 'cum_avg_force_wr_2',
       'cum_avg_full_wr_2', 'winner']
    temp.drop('team_2_2', axis=1, inplace=True)
    all_team_maps = temp


    
all_team_maps.to_csv('Teams_statistics.csv',index=False)