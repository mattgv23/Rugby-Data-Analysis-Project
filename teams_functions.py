import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def results_calculator(matches_df): 
    #Find the unique values of the scoreboard
    #Locals and visitors columns
    scoreboard_columns_local = r'scoreboard_local_\d{2}_type'
    scoreboard_columns_visitor = r'scoreboard_visitor_\d{2}_type'
    
    #Join conditions for filtering in one condition
    scoreboard_columns = f'{scoreboard_columns_local}|{scoreboard_columns_visitor}' #Cuidado con meter un espacio entre medias
    
    #Filter scorebodard type columns 
    scoreboard_df = matches_df.filter(regex=scoreboard_columns)
    
    #Store the types in array
    scoreboard_types = scoreboard_df.stack().dropna().unique()

    #The value per action is the next: 
    #'E': 5 points
    #'T': 2 points
    #'PC': 3 points
    #'D': 3 points
    #'EC': 5 points 
    
    scoreboard_values = [5, 2, 3, 3, 5]
    
    #Create a dictionary to associate action to values
    scoreboard_dict = dict(zip(scoreboard_types, scoreboard_values))
    
    print('Scoreboard values: ', scoreboard_dict)

    #Create a column for local points scoreboard and another for visitors
    #Evaluate every scoreboard_local/visitor_xx_type and add the value to the new columns created
    
    #Create new column as a series
    local_scoreboard_series = None
    temp_local_scoreboard_series = None
    
    local_try_series = None
    visitor_try_series = None
    
    local_conversion_series = None
    visitor_conversion_series = None
    
    local_penalty_kick_series = None
    visitor_penalty_kick_series = None
    
    local_drop_series = None
    visitor_drop_series = None
    
    local_penalty_try_series = None
    visitor_penalty_try_series = None

    for column in matches_df.filter(regex=scoreboard_columns_local).columns:
        if local_scoreboard_series is None:
            temp_local_scoreboard_series = matches_df[column].map(scoreboard_dict).fillna(0)
            local_scoreboard_series = temp_local_scoreboard_series.copy()
    
            local_try_series = matches_df[column].map(lambda x: 1 if x == 'E' else 0).fillna(0)
            local_conversion_series = matches_df[column].map(lambda x: 1 if x == 'T' else 0).fillna(0)
            local_penalty_kick_series = matches_df[column].map(lambda x: 1 if x == 'PC' else 0).fillna(0)
            local_drop_series = matches_df[column].map(lambda x: 1 if x == 'D' else 0).fillna(0)
            local_penalty_try_series = matches_df[column].map(lambda x: 1 if x == 'EC' else 0).fillna(0)
        else: 
            temp_local_scoreboard_series = matches_df[column].map(scoreboard_dict).fillna(0)
            local_scoreboard_series = local_scoreboard_series + temp_local_scoreboard_series 
    
            local_try_series = local_try_series + matches_df[column].map(lambda x: 1 if x == 'E' else 0).fillna(0)
            local_conversion_series = local_conversion_series + matches_df[column].map(lambda x: 1 if x == 'T' else 0).fillna(0)
            local_penalty_kick_series = local_penalty_kick_series + matches_df[column].map(lambda x: 1 if x == 'PC' else 0).fillna(0)
            local_drop_series = local_drop_series + matches_df[column].map(lambda x: 1 if x == 'D' else 0).fillna(0)
            local_penalty_try_series = local_penalty_try_series + matches_df[column].map(lambda x: 1 if x == 'EC' else 0).fillna(0)


    #Create new column as a series
    visitor_scoreboard_series = None
    temp_visitor_scoreboard_series = None
    
    for column in matches_df.filter(regex=scoreboard_columns_visitor).columns:
        if visitor_scoreboard_series is None:
            temp_visitor_scoreboard_series = matches_df[column].map(scoreboard_dict).fillna(0)
            visitor_scoreboard_series = temp_visitor_scoreboard_series.copy()
    
            visitor_try_series = matches_df[column].map(lambda x: 1 if x == 'E' else 0).fillna(0)
            visitor_conversion_series = matches_df[column].map(lambda x: 1 if x == 'T' else 0).fillna(0)
            visitor_penalty_kick_series = matches_df[column].map(lambda x: 1 if x == 'PC' else 0).fillna(0)
            visitor_drop_series = matches_df[column].map(lambda x: 1 if x == 'D' else 0).fillna(0)
            visitor_penalty_try_series = matches_df[column].map(lambda x: 1 if x == 'EC' else 0).fillna(0)
        else: 
            temp_visitor_scoreboard_series = matches_df[column].map(scoreboard_dict).fillna(0)
            visitor_scoreboard_series = visitor_scoreboard_series + temp_visitor_scoreboard_series 
    
            visitor_try_series = visitor_try_series + matches_df[column].map(lambda x: 1 if x == 'E' else 0).fillna(0)
            visitor_conversion_series = visitor_conversion_series + matches_df[column].map(lambda x: 1 if x == 'T' else 0).fillna(0)
            visitor_penalty_kick_series = visitor_penalty_kick_series + matches_df[column].map(lambda x: 1 if x == 'PC' else 0).fillna(0)
            visitor_drop_series = visitor_drop_series + matches_df[column].map(lambda x: 1 if x == 'D' else 0).fillna(0)
            visitor_penalty_try_series = visitor_penalty_try_series + matches_df[column].map(lambda x: 1 if x == 'EC' else 0).fillna(0)

    #Check the results
    
    matches_df['local_scoreboard_points'] = local_scoreboard_series
    matches_df['visitor_scoreboard_points'] = visitor_scoreboard_series
    
    matches_df['local_try'] = local_try_series
    matches_df['local_conversion'] = local_conversion_series
    matches_df['local_penalty_kick'] = local_penalty_kick_series
    matches_df['local_drop'] = local_drop_series
    matches_df['local_penalty_try'] = local_penalty_try_series
    
    matches_df['visitor_try'] = visitor_try_series
    matches_df['visitor_conversion'] = visitor_conversion_series
    matches_df['visitor_penalty_kick'] = visitor_penalty_kick_series
    matches_df['visitor_drop'] = visitor_drop_series
    matches_df['visitor_penalty_try'] = visitor_penalty_try_series
    

    return matches_df

#Jersey number to license number

def jersey_to_license(matches_df):
    #import pandas as pd
    #Create two dictionaries (one for local team and other for visitor) for every match with 
    #key: player_jersey_number and value: license_number.

    #The columns to associate in a dictionary shall be those that contain 'xx_jersey' and 'xx_license'

    #Not downcasting since this shall be removed in future versions
    try:
        pd.set_option('future.no_silent_downcasting', True)
    except:
        pass

    #Loop to run per match (row)
    for j in range(len(matches_df)):

        #Lists to store the jersey number and the license number per match
        local_team_jerseys = []
        local_team_license = []
        
        visitor_team_jerseys = []
        visitor_team_license = []  
        
        for i in range(1,24): 
            local_team_jerseys.append(matches_df[f"local_team_player_{i:02d}_jersey"][j])
            local_team_license.append(matches_df[f"local_team_player_{i:02d}_license_number"][j])
        
            visitor_team_jerseys.append(matches_df[f"visitor_team_player_{i:02d}_jersey"][j])
            visitor_team_license.append(matches_df[f"visitor_team_player_{i:02d}_license_number"][j])
        
        #Transform lists into a dictionary
        local_team_jersey_to_license = dict(zip(local_team_jerseys, local_team_license))
        visitor_team_jersey_to_license = dict(zip(visitor_team_jerseys, visitor_team_license))
        
        #Create a filter for the columns that transformation shall be implemented
        local_filter_01 = r'scoreboard_local_\d{2}_player'
        local_filter_02 = r'substitutions_local_\d{2}_player_in'
        local_filter_03 = r'substitutions_local_\d{2}_player_out'
        local_filter_04 = r'cards_local_\d{2}_player'
        
        visitor_filter_01 = r'scoreboard_visitor_\d{2}_player'
        visitor_filter_02 = r'substitutions_visitor_\d{2}_player_in'
        visitor_filter_03 = r'substitutions_visitor_\d{2}_player_out'
        visitor_filter_04 = r'cards_visitor_\d{2}_player'
        
        #Merge all the filter in a general one
        local_filter = f'{local_filter_01}|{local_filter_02}|{local_filter_03}|{local_filter_04}'
        visitor_filter = f'{visitor_filter_01}|{visitor_filter_02}|{visitor_filter_03}|{visitor_filter_04}'

        #Get the columns to filter
        local_columns = matches_df.iloc[j].filter(regex=local_filter)
        visitor_columns = matches_df.iloc[j].filter(regex=visitor_filter)

        #Replace values in filtered columns
        local_replace_columns = local_columns.replace(local_team_jersey_to_license)
        visitor_replace_columns = visitor_columns.replace(visitor_team_jersey_to_license)

        #Assign replaces values to corresponding columns in the original dataframe
        matches_df.loc[j, local_replace_columns.index] = local_replace_columns
        matches_df.loc[j, visitor_replace_columns.index] = visitor_replace_columns

        return matches_df
    

    #Number of players calculator

def number_of_players(matches_df):
    #Calculate the total players per match

    #Create two lists where these values shall be stored
    local_team_players = []
    visitor_team_players = []

    #Run through every match
    for i in matches_df.index:

        #Create a series with the players jerssey
        local_lineup = matches_df.iloc[i].filter(regex='local_team_player_\d{2}_jersey')
        visitor_lineup = matches_df.iloc[i].filter(regex='visitor_team_player_\d{2}_jersey')

        #Sum the amout of players per match and append to the lists
        local_team_players.append(local_lineup.notna().sum())
        visitor_team_players.append(visitor_lineup.notna().sum())


    matches_df['local_team_players'] = local_team_players
    matches_df['visitor_team_players'] = visitor_team_players

    return matches_df

#Create a dataframe for the teams
import numpy as np
def create_team_df (matches_df, team):
    
    #Create a dataframe for the team as team and its rival teams, respectively
    as_local_team_df = matches_df[matches_df['local_team'] == team].copy()
    as_visitor_team_df = matches_df[matches_df['visitor_team'] == team].copy()
    
    #Ajust column names
    #In as_local_team_df the word 'local' shall dissapear from column names, while 'visitor' shall be switched to 'rival', 
    #and a new column named 'as_local' with value equals to 1 shall be created and inserted after 'rival_team' column
    as_local_team_df.columns = as_local_team_df.columns.str.replace(r'local_', '', regex=True)
    as_local_team_df.columns = as_local_team_df.columns.str.replace(r'visitor', 'rival', regex=True)
    as_local_team_df.insert(7, 'as_local', 1)
    
    #In as_visitor_team the word 'visitor' shall dissapear from column names, while 'local' shall be switched to 'rival', 
    #and a new column named 'as_local' with value equals to 0 shall be created and insserted after 'rival_team' column. 
    as_visitor_team_df.columns = as_visitor_team_df.columns.str.replace(r'visitor_', '', regex=True)
    as_visitor_team_df.columns = as_visitor_team_df.columns.str.replace(r'local', 'rival', regex=True)
    as_visitor_team_df.insert(7, 'as_local', 0)
    
    #In addition, the columns including 'rival_team' and 'team' from the 'as_visitor_team_df' dataframe need to be 
    #reordered as the columns from 'as_local_team_df'
    as_local_team_columns = as_local_team_df.columns #Get the 'as_local_team_df' columns
    
    #Add every missing columns from 'as_local_team_df' to 'as_visitor_team_df'
    for column in as_local_team_columns: 
        if column not in as_visitor_team_df.columns: 
            as_visitor_team_df[column] = np.nan
    
    as_visitor_team_df = as_visitor_team_df[as_local_team_columns] #Reorder the columns
    
    team_df = pd.concat([as_local_team_df, as_visitor_team_df])
    team_df.sort_values(['date', 'match_time'], inplace=True)
    
    return team_df

#General points histograms in time
def points_to_histogram(team_df, team_to_study):
    #Create a figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    team_score_minutes_list = []
    rival_score_minutes_list = []

    #Run through every column
    for i in range(1, 27):

        #Get the tries minutes for team
        team_score_minutes_series = team_df[team_df[f'scoreboard_{i:02d}_type'].notna()][f'scoreboard_{i:02d}_minute']
        team_score_minutes_list = team_score_minutes_list + team_score_minutes_series.tolist()

        try:
            #Get the tries minutes for rival
            rival_score_minutes_series = team_df[team_df[f'scoreboard_rival_{i:02d}_type'].notna()][f'scoreboard_rival_{i:02d}_minute']
            rival_score_minutes_list = rival_score_minutes_list + rival_score_minutes_series.tolist()

        except KeyError:
            pass

    #Create a histogram for the points made by the team
    sns.histplot(x=team_score_minutes_list, bins=range(0, 90, 10), ax=ax[0])
    sns.histplot(x=rival_score_minutes_list, bins=range(0, 90, 10), color='orange', ax=ax[1])

    ax[0].set_title(f'{team_to_study}scoring actions in favour')
    ax[1].set_title(f'{team_to_study}scoring actions against')

    plt.show()

    return 

#Tries and penalties in time
def tries_to_histogram(team_df, team_to_study):

    for action in ['E', 'PC']:

        #Create a figure
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        team_tries_minutes_list = []
        rival_tries_minutes_list = []

        #Run through every column
        for i in range(1, 27):

            #Get the tries minutes for team
            team_tries_minutes_series = team_df[team_df[f'scoreboard_{i:02d}_type'] == action][f'scoreboard_{i:02d}_minute']
            team_tries_minutes_list = team_tries_minutes_list + team_tries_minutes_series.tolist()

            try:
                #Get the tries minutes for rival
                rival_tries_minutes_series = team_df[team_df[f'scoreboard_rival_{i:02d}_type'] == action][f'scoreboard_rival_{i:02d}_minute']
                rival_tries_minutes_list = rival_tries_minutes_list + rival_tries_minutes_series.tolist()

            except KeyError:
                pass

        #team_df[team_df[y_axis_columns] == 'E']['scoreboard_01_type'].head()
        sns.histplot(x=team_tries_minutes_list, bins=range(0,90,10), ax=ax[0])
        sns.histplot(x=rival_tries_minutes_list, bins=range(0,90,10), color='orange', ax=ax[1])

        if action == 'E':
            ax[0].set_title(f'{team_to_study}tries in favour')
            ax[1].set_title(f'{team_to_study}tries against')
        elif action == 'PC':
            ax[0].set_title(f'{team_to_study}penalties in favour')
            ax[1].set_title(f'{team_to_study}panlties against')
        

        plt.show()

    return 

def season_scoreboard(team_df):

    team_df['rival_scoreboard_points(-)'] = team_df['rival_scoreboard_points'] * -1

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(x='date', y='scoreboard_points', data=team_df, errorbar=None)
    ax = sns.barplot(x='date', y='rival_scoreboard_points(-)', data=team_df, color='orange', errorbar=None)

    # Agregar los valores encima de cada columna
    for p in ax.patches:
        ax.annotate(
            f'{abs(int(p.get_height()))}',  # Texto a mostrar (el valor de la barra)
            (p.get_x() + p.get_width() / 2., p.get_height()),  # Posición (x, y)
            ha='center',  # Alineación horizontal
            va='center',  # Alineación vertical
            xytext=(0, 10),  # Desplazamiento del texto (x, y)
            textcoords='offset points'  # Tipo de coordenadas para el desplazamiento
        )

    plt.xticks(ticks=range(len(team_df)), labels=team_df['rival_team'], rotation=90)
    plt.xlabel('Equipo rival')
    plt.ylabel('Puntos')
    plt.title('Puntos a favor y en contra por partido')
    plt.axhline(0, color='black', linewidth=0.8)  # Línea horizontal en y=0

    plt.show()

    return 

#Tactical substitutions histograms in time
def subs_to_histogram(team_df, n_matches):
    #Create 
    team_subs_minutes_list = []

    #Run through every column. Every match permits up to 8 substituions per team
    for i in range(1, 9):

        try:
            #Get the tries minutes for rival
            subs_minutes_series = team_df[team_df[f'substitutions_{i:02d}_type'] == 'CT' ][f'substitutions_{i:02d}_minute'][-n_matches:]
            team_subs_minutes_list = team_subs_minutes_list + subs_minutes_series.tolist()

        except KeyError:
            pass

    #Create a histogram for the points made by the team
    sns.histplot(x=team_subs_minutes_list, bins=range(0, 90, 10))
    plt.show()

    return 