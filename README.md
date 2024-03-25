# Esports-prediction
Contains the code and methodology behind my master's thesis "Machine learning-based methods for predicting esports' results.

The main goal of this project was to propose and examine different neural network architectures
and their performance when dealing with the problem of predicting esport events' results. 
The secondary goal was to investigate the impact of auxiliary information introduced to the data used in the process of
training those models.


In this repository, you will find the following files.

 # [DATA PREPROCESSING]
  1. "Player_statistics.py" file - A Python script that was used to generate data for model
  training. The output file contains non-aggregated player statistics without any auxiliary
  information.
  2. "Team_statistics.py" file - A Python script that was used to generate data for model training.
  The output file contains aggregated player statistics on the team level. The auxiliary
  information can be added to the resulting dataset by enabling flags set inside the script. A
  detailed description and documentation can be found inside the file.

#  [MODELS AND USE CASES]
  3. "ANN_code.ipynb" file - A Jupyter notebook containing all functions and structures used
  in the process of obtaining results for the proposed ANN networks, alongside the networks
  themselves. All of the functions are documented and an example use case is presented.
  4. "CNN_code.ipynb" file - A Jupyter notebook containing all functions and structures used
  in the process of obtaining results for the proposed simple CNN network, alongside the
  network itself. All of the functions are documented and an example use case is presented.
  5. "VGG_code.ipynb" file - A Jupyter notebook containing all functions and structures used
  in the process of obtaining results for the proposed VGG network, alongside the network
  itself. All of the functions are documented and an example use case is presented.


#  [SAMPLE DATA]
  Alongside those files, there are attached compressed files in the Data_samples folder  which
  need to be unpacked directly into the parent directory. Those files contain the following:

  1. "Player_statistics.csv" file - A CSV file containing the baseline non-aggregated player
  statistics that were used for model training.
  2. "Basic_team_statistics.csv" file - A CSV file containing the baseline aggregated team
  statistics that were used for model training.
  3. "Team_statistics_with_economy.csv" file - A CSV file containing the aggregated team
  statistics with the auxiliary economy data, that was used for model training.
  4. "Team_statistics_with_updates.csv" file - A CSV file containing the aggregated team
  statistics with the auxiliary data about game updates, that was used for model training.
  5. "Teams_statistics_with_roster_changes.csv" file - A CSV file containing the aggregated
  team statistics with the auxiliary data about roster changes, that was used for model training.
