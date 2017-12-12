# rugby-tipsterbot

A machine learning project to try and predict the results of rugby matches.

The basic approach used initially will mirror the approach used here:
http://www.academia.edu/3165997/Artificial_Intelligence_in_Sports_Prediction

## Data

Data was scraped initially from esnp.co.uk/rugby.  Code for the scraping is [here](https://github.com/lfyorke/rugby-webscraper) and fetches full stats for matches on a player by player basis.  For now the project only requires the result of each match and the scores for each team, so we can use the same dataset and just aggregate it to get what we need.

Future versions of the project could use more features derived from the player by player data.


## Approach

The initial approach will be a back propagated neural network with a simplistic set of features.

##  Results

The intention is to measure success by comparison to actual results of matches.