# rugby-tipsterbot

A machine learning project to try and predict the results of rugby matches.

The basic approach used initially will mirror the approach used here:
http://www.academia.edu/3165997/Artificial_Intelligence_in_Sports_Prediction

## Data

Data was scraped initially from esnp.co.uk/rugby.  Code for the scraping is [here](https://github.com/lfyorke/rugby-webscraper) and fetches full stats for matches on a player by player basis.  For now the project only requires the result of each match and the scores for each team, so we can use the same dataset and just aggregate it to get what we need.

Future versions of the project could use more features derived from the player by player data.


## Approach

The initial approach will be a back propagated neural network with a simplistic set of features.

## Features

| Feature                            | Definition |
|------------------------------------|------------|
| Points For                         |The total points scored so far by the team |
| Points Against                     |The total points conceded so far by the team |
| Overall Performance                |The teams performance so far, 2 points for a win, 1 for a draw and 0 for a loss.  Performance is the sum of these values.|
| Home Performance                   |As with overall performance but home games only|
| Away Performance                   |As with overall performance but away games only|
| Performance in Previous Game       |The teams performance in their most recent game|
| Performance in Previous n games    |The teams performance in their most recent n games|
| Team Ranking                       |The teams position in the league ladder sorted by overall performance|
| Points for in Previous n games     |Points scored in previous n games|
| Points against in previous n games |Points scored in previous n games|
| Location                           |Boolean indicating whether the team is home or away|

##  Results

The intention is to measure success by comparison to actual results of matches.