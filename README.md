# rugby-tipsterbot

A machine learning project to try and predict the results of rugby matches.

The basic approach used initially will mirror the approach used here:
http://www.academia.edu/3165997/Artificial_Intelligence_in_Sports_Prediction

## Data

Data was scraped initially from espn.co.uk/rugby.  Code for the scraping is [here](https://github.com/lfyorke/rugby-webscraper) and fetches full stats for matches on a player by player basis.  For now the project only requires the result of each match and the scores for each team, so we can use the same dataset and just aggregate it to get what we need.

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

##  Method
The feature set values were calculated for each team for each round of competition. The MLP was trained using all examples from previous rounds, going back to the previous season if necessary, and re-trained after each round. Predictions were made for the current round by using the MLP to calculate an output value for each team based on that team’s feature set. An output value of close to one for a particular team indicated a high level of confidence that the team was going to win their upcoming match, and an output value closer to zero indicated a lower confidence level.The output values for the two teams competing in each game were calculated and the team which had the highest output
 
value (i.e., the highest confidence that the team would be victorious) was taken as the predicted winner (or tip) for that match. Success rates were then calculated as the proportion of tips for which the predicted winner matched the actual winner

## Success

TBD!