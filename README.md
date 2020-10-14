# DaisyGANv5
Technically a generative adversarial network anymore. 

## Example Usage
- ```./cfdgan retrain <optional file path>```
Train the network from the provided dataset.

- ```./cfdgan best```
This will iterate each optimiser 6 times outputting the RMSE in-order to illustrate the optimiser which produces the best RMSE on the current dataset.

- ```./cfdgan "this is an example scentence"```
Get a percentage of likelyhood that the sampled dataset wrote the provided message.

- ```./cfdgan rnd```
Get the percentage of likelyhood that the sampled dataset wrote a provided random message.

- ```./cfdgan ask```
A never ending console loop where you get to ask what percentage likelyhood the sampled dataset wrote a given message.

- ```./cfdgan gen <optional max error>```
The brute-force random message/quote generator.

- ```./cfdgan```
Bot service, will digest the botmsg.txt and botdict.txt every x messages and generate a new set of quotes.
