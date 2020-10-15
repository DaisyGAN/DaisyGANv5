# DaisyGANv5
Technically a generative adversarial network anymore. 

## Example Usage
- ```./cfdgan retrain <optional file path>```
<br>Train the network from the provided dataset.

- ```./cfdgan best```
<br>This will iterate each optimiser 6 times outputting the RMSE in-order to illustrate the optimiser which produces the best RMSE on the current dataset.

- ```./cfdgan rndbest```
<br>This will run the best function with random learning parameters within an acceptable range.

- ```./cfdgan "this is an example scentence"```
<br>Get a percentage of likelyhood that the sampled dataset wrote the provided message.

- ```./cfdgan rnd```
<br>Get the percentage of likelyhood that the sampled dataset wrote a provided random message.

- ```./cfdgan ask```
<br>A never ending console loop where you get to ask what percentage likelyhood the sampled dataset wrote a given message.

- ```./cfdgan gen <optional max error>```
<br>The brute-force random message/quote generator.

- ```./cfdgan```
<br>Bot service, will digest the botmsg.txt and botdict.txt every x messages and generate a new set of quotes.
