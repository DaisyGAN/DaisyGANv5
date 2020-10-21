# DaisyGANv5
Technically not a generative adversarial network anymore. 

This version can randomly generate its learning parameters and then test to see how successful the generated weights were by doing a variance test against random data; it goes by the rule that if random strings fail at-least 70% of the time, then the weights are successful.

`main_revision.c` - this is a WIP edit where I attempt to improve performance over the original `main.c` so far I have added weight decay to little avail, batching of the backprop which had a small improvement of ~2 seconds per pass entire training pass.

## Example Usage
- ```./cfdgan retrain <optional file path>```
<br>Train the network from the provided dataset.

- ```./cfdgan best```
<br>This will iterate each optimiser 6 times outputting the RMSE in-order to illustrate the optimiser which produces the best RMSE on the current dataset.

- ```./cfdgan rndbest <optional 0-100 fail variance>```
<br>This will run the best function with random learning parameters within an acceptable range. You want the fail variance to be high as the test is against random strings, if the generation was good, random strings should fail most of the time but not all of the time.

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
