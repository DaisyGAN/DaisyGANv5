# DaisyGANv5
Technically not a generative adversarial network anymore. 

This version can randomly generate its learning parameters and then test to see how successful the generated weights were by doing a variance test against random data; it goes by the rule that if random strings fail at-least 70% of the time, then the weights are successful.

`main_revision.c` - this is a WIP edit where I attempt to improve performance over the original `main.c` so far I have added weight decay to little avail, batching of the backprop which had a small improvement of ~2 seconds per pass entire training pass. I had a try with some tanh approximations and I'ved changed the training method to use a cross-validation system, where 70% of the training data is used for training and the remaining 30% for testing the RMSE.

`[21:10:20]` - I have now allowed for a multi-process model to compete for the lowest fail variance when using `./cfdgan best`.

## Example Usage
- ```./cfdgan retrain <optional file path>```
<br>Train the network from the provided dataset.

- ```./cfdgan check```
<br>Chech the fail variance of the current weights.

- ```./cfdgan reset <optional fail variance lower limit>```
<br>Reset the current weights. The optional parameter allows you to set the minimum viable fail variance value for a set of computed weights, all weights below this value are discarded.

- ```./cfdgan best```
<br>Randomly iterate each parameter and recompute the weights until the best solution is found. This function is multi-process safe.

- ```./cfdgan bestset```
<br>Randomly iterate each parameter and recompute the weights outputting the average best parameters to `best_average.txt`. This function is multi-process safe.

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
