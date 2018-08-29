# A3C
Asynchronous actor critic method - Has been successfully tested on Pong. While based on some of the other implementations out there such as:
https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
and 
https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c
This version saves the LSTM cell states and uses the last cell state of one roll out to initialize the next.
This seemed to be not only helpfull but essential for making it work reliably on Atari games...
Learned to beat Pong in around 10 Hours on my 16 core machine
