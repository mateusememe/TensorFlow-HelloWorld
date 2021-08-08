import tensorflow as tf
import numpy as np
from tensorflow import keras


'''
    Say hello to the "Hello, World" of machine learning
    
Consider the following sets of numbers. Can you see the relationship between them ?
X: |-1 |0 |1 |2 |3 |4 
Y: |-2 |1 |4 |7 |10|13

The relationship is:  Y = 3X + 1
'''
# Define and compile the neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Compiled the model
model.compile(optimizer='sgd', loss='mean_squared_error')

'''
optimizer:
    Next, the model uses the optimizer function to make another guess.
    Based on the loss function's result, it tries to minimize the loss.
    At this point, maybe it will come up with something like Y=5X+5.
    While that's still pretty bad, it's closer to the correct result 
    (the loss is lower).

loss:
    When the computer is trying to learn that, it makes
    a guess, maybe Y=10X+10. The loss function measures
    the guessed answers against the known correct answers
    and measures how well or badly it did.
'''

# Provide the data, with the relationship is => "Y = 3X + 1"
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Training the neural network
model.fit(xs, ys, epochs=668)

'''
You have a model that has been trained to learn the relationship between X and Y.
You can use the model.predict method to have it figure out the Y for a previously unknown X.
For example, if X is 10, what do you think Y will be?
'''

print("\n\nPredict: ",model.predict([11.0]))


'''
You might have thought 31, but it ended up being a little over. Why do you think that is?
Neural networks deal with probabilities, so it calculated that there is a very high probability that the relationship between X and Y is Y=3X+1,
but it can't know for sure with only six data points. The result is very close to 31, but not necessarily 31.
As you work with neural networks, you'll see that pattern recurring. You will almost always deal with probabilities
not certainties, and will do a little bit of coding to figure out what the result is based on the probabilities, particularly when it comes to classification.
'''