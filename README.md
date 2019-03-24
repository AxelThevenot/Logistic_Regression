[WORK IN PROGRESS...]

## Logistic regression principle


![logistic regression by gradient descent](/src/logistic_regression.gif)

## Logistic regression application
![sigmoid](/src/sigmoid.png)


![maths logistic](/src/logistic.png)


## Gradient descent : Introduction to Deep Learning

![maths gradient](/src/gradient.png)
![maths gradient descent](/src/gradient_descent.png)

On one hand, if you set a learning rate too low, learning will take too long.

On the other hand, if you set a learning rate too high, the variable's value jumps randomly whitout reaching the bottom of the cost function.

The aim is therefore to choose (experimentally most of the time) a learning rate that is neither too high nor too low
![learning rate](/src/learning_rate.png)
## Example of French students


## Pseudo Code

Gradient descent method for logistic regression

```

```

## Let's start with python

### Import

Firstly we need to import csv to extract the data from a .csv file, numpy to deal with array or matrix and we need to import matplotib for the render.

```python
import csv  # to deal with .csv file
import numpy as np  # to deal with array and matrix calculation easily
import matplotlib.pyplot as plt  # to plot
import matplotlib.animation as animation  # to animate the plot
from mpl_toolkits.mplot3d import Axes3D  # for 3D plot
import matplotlib.patches as mpatches  # to add legends easily
import matplotlib.font_manager as font_manager  # to change the font size
```

### Variables

As usual I made a region to change the variables to an easier understanding. Variables that can be manually changed are those, which are in uppercases. They are classed as variables for : 
* 
* 
* 
* Rendering the algorithm output

```python
X = None  # features matrix
Y = None   # label array

# weights and bias for logistic regression
weights, bias = None, None

# logistic regression with gradient descent
LEARNING_RATE = 0.05
MAX_EPOCH = 20000  # stop the run at MAX_EPOCH
DISPLAY_EPOCH = 500  # update the plot each DisPLAY_EPOCH epoch
epoch = 0  # epoch counter
cost = 0  # actual cost of the logistic regression

# to display the graph of points
fig = plt.figure(1, figsize=(8, 4.5))  # to plot
ax = fig.add_subplot(111, projection='3d')  # the axes
ani = None  # to animate
started = False  # variable to indicate if the run is started or not
legends = {1: 'pass the two exams', 0: 'does not pass the two exams'}  # legends
color = {1: '#2F9599', 0: '#999999'}  # category's color for points

```

### Load dataset



```python
def load_dataset(filename):
    """
    load a dataset from a filename
    :param filename: filename of the dataset
    :return: X features matrix, Y label array  (as np.array)
    """

    # read the file
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        # get header from first row
        headers = next(reader)  # if headers are needed
        # get all the rows as a list
        dataset = list(zip(*reader))  # transpose the dataset matrix

        # change the label as 0 or 1 values
        dataset = [dataset[3], dataset[2], [1 if label == '2' else 0 for i, label in enumerate(dataset[4])]]

        # transform data into numpy array
        dataset = np.array(dataset).astype(float)

        X = dataset[:-1].transpose()  # features matrix
        Y = dataset[-1].transpose()  # label array
        return X, Y
```


### Logistic regression


```python
def sigmoid(z):
    """
    Sigmoid function
    :param z: input of the sigmoid function
    :return: sigmoid calculation
    """
    return 1 / (1 + np.exp(-z))
```



```python
def step_logistic_gradient(X, Y, learning_rate):
    """
    One step of the gradient descent in logistic regression case
    :param X: features matrix
    :param Y: label array
    :param learning_rate: learning rate of interest
    :return: weights, bias, cost of the actual state
    """
    global weights, bias  # pick up the weights and the bias
    N = X.shape[0]  # number of training samples

    # calculation of the z value as z = weigths * x + b
    z = np.dot(X, np.hstack((weights, bias)))

    # calculation of the sigmoid value of z
    h = sigmoid(z)
    
    # calculation of the cost
    cost = sum(- Y * np.log(h) - (1 - Y) * np.log(1 - h))
    
    # Update gradients
    # gradient weights uptdate rule dJ/dweigth_i = 1/n * sum((h(x_i) - y_i) * x_i)
    # gradient bias uptdate rule dJ/dbias = 1/n * sum((h(x_i) - y_i))
    error = h - Y
    gradient = np.divide(np.dot(error, X), N)

    # Update weights
    # weights update rule weigth_i := weigth_i - learning_rate * (dJ/dweigth_i)
    # bias update rule bias := bias - learning_rate * (dJ/dbias)
    weights -= learning_rate * gradient[:-1]
    bias -= learning_rate * gradient[-1]

    return weights, bias, cost
```

```python
def logistic_gradient(frame_number):
    """
    Run the logistic regression
    """
    if started:  # waiting for start
        global weights, bias, cost, epoch  # pick up global variables

        # Initialize the weigths and the bias with a 0 value (can be randomized too)
        weights = np.zeros(X.shape[1])
        bias = np.zeros(np.shape(1))

        # add a 1 variable to add the bias on the equation
        X_with_bias = np.hstack((X, (np.ones((X.shape[0], 1)))))

        while epoch < MAX_EPOCH:  # run the logistic regression
            # update continuously the weigths and the bias
            weights, bias, cost = step_logistic_gradient(X_with_bias, Y, LEARNING_RATE)

            if epoch % DISPLAY_EPOCH == 0: # update the plot
                display()
            epoch += 1  # update the counter
```

### Display



```python
def display():
    """
    plot the points of the training set
    """
    global X, Y  # pick the the features and the labels

    ax.clear()  # clear the plot before the update

    # set the title, the axes and the legends
    ax.set_title('Passing final exams according to subject\'s average')
    ax.set_xlabel('Maths')
    ax.set_ylabel('French')
    ax.set_ylim(4, 19)  # only for a better view
    plt_legends = [mpatches.Patch(color=color[key], label=legends[key]) for key, _ in enumerate(legends)]

    # plot the points of the training set
    for i, sample in enumerate(X):
        ax.scatter(sample[0], sample[1], Y[i], c=color[Y[i]])

    if started:  # waiting for start
        # The logistic regression is an area (infinity number of point)
        # So we will represent it line by line
        # It is a huge time-consuming process so put n as lower as possible
        n = 20  # number of line to plot

        # Create n different lines to render the logistic regression function[...]
        # [...] with a constant x value [...]
        lines = np.linspace(min(X.transpose()[0]), max(X.transpose()[0]), num=n)

        # [...] from the min to the max value of the second feature[...]
        min_y, max_y = min(X.transpose()[1]), max(X.transpose()[1])
        y_array = np.linspace(min_y, max_y, num=n)  # from min to max eah time
        # [...] with the sigmoid value height according to the features.

        # for each line to represent
        for _, x in enumerate(lines):
            x_array = np.full(n, x)  # create x constant array
            z_array = np.array([])  # to append with the sigmoid values

            for _, y in enumerate(y_array):  # append with the sigmoid value
                z_array = np.hstack((z_array, np.array(sigmoid(weights[0] * x + weights[1] * y + bias))))

            # plot the line
            ax.plot(x_array, y_array, z_array, c='#8800FF', alpha=0.2)

        # add a big legend to describe the state of the logistic regression
        label = 'Logistic Regression :\n'
        label += 'Equation : {0} * math + {1} * french + {2}\n'. \
            format(round(weights[0], 2), round(weights[1], 2), int(bias))
        label += 'Epoch : {0}\n'.format(epoch)
        label += 'Learning Rate : {0}\n'.format(LEARNING_RATE)
        label += 'Cost : {0}'.format(round(cost, 2))
        # add the created legend
        plt_legends.append(mpatches.Patch(color='#8800FF', label=label))

    # to have smaller font size
    font_prop = font_manager.FontProperties(fname='C:\Windows\Fonts\Arial.ttf', size=6)

    # Put a legend above current axis
    plt.legend(handles=plt_legends, loc='upper center', bbox_to_anchor=(0.5, 0.1), prop=font_prop, ncol=3)

    # then plot everything
    fig.canvas.draw()
```

Finally, implement the `key_pressed()` function to activate the algorithm when Enter key is pressed.

```python
def key_pressed(event):
    """
    To start to run the programme by enter key
    :param event: key_press_event
    """
    if event.key == 'enter':
        global started
        started = not started
```
### Run it ! 

```python
if __name__ == '__main__':
    X, Y = load_dataset('student.csv')  # load dataset

    display()  # first display to show the samples

    # connect to the key press event to start the program
    fig.canvas.mpl_connect('key_press_event', key_pressed)
    # to animate the plot and launch the gradient descent update
    ani = animation.FuncAnimation(fig, logistic_gradient)

    plt.show() # show the plot
```
