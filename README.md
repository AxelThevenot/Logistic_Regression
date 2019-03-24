[WORK IN PROGRESS...]

## Logistic regression principle
The logistic regression is a binomial regression model. As every binomial regression, it tends to model a mathematical model according to a vector of observations, which are the variables or parameters. The logistic regression seeks to find an indicator variables such as win/fail, alive/dead according to the parameters. Like other regression models, it is used both to predict a phenomenon and to explain it. The logistic regression is commonly used in Machine Learning and is also one of the simpliest neuron in Deep  Learning applications. The logistic regression is very close to the [linear regression](https://github.com/AxelThevenot/Linear_Regression). Contrary to the linear regression, where the variable to explain/predict is quantitative, the linear regression is used when the variables to explain/predict is qualitative.

![logistic regression by gradient descent](/src/logistic_regression.gif)

## Logistic regression application

The logistic regression is used to predict a variable, which is not quantitative. It is qualitative. To numerically convert the qualitative variable, it uses the sigmoid function, which returns a squashed value between 0 and 1. In other word, the sigmoid function is used to give a probability a sample has to possess a distinct value of the qualitative variable. Take the example of a model with only one parameter, notate `z`, and we want to predict if a sample is healthy/sick (refer to the graph below to a better understanding). The logistic regression aims to find the best weight according to `z` to fit the best the reality by the sigmoid value.
That way, the sigmoid function will return a value between 0 and 1, which will be the probability for a given point to be healthy.
So for a sample of a `z` parameter value, `0 < σ(z) < 1` and :
* `σ(z) > ½` →  the point is classified as healthy.
* `σ(z) < ½` →  the point is classified as healthy.
* `σ(z) = ½` →  the classification is not possible : choose arbitrarily a value.

![sigmoid](/src/sigmoid.png)

The aim of the logistic regression is to minimize the cost function, which is calculating an error like value. Indeed, we can not mathematically calculate an error with a qualitative value. So it uses the `h` squashed value and compares it to the `y` value, which correspond to the true value of the quantitative variable we want to predict. 

To sum up, the logistic regression uses a sigmoid function to give a probability of a sample to have a particular qualitative value according to its parameters. A weigth is associated to each parameter to minimize the cost function, which indicates how much the regression tends to fit the training samples.

The weight will tell how smooth is the slope.

A bias value is added to the parameters and associated weights. It allows to shift the sigmoid function to the left or right. 


![maths logistic](/src/logistic.png)

## Gradient descent : Introduction to Deep Learning

To make the logistic regression fit the best with the training set, it works step by step. A step is named an epcoh. For each epoch, the weights associated to the parameters are updated according to their influence on the current cost function value. It seeks to reach the global minimum of the cost, which correspond to the best fit with the training samples. In another words, it needs to calculate the partial derivative of the cost function by each weight. This way, it will know what is the slope value. Visually speaking, a positive value of the slope means a left shift is needed and vice-versa. To still speak visually, the cost function is a bowl and the goal is to reach its bottom, which is the best fitting spot.
So we had to calculate the gradient (partial derivatives of the cost function by the weights).

![maths gradient](/src/gradient.png)

As we now know how to calculate the gradients, we have to minimize the cost function value. To ensure this, an updates of the weight values is required by substract their associated gradients. 

![maths gradient descent](/src/gradient_descent.png)

On one hand, if you set a learning rate too low, learning will take too long.

On the other hand, if you set a learning rate too high, the variable's value jumps randomly whitout reaching the bottom of the cost function.

The aim is therefore to choose (experimentally most of the time) a learning rate that is neither too high nor too low.

![learning rate](/src/learning_rate.png)


## Example of French students

To illustrate the logistic regression algorithm I will take an easy example. Easy means with only 2 parameters. This way, we will be able to display outputs on a graph without reducing the dimensions. It will be more pleasant to look at and therefore to interpret.

In this case, we have a dataset of average mark in maths and French of 300 French students during the year 2018. We also know what final exam they passed (Assuming they took only maths and French final exams at the end of the year and that we know the average marks only for those students, who had passed at least one of the exams). The goal of the logistic regression here will be to predict who will pass or not both exams according to its maths and french average mark.
The dataset is a .csv file and its associated spreadsheet is represented below. The last column is for the label. The label is for the classification : `0`, `1` and `2` respectively mean pass the French exam, pass the maths exam and pass both.


Of course a simple logistic regression can not classify into 3 differents categories as it only output a probability to belong to one category. So to predict we will have to modify the label (`0`, `1` and `2`) :
* `0` and `1` become `0`
* `2` become `1`
So `0` now means to fail at least of the two exam and `1` to pass both.

![Dataset](/src/dataset.png)

To that end, we could use the [One vs All](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/one-vs-all) process. It simply use the logistic regression algorithm to test the probability for a sample to be on each category. The predicted categroy is therefore the one with the highest probability.

## Pseudo Code

Gradient descent method for logistic regression

```
Give to the weigths 'w' and to the bias 'b' a starting value (arbitrarily or randomly)

Choose the learning rate 'L'

While the minimum chosen cost or the number of maximum epoch chosen are not reaching
    
    Calculate the error (sum of each point's error)
    
    Calculate grad(w) for each weight and grad(b)
    
    Update w as w = w - grad(w) * L for each weight
    
    Update b as b = b - grad(b) * L
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
* Features (parameters) and label (category)
* Weights and bias
* Gradient descent process
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

We need to load the dataset. The function `loadDataset()` takes the .csv file name as argument. It removes the headers first. Then it returns an matrix containing feature's values and a label array.

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

The cornerstone of the logistic regression is the sigmoid function. So we start by its implementation.

```python
def sigmoid(z):
    """
    Sigmoid function
    :param z: input of the sigmoid function
    :return: sigmoid calculation
    """
    return 1 / (1 + np.exp(-z))
```

Then the `step_logistic_gradient()` calculate the error, the weights an bias gradients. Finally it updates the weights and bias values according to their gradient and to the learning rate chosen in the variable's region. It returns the updates weights and bias. 

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

At last we create the `logistic_gradient()` function, which will call the `step_logistic_gradient()` while the epoch number `MAX_EPOCH` is not reached. 

This function is a little special as it will be called by the matplotlib.animation object. That's why there is the variable `frame_number` that I will not explain in this page. In my program, we will activate the logistic regression method by pressing the Enter key. Only when the key will be pressed, the `logistic_gradient()` will run the `step_logistic_gradient()` function.

As you may have seen, the `display()` is not written yet. I addition to that, the function, which links the Enter key to the start is also not written. I will be next.


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

That is the time to add the display() function to render the algorithm. It only shows the samples values and waits for the Enter key too to display the logistic regression curve.

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
