---
title: Python Lambda Functions - how and when to use them - introduction tutorial
author: Amine M'Charrak
layout: post
date: 2020-03-15 04:00:00
---

#### In this tutorial we will explore what lambda functions are and how to use them.

A Python lambda is just the pythonic way of a function. A lambda function has the following basic syntax:

```python
lambda arguments : expression
```

We are already familiar with the "classical" way of defining functions in Python such as:


```python
def square(x):
    return (x*x)

def square_vals(list_of_vals):
    for (idx, val) in enumerate(list_of_vals):
        list_of_vals[idx] = square(val)
    return (list_of_vals)
```

Which we then can use as follows:


```python
vals = [1,2,3,4,5]

squared_val = square(vals[-1])
print(squared_val)

squared_vals = square_vals(vals)
print(squared_vals)
```

    25
    [1, 4, 9, 16, 25]


Now, lambda is just another way of defining such a function. The lambda function can take any number of inputs/arguments but it can only have one expression. Lambda returns the evaluated expression for the arguments it received.

Now, let us define the same functions as above but using the lambda method:


```python
lambda_square = lambda x : x*x
```

Next we need to define a lambda function which takes as argument/input a list - and then operates on each item in that list. For that, we will have to make use of the python built-in function: map. Map is has two arguments:

1. a *function* and
2. a *sequence/list*

What it then does is to apply/call the function on every item in the list. So we do the following to square all items in our list using our already defined lambda function for squaring a single value:


```python
vals = [1,2,3,4,5]

squared_vals = list( map(lambda_square, vals) )
# we apply list() on the output of the map function because compared to Python 2, in
# Python 3 map() returns a map object and not a list (as it was previously in Python 2).
print(squared_vals)
```

    [1, 4, 9, 16, 25]


The last questions we need to answer is:

    Why do we even need lambda functions if we can do the same with regular functions as above.

Lambda function are useful, when we need to define a function that we are not planning to use more than once, so in effect we think of it as a one-off function. Such a one-time-use function is formally knonw as: **anonymous function**.

Finally lets see how we can

1. define a lambda function with *more than one argument*
2. define a lambda function with *zero arguments*


```python
#1. MORE THAN ONE ARGUMENT
f1_is_right_angled_triangle = lambda a,b,c : (c**2 == a**2 + b**2)
```


```python
# lets test two triangles
# Triangle 1 with sides a=3,b=4,c=5
print(f1_is_right_angled_triangle(a=3,b=4,c=5))
# Triangle 1 with sides a=4,b=5,c=6
print(f1_is_right_angled_triangle(a=4,b=5,c=6))
```

    True
    False



```python
#2. NO/ZERO ARGUMENTS
f2_false = lambda : 1==3
```


```python
print(f2_false())
```

    False


That's it. I hope these examples have helped you to understand how we can use lambda functions in Python and when they can be helpful. If you have questions or ideas, please feel free to drop me a message.
