# DeepDendro
![logo.jpg](media%2Flogo.jpg)

### Prerequisites
- *Eigen* library
  - [Download link](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip)
  - Move it to the path ```/usr/local/include/eigen3``` (or create a symbolic link):
  
    ```sudo ln -s $INSTALLATION_PATH$ /usr/local/include/eigen3```
  

- C++ 20



### Props
- ```indicators``` [library](https://github.com/p-ranav/indicators)


## Usage example


![carbon](https://user-images.githubusercontent.com/92575094/230293711-dcc58672-6d19-46ef-8f40-2992b89e742e.png)


![itsgift](https://user-images.githubusercontent.com/92575094/230293555-980fba42-5c51-461b-8496-6e851bdb3aa2.gif)

# Features
## Convolutional layers

### 2D

![conv2d.png](media%2Fconv2d.png)

## 3D

```c++
Convolutional3D conv3d{
        N_Filters,
        {3, 3, 1},
        activation::relu,
        {28, 28, 1}};
```
To better understand the dimensions, one can use the ```.print_structure()``` method.
For example, here the output of the layer would be a Tensor with shape $(26, 26, N_Filters)$.

```c++
conv3d.print_structure();
```



## Pooling layers


### 2D

```c++
Shape input_shape{4, 16};
Shape grid{2, 2};
Shape stride{2, 2};

MaxPool2D maxPool2D{input_shape, grid, stride};
```

### 3D

```c++
Shape input_shape{20, 20, 16};
Shape grid{2, 2, 1};
Shape stride{2, 2, 1};

MaxPool3D maxPool3D{input_shape, grid, stride};
```
This is the example of a 3D Max Pooling layer. The output of the layer would be a Tensor with shape $(10, 10, 16)$.
The pooling layers are used to decrease the dimensions of the input, and to reduce the number of parameters in the network.

