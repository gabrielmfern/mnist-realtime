# MNIST Real-time

A very simple Real-time application that you can draw that then goes through an [Intricate](https://github.com/gabrielmfern/intricate) Model trained with the MNIST example.

## Arcitecture

The arcitecture for this model is very simple and the Model can be written with the following Intricate code:

```rust
use intricate::Model;
use intricate::layers::{Conv2D, Dense};
use intricate::layers::activations::{ReLU, SoftMax};

let mut mnist_model: Model = Model::new(vec![
    Conv2D::new((28, 28), (3, 3)),
    ReLU::new(26 * 26),

    Dense::new(26 * 26, 10),
    SoftMax::new(10),
]);
```

This though is not shown in the code here since it is loaded using [savefile](https://github.com/avl/savefile).

## Training

To know how it was trained checkout the acutal training code disposed in the Intricate MNIST 
[example](https://github.com/gabrielmfern/intricate/blob/main/examples/mnist/main.rs).

## Usage

![Testing the MNIST Real-time](./testing.gif)

To use this you will need SDL and when building you will need SDL2 libs as well as SDL2_image's lib.

When pressing `CTRL-Z` the drawn image will be deleted.

When clicking inside the window you will start to draw into 
the window and the model will be predicting in real-time to the terminal.
