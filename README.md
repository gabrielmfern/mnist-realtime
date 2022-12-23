# MNIST Real-time

A very simple Real-time application that you can draw that then goes through an [Intricate](https://github.com/gabrielmfern/intricate) Model trained with the MNIST example.

## Arcitecture

The arcitecture for this model is very simple and can its layers can be written with the following Intricate code:

```rust
let mut mnist_model: Model = Model::new(vec![
    Conv2D::new((28, 28), (3, 3)),
    ReLU::new(26 * 26),

    Dense::new(26 * 26, 10),
    SoftMax::new(10),
]);
```

This though is not shown in the code here since it is loaded using [savefile](https://github.com/avl/savefile).

## Some cool images

![Drawing a two](https://github.com/gabrielmfern/mnist-realtime/blob/main/two-test.png?raw=true)
![Drawing a three](https://github.com/gabrielmfern/mnist-realtime/blob/main/three-test.png?raw=true)

## Usage

To use this you will need SDL and when building you will need SDL2 libs as well as SDL2_image's lib.

When pressing `CTRL-Z` the drawn image will be deleted.
When clicking it will start being drawn into the window and the model will be predicting in real-time to the terminal.
