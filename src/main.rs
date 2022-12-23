use intricate::{Model, utils::{setup_opencl, opencl::DeviceType}};
use savefile::prelude::*;
use sdl2::pixels::PixelFormatEnum;
use simple::{self, MouseButton, Key, Rect};
use rayon::prelude::*;

fn convert_from_pixel_bytes_to_grayscale_pixel_matrix(bytes: Vec<u8>) -> Vec<f32> {
    bytes.par_iter()
        .enumerate()
        .filter_map(|(i, byte)| {
            if i % 3 == 0 {
                Some((byte, bytes[i + 1], bytes[i + 2]))
            } else {
                None
            }
        })
        .map(|(r, g, b)| (*r as f32 + g as f32 + b as f32) / 3.0)
        .collect()
}

fn compress_image_to_28_by_28(pixels: Vec<f32>) -> Vec<f32> {
    (0..784).into_par_iter().map(|output_index: usize| {
        let first_pixel_y = (output_index as f32 / 28.0).floor() as usize * 28;
        let first_pixel_x = (output_index * 28) % 784;
        (0..28).map(|local_y| {
            let pixel_y = first_pixel_y + local_y;
            (0..28).map(|local_x| {
                let pixel_x = first_pixel_x + local_x;
                let pixel_index = pixel_y * 784 + pixel_x;
                pixels[pixel_index]
            }).sum::<f32>()
        }).sum::<f32>() / 784.0 / 255.0 // divide by 255 to get numbers from 0 to 1
    }).collect()
}

fn main() {
    let opencl_state = setup_opencl(DeviceType::GPU)
        .expect("Não foi possível inicializar OpenCL!");

    let mut model: Model = load_file("mnist-model.bin", 0)
        .expect("Não foi possível carergar o modelo de MNIST com convolução!");
    model.init(&opencl_state).expect("Não foi possível inicializar modelo do MNIST!");

    let mut app = simple::Window::new("MNIST desenhar", 28 * 28, 28 * 28);

    while app.next_frame() {
        if app.is_mouse_button_down(MouseButton::Left) {
            let mouse_position = app.mouse_position();
            let rect = Rect::new(mouse_position.0, mouse_position.1, 56, 56);
            app.fill_rect(rect);
        } else if app.is_key_down(Key::LCtrl) && app.is_key_down(Key::Z) {
            app.clear();
        }

        let pixel_bytes = app.canvas.read_pixels(app.canvas.viewport(), PixelFormatEnum::RGB24)
            .expect("Não foi possível ler os pixels da tela");
        assert_eq!(pixel_bytes.len(), 784 * 784 * 3);
        let grayscale_full_image = convert_from_pixel_bytes_to_grayscale_pixel_matrix(pixel_bytes);
        assert_eq!(grayscale_full_image.len(), 784 * 784);
        let downscaled_image = compress_image_to_28_by_28(grayscale_full_image);
        assert_eq!(downscaled_image.len(), 28 * 28);

        // for y in 0..28 {
        //     for x in 0..28 {
        //         let index = y * 28 + x;
        //         print!("{:.1}", downscaled_image[index]);
        //     }
        //     print!("\n");
        // }

        model.predict(&vec![downscaled_image])
            .expect("Não foi possível propagar o modelo de MNIST a imagem comprimida");
        let per_digit_probability = model.get_last_prediction()
            .expect("Não foi possível pegar o output da últiam propagaçao do modelo de MNIST");
        assert_eq!(per_digit_probability.len(), 10);

        print!("\x1B[2J");

        println!("Probabilidade de ser cada digito: ");
        println!("    '0': {}", per_digit_probability[0]);
        println!("    '1': {}", per_digit_probability[1]);
        println!("    '2': {}", per_digit_probability[2]);
        println!("    '3': {}", per_digit_probability[3]);
        println!("    '4': {}", per_digit_probability[4]);
        println!("    '5': {}", per_digit_probability[5]);
        println!("    '6': {}", per_digit_probability[6]);
        println!("    '7': {}", per_digit_probability[7]);
        println!("    '8': {}", per_digit_probability[8]);
        println!("    '9': {}", per_digit_probability[9]);
    }
}
