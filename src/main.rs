mod hello_ml;
mod gpt2;
mod bigram;
use candle_core::{Device, Result, Tensor};

use crate::gpt2::{get_batch, load_data};
use crate::bigram::{Bigram};

fn main() -> Result<()> {
    /* hello_ml
    // Use Device::new_cuda(0)?; to use the GPU.
    let device = Device::cuda_if_available(0)?;

    let weight = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (100, ), &device)?;
    let first = hello_ml::Linear{weight, bias};
    let weight = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (10, ), &device)?;
    let second = hello_ml::Linear{weight, bias};
    let model = hello_ml::Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    // Inference on the model
    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
     */

    // Load data 
    // B = 21, T = 85, C = 91
    let (train_data, _val_data) = load_data();
    let(x,y) = get_batch("train", train_data);
    // println!("X tensor: {:?}", x.to_vec2::<u32>().unwrap());

    println!("x[0] first sequence: {:?}", x.to_vec2::<u32>().unwrap()[0]);
    println!("y[0] first sequence: {:?}", y.to_vec2::<u32>().unwrap()[0]);

    println!("{:?}", y);


    // Bigram loss function
    // vocab_size = n of unique tokens
    let bigram_model = Bigram::new(91).unwrap();
    let (_logits, loss) = bigram_model.forward(&x, Some(&y));
    println!("loss: {:?}", loss); 
    bigram_model.generate(&x,1);
     
    Ok(())
}