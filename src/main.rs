mod gpt2;
mod bigram;
mod env_runtime;

use candle_core::{Device, Result, Tensor};

use crate::gpt2::{get_batch, load_data};
use crate::bigram::{Bigram};

static DEVICE: &Device = &Device::Cpu; // CPU Device
// static DEVICE: &Device = &Device::new_cuda(0).unwrap(); // GPU Device

fn main() -> Result<()> {

    // Load data 
    // B = 21, T = 85, C = 91
    let (train_data, _val_data) = load_data();
    let(x,y) = get_batch("train", train_data);
    // println!("X tensor: {:?}", x.to_vec2::<u32>().unwrap());

    //println!("x[0] first sequence: {:?}", x.to_vec2::<u32>().unwrap()[0]);
    //println!("y[0] first sequence: {:?}", y.to_vec2::<u32>().unwrap()[0]);
    //println!("{:?}", y);


    // Bigram loss function
    // vocab_size = n of unique tokens
    let bigram_model = Bigram::new(91).unwrap();
    //let (_logits, loss) = bigram_model.forward(&x, Some(&y));
    //println!("loss: {:?}", loss);
    bigram_model.generate(&x,1);
     
    Ok(())
}