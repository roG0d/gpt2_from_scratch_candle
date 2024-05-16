use candle_core::{Device, Tensor};
use std::{env, fmt};
use dotenv::dotenv;

#[derive(Debug)]
pub struct Env {
    pub debug: bool,
    pub device: Device,
}

impl Env {
    pub fn new() -> Self {
        dotenv().ok();
    
        Self {

            debug : match env::var("DEBUG").unwrap().as_str(){
                "true" => true,
                _ => false,
            },
        
            device : match env::var("DEVICE").unwrap().as_str(){
                "&Device::Cpu;" => Device::Cpu,
                _ => Device::new_cuda(0).unwrap(),
            },

        }
    } 
}

impl fmt::Display for Env {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Runtime environment:")?;

        writeln!(f, "DEBUG: {}", env::var("DEBUG").unwrap().as_str())?;
        writeln!(f, "DEVICE: {}", env::var("DEVICE").unwrap().as_str())?;

        Ok(())
    }
}

// sample_multinomial not implemented in candle -> https://github.com/jeroenvlek/gpt-from-scratch-rs/blob/main/src/sampling.rs
use rand::{distributions::Distribution, thread_rng};
pub fn sample_multinomial2d(logits: Tensor, device: &Device) -> candle_core::Result<Tensor> {
    let mut rng = thread_rng();
    let dim: usize = logits.shape().dims2().unwrap().0;
    let mut data:Vec<u32> = Vec::with_capacity(dim);
  
    for i in logits.to_vec2::<f32>().unwrap(){
      let distribution = rand::distributions::WeightedIndex::new(i).unwrap();
      let next_token = distribution.sample(&mut rng) as u32;
      data.push(next_token);
    }
  
    Ok(Tensor::from_vec(data, (dim,1), device).unwrap())
  }