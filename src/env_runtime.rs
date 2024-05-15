use candle_core::Device;
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