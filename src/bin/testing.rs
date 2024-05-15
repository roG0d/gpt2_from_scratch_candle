#[path = "../env_runtime.rs"]
mod env_runtime;

use candle_core::{Device, Result, Tensor};
use candle_nn::{embedding, loss::cross_entropy, Embedding, Module, VarBuilder, VarMap, ops::softmax};
use ndarray::Array3;
use std::env;
use dotenv::dotenv;

use crate::env_runtime::Env;


//cargo run --bin testing
fn main(){   
  
  let env_runtime = Env::new();

    
    // .env
    let debug_var = env::var("DEBUG").unwrap();
    println!("{}",debug_var);
    // Candle
    let d1: [u32; 3] = [1u32, 2, 3];
    let d1_tensor: Tensor = Tensor::new(&d1, &env_runtime.device).unwrap();
    println!("dim1 tensor: {:?}", d1_tensor.to_vec1::<u32>().unwrap());
    
    // Basic GETs operations
    let d2: [[f32; 3]; 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let d2_tensor = Tensor::new(&d2, &env_runtime.device).unwrap();
    println!("dim2 tensor to vec: {:?}", d2_tensor.to_vec2::<f32>().unwrap());
    println!("dim 2 shape: {:?}", d2_tensor.shape());
    println!("get dim 0 index 0: {:?}", d2_tensor.get_on_dim(0, 0).unwrap());
    println!("get dim 0 index 0 first element: {:?}", d2_tensor.get_on_dim(0, 0).unwrap().get(0).unwrap());
    println!("get dim 1 index 0: {:?}", d2_tensor.get_on_dim(1, 0).unwrap());
    println!("get dim 0 index 0 first element: {:?}", d2_tensor.get_on_dim(1, 0).unwrap().get(0).unwrap());

    // i operator
    let d3_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    println!("Array3d data: {:?}", d3_data);
    let d3_array = Array3::from_shape_vec((3, 2, 2), d3_data).unwrap();
    println!("Array3d: {:?}", d3_array);
    // let d3_tensor = 
    // let d3i = d3_ar.i((0, generated_ids_cond_length - 1, ..))?;
    // Softmax testing
    println!("get dim 0 index 0 vector: {:?}", d2_tensor.get_on_dim(0, 0).unwrap().to_vec1::<f32>().unwrap());
    let probs = softmax(&d2_tensor, 1).unwrap();
    // Sofmax creates a  probability distribution from 0 to 1 with every element dim-wise -> in a Tensor(2,2) dim=0 column-wise, dim=1 row-wise
    println!("get dim 0 index 0 vector after softmax: {:?}", probs.get_on_dim(0, 0).unwrap().to_vec1::<f32>().unwrap());

    //Ndarray
    let d3_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    println!("Array3d data: {:?}", d3_data);
    let d3_array = Array3::from_shape_vec((3, 2, 2), d3_data).unwrap();
    println!("Array3d: {:?}", d3_array);
}
