use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, loss::cross_entropy, Embedding, Module, VarBuilder, VarMap, ops::softmax};
use gpt2::Env;
use ndarray::Array3;
use rand::{distributions::Distribution, thread_rng};
use std::env;
use dotenv::dotenv;



//cargo run --bin testing
fn main(){   
  
  let env_runtime = Env::new();

    
    // .env
    let debug_var = env::var("DEBUG").unwrap();
    //println!("{}",debug_var);
    // Candle
    let d1: [u32; 3] = [1u32, 2, 3];
    let d1_tensor: Tensor = Tensor::new(&d1, &env_runtime.device).unwrap();
    //println!("dim1 tensor: {:?}", d1_tensor.to_vec1::<u32>().unwrap());
    
    // Basic GETs operations
    let d2: [[f32; 3]; 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let d2_tensor = Tensor::new(&d2, &env_runtime.device).unwrap();
    println!("dim2 tensor to vec: {:?}", d2_tensor.to_vec2::<f32>().unwrap());
    println!("dim 2 shape: {:?}", d2_tensor.get(0));
    println!("get dim 0 index 0: {:?}", d2_tensor.get_on_dim(0, 0).unwrap());
    println!("get dim 0 index 0 first element: {:?}", d2_tensor.get_on_dim(0, 0).unwrap().get(0).unwrap());
    println!("get dim 1 index 0: {:?}", d2_tensor.get_on_dim(1, 0).unwrap());
    println!("get dim 0 index 0 first element: {:?}", d2_tensor.get_on_dim(1, 0).unwrap().get(0).unwrap());

    /* i operator: Return a slice given an index.
      Given a 3x4 Tensor:
    - if the index is a scalar, it will return the row[index] Ej: i=0 -> [1.0, 2.0]
    - if the index is a tuple, it will return the row[column[index]] Ej: i=(0,0) -> 1.0
    */
    let d2_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let d2i_tensor = Tensor::from_vec(d2_data, (3,4), &env_runtime.device).unwrap();
    let d2i = d2i_tensor.i(1).unwrap();
    println!("Tensor2d i op: {:?}", d2i.to_vec1::<f32>());
    /* Given a 3x2x2 Tensor: 
    - if the index is a scalar, it will return the matrix[index] Ej: i=0 -> [[1.0, 2.0], [2.0, 3.0]]
    - if the index is a tuple, it will return the matrix[row[index]] Ej: i=(0,0) -> [1.0, 2.0]
    - if the index is a 3-tuple, it will reutn the matrix[row[column[index]]] Ej: i=(0,0,0) -> 1.0
    */
    let d3_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    //println!("Tensord 3d data: {:?}", d3_data);
    let d3_tensor = Tensor::from_vec(d3_data, (3,2,2), &env_runtime.device).unwrap();

    let d3i = d3_tensor.i((0,1,0)).unwrap();
    //println!("Tensor3d i op: {:?}", d3i.to_vec0::<f32>());
    
    // Softmax testing
    // println!("get dim 0 index 0 vector: {:?}", d2_tensor.get_on_dim(0, 0).unwrap().to_vec1::<f32>().unwrap());
    let probs = softmax(&d2_tensor, 1).unwrap();
    // Sofmax creates a  probability distribution from 0 to 1 with every element dim-wise -> in a Tensor(2,2) dim=0 column-wise, dim=1 row-wise
    // println!("get dim 0 index 0 vector after softmax: {:?}", probs.get_on_dim(0, 0).unwrap().to_vec1::<f32>().unwrap());

    //Ndarray
    let d3_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    // println!("Array3d data: {:?}", d3_data);
    let d3_array = Array3::from_shape_vec((3, 2, 2), d3_data).unwrap();
    //println!("Array3d: {:?}", d3_array);


    //Multinomial
    let samples = sample_multinomial2d(d2_tensor, env_runtime.device);
    //println!("samples!: {:?}", samples.unwrap().to_vec2::<u32>().unwrap())

}
pub fn sample_multinomial2d(logits: Tensor, device:Device) -> candle_core::Result<Tensor> {
  let mut rng = thread_rng();
  let dim: usize = logits.shape().dims2().unwrap().0;
  let mut data:Vec<u32> = Vec::with_capacity(dim);

  for i in logits.to_vec2::<f32>().unwrap(){
    let distribution = rand::distributions::WeightedIndex::new(i).unwrap();
    let next_token = distribution.sample(&mut rng) as u32;
    data.push(next_token);
  }

  Ok(Tensor::from_vec(data, (dim,1), &device).unwrap())
}
