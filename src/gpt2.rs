use candle_core::{Device,Tensor};
use rand::{distributions::Uniform, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use std::{collections::{HashMap, HashSet},  fs, mem::take};

const DEBUG:bool = true;

const BATCH_SIZE: usize = 21;  // usize for non-negative sequence counts
const BLOCK_SIZE: usize = 85;  // usize for context lengths
const MAX_ITERS: usize = 5000; // usize for iteration counts
const EVAL_INTERVAL: usize = 500; // usize for intervals
const LEARNING_RATE: f32 = 3e-4; // f32 for floating-point precision
const EVAL_ITERS: usize = 200; // usize for iteration counts
const N_EMBD: usize = 128;  // usize for embedding dimensions
const N_HEAD: usize = 6;    // usize for attention heads
const N_LAYER: usize = 6;   // usize for layer counts
const DROPOUT: f32 = 0.2;   // f32 for dropout probability


static DEVICE: &Device = &Device::Cpu; // CPU Device
// static DEVICE: &Device = &Device::new_cuda(0).unwrap(); // GPU Device


pub fn load_data() -> (Tensor, Tensor){

    let text = fs::read_to_string("./assets/input.txt").unwrap();

    // Unique characters that occur in the dataset
    let mut chars:Vec<char> = text.chars().into_iter().collect::<HashSet<char>>().into_iter().collect();
    chars.sort();
    if DEBUG {println!("Chars: {:?}",chars);}
    if DEBUG {println!("Chars size: {:}",chars.len());}


    // Encoder
    let mut encoder: HashMap<char, u32> = HashMap::new();
    for (i, c) in chars.iter().enumerate(){
        encoder.insert(*c, i as u32);
    }  
    if DEBUG {println!("Encode S: {:?}",encoder.get(&'S'));}
    
    // Decoder
    let mut decoder: HashMap<u32, char> = HashMap::new();
    for (i, c) in chars.iter().enumerate(){
        decoder.insert(i as u32, *c);
    }
    if DEBUG {println!("decode 38: {:?}",decoder.get(&38));}

    // Train and test splits
    let encoded_text:Vec<u32> = text.chars().map(|c|*encoder.get(&c).unwrap()).collect();
    let len_data = encoded_text.len();
    let n = (len_data as f64 * 0.9) as usize;

    let sliced_data:Vec<Vec<u32>> = encoded_text.chunks(n).map(|s| s.into()).collect();
    let len_train = sliced_data[0].len();
    let len_val = sliced_data[1].len();
    // GPU Tensors
    let train_data = Tensor::from_vec(sliced_data.iter().nth(0).unwrap().to_vec(), len_train, DEVICE).unwrap();
    let val_data = Tensor::from_vec(sliced_data.iter().nth(1).unwrap().to_vec(), len_val, DEVICE).unwrap();
    if DEBUG {println!("train data tensor: {:?}",train_data);}
    if DEBUG {println!("validation data tensor: {:?}",val_data);}
    // CPU Tensor
    //let data = Tensor::from_vec(encoded_text, len_et, &Device::Cpu).unwrap();
   (train_data,val_data)
}

pub fn get_batch(split: &str, data: Tensor)-> (Tensor,Tensor){
    //Random Seed (It would be nice to be a const but there isn't any native support)
    let seed:ChaCha8Rng = rand_chacha::ChaCha8Rng::seed_from_u64(1337);
    let data_vec: Vec<u32> = data.to_vec1().unwrap();
    // generate a small batch of data of inputs x and targets y
    // https://stackoverflow.com/a/48219147

    // Upper limit for the uniform distribution = length data - block size -> as then we are going to take [i:i+block_size] batches
    let upper_limit = data.shape().dims1().unwrap() - BLOCK_SIZE;
    // Range to get uniform samples from
    let range = Uniform::from(0..upper_limit);
    // Rand values which will be the initial indexes of the sequences, we take up to BATCH_SIZE
    let rand_values: Vec<usize> = seed.sample_iter(range).take(BATCH_SIZE).collect();
    if DEBUG {println!("Random indexes for batching: {:?}",rand_values);}

    // Creating X Tensor
    // X is a BATCH_SIZE vector with BLOCK_SIZE vectors -> a full batch of sequences
    // OPTIMIZATION: LLMs manage different size sequences by padding -> Fuse variable lenght sequences with prepacking https://twitter.com/siyan_zhao/status/1780288750624612850
    let x: Vec<Vec<u32>> = rand_values.iter().map(|v| data_vec[*v.. *v+BLOCK_SIZE].to_vec()).collect();
    let array_x:[[u32;BLOCK_SIZE];BATCH_SIZE] = create_array(&x).unwrap();
    let tensor_x:Tensor = Tensor::new(&array_x,  DEVICE).unwrap();

    // Creating Y Tensor
    let y: Vec<Vec<u32>> = rand_values.iter().map(|v| data_vec[*v+1.. *v+BLOCK_SIZE+1].to_vec()).collect();
    let array_y:[[u32;BLOCK_SIZE];BATCH_SIZE] = create_array(&y).unwrap();
    let tensor_y:Tensor = Tensor::new(&array_y,  DEVICE).unwrap();
    (tensor_x, tensor_y)
}





// AUXILIAR FUNCTIONS
// Review generated code:
fn create_array(x: &Vec<Vec<u32>>) -> Option<[[u32; BLOCK_SIZE]; BATCH_SIZE]> {
    // Assuming BLOCK_SIZE and BATCH_SIZE are known constants 

    if x.len() == BATCH_SIZE && x.iter().all(|row| row.len() == BLOCK_SIZE) {
        let mut array_x = [[0; BLOCK_SIZE]; BATCH_SIZE];

        for (i, row) in x.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                array_x[i][j] = *value;
            }
        }

        Some(array_x)
    } else {
        None // Inner vectors have wrong lengths or mismatch with BATCH_SIZE
    }
}

/*
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
 */


