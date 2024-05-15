use std::f32::consts::E;
use candle_core::{shape, DType, Device, IndexOp, NdArray, Tensor};
use candle_nn::{embedding, loss::cross_entropy, Embedding, Module, VarBuilder, VarMap, ops::softmax};
use rand::{thread_rng, Error};
use ndarray::{s, Array};

/*
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
 */


#[derive(Debug)]
pub struct Bigram {
    token_emmbeding_table: Embedding,
    env_runtime: Env,
}

/* Credits to: https://github.com/huggingface/candle/issues/406 */
impl Bigram {
    // Equivalent to init in python torch:
    pub fn new(vocab_size: usize) -> Result<Self, Error> {
        
        let env_runtime = Env::new();
        println!("{}", env_runtime);

        // VarBuilder initializes weights for a model:
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &env_runtime.device);
        let token_embedding_table = embedding(vocab_size, vocab_size, vb).unwrap();
        
        Ok(Bigram {
            token_emmbeding_table: token_embedding_table,
            env_runtime: env_runtime
        })
    }

    pub fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> (Tensor, Tensor){
        
        let logits = self.token_emmbeding_table.forward(idx).unwrap(); // Dimension = (B, T, C)

        // Logits and targets needs to be reshaped so it corresponds cross_entropy function https://github.com/huggingface/candle/blob/72110091790da645febcd03fbc044968310de73f/candle-nn/src/loss.rs#L30
        let shape = logits.shape().dims();
        let logits = logits.reshape(&[shape[0]*shape[1], shape[2]]).unwrap(); // Dimension = (B*T, C)
        let loss;

        // Optional targets inside Some()
        if let Some(targets) = targets{
            let targets = targets.reshape(&shape[0]*shape[1]).unwrap(); // Dimension = (B*T)
            
            loss = cross_entropy(&logits, &targets).unwrap();
            
        }else{
            loss = Tensor::zeros((1,1), DType::F32, &self.env_runtime.device).unwrap();
        }
        (logits, loss)
    }

    pub fn generate(&self, idx: &Tensor, max_new_tokens: usize){

        for i in 0.. max_new_tokens{
            let (logits, _) = self.forward(idx, None); // (B*T,C)
            if self.env_runtime.debug {println!("logits shape: {:?}",logits.shape());}

            // Logits is already (B, C) wtf karpathy https://youtu.be/kCc8FmEb1nY?t=1826 CAREFUL
            //let slice = 
            //let array_logits = slice.slice(s![.., -1, ..]).to_owned();
            //let array_logits = logits.i(index)

            let probs = softmax(&logits, 1).unwrap();
            let probs_dim2 = probs.to_vec2::<f32>().unwrap();
            //let idx_next = sample_multinomial(&probs_dim2);
        }
    }
}

// sample_multinomial not implemented in candle -> https://github.com/jeroenvlek/gpt-from-scratch-rs/blob/main/src/sampling.rs
use rand::distributions::Distribution;

use crate::env_runtime::Env;
pub fn sample_multinomial(logits: &Vec<f32>) -> candle_core::Result<u32> {
    let mut rng = thread_rng();
    let distribution = rand::distributions::WeightedIndex::new(logits).unwrap();
    let next_token = distribution.sample(&mut rng) as u32;

    Ok(next_token)
}