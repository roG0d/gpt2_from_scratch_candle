use std::f32::consts::E;

use candle_core::{shape, DType, Device, IndexOp, NdArray, Tensor};
use candle_nn::{embedding, loss::cross_entropy, Embedding, Module, VarBuilder, VarMap, ops::softmax};
use rand::{thread_rng, Error};
use ndarray::{s, Array};


const DEBUG:bool = true;
/* Credits to: https://github.com/huggingface/candle/issues/406 */

#[derive(Debug)]
pub struct Bigram {
    token_emmbeding_table: Embedding,
}

impl Bigram {
    // Equivalent to init in python torch:
    pub fn new(vocab_size: usize) -> Result<Self, Error> {

        // VarBuilder initializes weights for a model:
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::new_cuda(0).unwrap());
        let token_embedding_table = embedding(vocab_size, vocab_size, vb).unwrap();
        
        Ok(Bigram {
            token_emmbeding_table: token_embedding_table,
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

            if DEBUG {println!("logits shape: {:?}", logits.shape());}
            if DEBUG {println!("targets shape: {:?}", targets.shape());}
            
            loss = cross_entropy(&logits, &targets).unwrap();
            
        }else{
            loss = Tensor::zeros((1,1), DType::F32, &Device::new_cuda(0).unwrap()).unwrap();
        }
        (logits, loss)
    }

    pub fn generate(&self, idx: &Tensor, max_new_tokens: usize){

        for _ in 0.. max_new_tokens{
            let (logits, _) = self.forward(idx, None);
            println!("{:?}", logits.shape());

            // Logits is already (B, C) wtf karthapy https://youtu.be/kCc8FmEb1nY?t=1826 CAREFUL
            //let slice = 
            //let array_logits = slice.slice(s![.., -1, ..]).to_owned();
            //let array_logits = logits.i(index)
            let probs = softmax(&logits, 1).unwrap();

            // sample_multinomial not implemented in candle -> https://github.com/jeroenvlek/gpt-from-scratch-rs/blob/main/src/sampling.rs
            let idx_next = sample_multinomial(&idx.to_vec1().unwrap());

                                                                // to_vec2 cause 2 dimensions
        }
    }
}

use rand::distributions::Distribution;
pub fn sample_multinomial(logits: &Vec<f32>) -> candle_core::Result<u32> {
    let mut rng = thread_rng();
    let distribution = rand::distributions::WeightedIndex::new(logits).unwrap();
    let next_token = distribution.sample(&mut rng) as u32;

    Ok(next_token)
}