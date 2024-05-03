use candle_core::{Result, Tensor};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn forward(&self, x:&Tensor) -> Result<Tensor>{
        let x = x.matmul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}
pub struct Model {
    pub first: Linear,
    pub second: Linear,
}

impl Model {
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}