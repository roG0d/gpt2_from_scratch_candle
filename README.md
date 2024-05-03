Install cuda on Manjaro:
`sudo pacman -S cuda cudnn`
(Reboot so binaries can be sourced to bash)

Add dependency candle-core to rust project:
`cargo add --features candle-core/cuda --git https://github.com/huggingface/candle.git candle-core`



### References
- [Rust random number gen](https://rust-random.github.io/book/guide-seeding.html)
- [Candle docs](https://huggingface.github.io/candle/guide/hello_world.html)
- [Candle tutorial](https://github.com/ToluClassics/candle-tutorial?tab=readme-ov-file#tensors)
- [Another gpt-2 Andrej Karpathy implementation](https://github.com/jeroenvlek/gpt-from-scratch-rs)
