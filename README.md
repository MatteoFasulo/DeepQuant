# DeepQuant

A library for true-quantization and optimization of neural networks.

DeepQuant is developed as part of the PULP project, a joint effort between ETH Zurich and the University of Bologna.

## License

Unless specified otherwise in the respective file headers, all code checked into this repository is made available under a permissive license. All software sources and tool scripts are licensed under Apache 2.0, except for files contained in the `scripts` directory, which are licensed under the MIT license, and files contained in the `DeeployTest/Tests`directory, which are licensed under the [Creative Commons Attribution-NoDerivates 4.0 International](https://creativecommons.org/licenses/by-nd/4.0) license (CC BY-ND 4.0).

## Installation

Start by creating a new env with `Python 3.11` or higher. Then clone the repo and install the library as an editable package with:
```
pip install -e .
```

### EMG Transformer

To use the EMG Transformer model, the main script is `my_model.py`. You can run it with:

```bash
python my_model.py
```

and it will apply pre-quantization, quantization, and post-quantization optimizations to the EMG Transformer model. You can have a look at the script, it incorporate additional procedures to load pre-trained weights, perform static quantization with calibration data and verify the performances with test data under both FP32 and INT8 precision using ONNX Runtime.

Pre-quantization optimizations are largely inspired by the one of CCT applying Transpose-Add fusions, QKV fixes and adding dequant nodes before matmuls. Not all of them will be applied for EMG Transformer due to differences in architecture with respect to CCT.

Post-quantization optimizations include checking for explicit matmuls in the form of @ operator fusion and other graph simplifications.

After producing the output ONNX model, since the ONNX opset used is 17, GELU Op. is not represented with its dedicated op but with a subgraph. To have a more efficient implementation that is directly mapped to GELU, we can apply the following transformation:

```bash
python -m onnxruntime.transformers.optimizer --input Tests/ONNX/network.onnx --output Tests/ONNX/network.onnx --model_type vit --num_heads 3 --hidden_size 192 --use_multi_head_attention --disable_bias_gelu --disable_bias_skip_layer_norm --disable_skip_layer_norm --use_multi_head_attention --opt_level 0
```

this will apply ONNX Runtime's built-in transformer optimizations to the model. Note that optimization level 0 does not apply any further optimization unless explicitly specified, so we can safely use it to just convert the GELU subgraph to the dedicated op by specifying the model type, number of heads and hidden size together with other flags for LayerNorm and MultiHeadAttention fusion.

Now, since applying these optimizations might change the graph structure, it is recommended to re-apply symbolic shape inference to have correct shapes for all nodes:

```bash
python -m onnxruntime.tools.symbolic_shape_infer --input Tests/ONNX/network.onnx --output Tests/ONNX/network.onnx
```

which will be applied as in the previous case **in-place** overwriting the input model producing an updated model with correct shapes.

The final ONNX model together with input and output test data can be found in `Tests/ONNX/` folder.

## Running Tests

We provide comprehensive tests with pytest, to execute all tests, simply run `pytest`. We mark our tests in two categories, `SingleLayerTests` and `ModelTests`, to execute the tests of the specific category, you can run `pytest -m <category>`. For instance, to execute only the single layer tests, you can run `pytest -m SingleLayerTests`.

## ⚠️ Disclaimer ⚠️
This library is currently in **beta stage** and under active development. Interfaces and features are subject to change, and stability is not yet guaranteed. Use at your own risk, and feel free to report any issues or contribute to its improvement.
