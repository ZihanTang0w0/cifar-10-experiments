I am trying to train a diffusion model and a flow model over CIFAR-10 with CFG to test if naive conditional generation fails.

See [visualization](visualization.ipynb) for detailed model architecture, training, sampling and evaluation details as well as how the graphs are plotted. 

## Results: Diffusion vs Flow Comparison

### FID vs CFG Scale
| Diffusion | Flow |
|-----------|------|
| ![Diffusion FID vs CFG](graphs/eval_batch_diffusion_1000/graph1a_fid_vs_cfg.png) | ![Flow FID vs CFG](graphs/eval_batch_flow_1000/graph1a_fid_vs_cfg.png) |

### FID vs Sampling Steps
| Diffusion | Flow |
|-----------|------|
| ![Diffusion FID vs Steps](graphs/eval_batch_diffusion_1000/graph1b_fid_vs_steps.png) | ![Flow FID vs Steps](graphs/eval_batch_flow_1000/graph1b_fid_vs_steps.png) |

### Quality-Diversity Tradeoff
| Diffusion | Flow |
|-----------|------|
| ![Diffusion Quality-Diversity](graphs/eval_batch_diffusion_1000/graph2_quality_diversity.png) | ![Flow Quality-Diversity](graphs/eval_batch_flow_1000/graph2_quality_diversity.png) |

### Class Separability
| Diffusion | Flow |
|-----------|------|
| ![Diffusion Separability](graphs/eval_batch_diffusion_1000/graph3_separability.png) | ![Flow Separability](graphs/eval_batch_flow_1000/graph3_separability.png) |
