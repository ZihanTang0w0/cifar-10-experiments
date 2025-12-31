I am trying to train a diffusion model and a flow model over CIFAR-10 with CFG to test if naive conditional generation fails.

See [visualization](visualization.ipynb) for detailed model architecture, training, sampling and evaluation details as well as how the graphs are plotted.

## Sample Generations

### Flow Model (CFG 1.0 vs 3.2)

<table>
<tr>
<th></th>
<th>airplane</th>
<th>automobile</th>
<th>bird</th>
<th>cat</th>
<th>deer</th>
</tr>
<tr>
<td><b>Naïve conditional samples</b></td>
<td><img src="final_grids/flow/class_0_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/flow/class_1_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/flow/class_2_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/flow/class_3_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/flow/class_4_cfg_1.0.png" width="150"/></td>
</tr>
<tr>
<td><b>CFG Samples</b></td>
<td><img src="final_grids/flow/class_0_cfg_3.2.png" width="150"/></td>
<td><img src="final_grids/flow/class_1_cfg_3.2.png" width="150"/></td>
<td><img src="final_grids/flow/class_2_cfg_3.2.png" width="150"/></td>
<td><img src="final_grids/flow/class_3_cfg_3.2.png" width="150"/></td>
<td><img src="final_grids/flow/class_4_cfg_3.2.png" width="150"/></td>
</tr>
</table>

<table>
<tr>
<th></th>
<th>dog</th>
<th>frog</th>
<th>horse</th>
<th>ship</th>
<th>truck</th>
</tr>
<tr>
<td><b>Naïve conditional samples</b></td>
<td><img src="final_grids/flow/class_5_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/flow/class_6_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/flow/class_7_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/flow/class_8_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/flow/class_9_cfg_1.0.png" width="150"/></td>
</tr>
<tr>
<td><b>CFG Samples</b></td>
<td><img src="final_grids/flow/class_5_cfg_3.2.png" width="150"/></td>
<td><img src="final_grids/flow/class_6_cfg_3.2.png" width="150"/></td>
<td><img src="final_grids/flow/class_7_cfg_3.2.png" width="150"/></td>
<td><img src="final_grids/flow/class_8_cfg_3.2.png" width="150"/></td>
<td><img src="final_grids/flow/class_9_cfg_3.2.png" width="150"/></td>
</tr>
</table>

### Diffusion Model (CFG 1.0 vs 4.0)

<table>
<tr>
<th></th>
<th>airplane</th>
<th>automobile</th>
<th>bird</th>
<th>cat</th>
<th>deer</th>
</tr>
<tr>
<td><b>Naïve conditional samples</b></td>
<td><img src="final_grids/diffusion/class_0_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_1_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_2_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_3_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_4_cfg_1.0.png" width="150"/></td>
</tr>
<tr>
<td><b>CFG Samples</b></td>
<td><img src="final_grids/diffusion/class_0_cfg_4.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_1_cfg_4.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_2_cfg_4.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_3_cfg_4.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_4_cfg_4.0.png" width="150"/></td>
</tr>
</table>

<table>
<tr>
<th></th>
<th>dog</th>
<th>frog</th>
<th>horse</th>
<th>ship</th>
<th>truck</th>
</tr>
<tr>
<td><b>Naïve conditional samples</b></td>
<td><img src="final_grids/diffusion/class_5_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_6_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_7_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_8_cfg_1.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_9_cfg_1.0.png" width="150"/></td>
</tr>
<tr>
<td><b>CFG Samples</b></td>
<td><img src="final_grids/diffusion/class_5_cfg_4.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_6_cfg_4.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_7_cfg_4.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_8_cfg_4.0.png" width="150"/></td>
<td><img src="final_grids/diffusion/class_9_cfg_4.0.png" width="150"/></td>
</tr>
</table>

## Results: Diffusion vs Flow Comparison

### Global FID vs CFG Scale
| Diffusion | Flow |
|-----------|------|
| ![Diffusion FID vs CFG](graphs/eval_batch_diffusion_1000/graph1a_fid_vs_cfg.png) | ![Flow FID vs CFG](graphs/eval_batch_flow_1000/graph1a_fid_vs_cfg.png) |

### Global FID vs Sampling Steps
| Diffusion | Flow |
|-----------|------|
| ![Diffusion FID vs Steps](graphs/eval_batch_diffusion_1000/graph1b_fid_vs_steps.png) | ![Flow FID vs Steps](graphs/eval_batch_flow_1000/graph1b_fid_vs_steps.png) |

### Class-wise Quality-Diversity Tradeoff
| Diffusion | Flow |
|-----------|------|
| ![Diffusion Quality-Diversity](graphs/eval_batch_diffusion_1000/graph2_quality_diversity.png) | ![Flow Quality-Diversity](graphs/eval_batch_flow_1000/graph2_quality_diversity.png) |

### Class Separability
| Diffusion | Flow |
|-----------|------|
| ![Diffusion Separability](graphs/eval_batch_diffusion_1000/graph3_separability.png) | ![Flow Separability](graphs/eval_batch_flow_1000/graph3_separability.png) |
