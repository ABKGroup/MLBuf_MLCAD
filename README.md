# MLBuf: Recursive Learning-Based Virtual Buffering for Analytical Global Placement 

An open-source learning-driven virtual 
buffering-aware analytical global placement framework, 
built on top of the OpenROAD infrastructure.

<!-- [Paper link](https://vlsicad.ucsd.edu/Publications/Conferences/416/c416.pdf) (Best Artifact Award, MLCAD25) -->
<!-- 
![MLBuf-RePlAce Workflow](images/workflow_mlbuf_replace.png)
 -->

## Important Notice: MLBuf v2 (November 2026)

This version replaces the MLCAD25 release. It fixes the unit conversion issue reported in [Issue #6](https://github.com/ABKGroup/MLBuf_MLCAD/issues/6) and provides a retrained model, **MLBuf v2**.

What is fixed:
- **Unit conversion.** Predicted buffer boxes (microns) are converted to DBU before their area is reserved in the placer. 
- **Model.** MLBuf v2 is trained with unbuffered nets as negative samples and can predict "no buffer" for a net. 
- **Reproducibility.** Inference is deterministic. The Ad-Hoc baseline is seeded.
<!-- - **Ad-Hoc baseline.** The wire capacitance unit is corrected. -->
<!-- - **Robustness.** Environment variables are read safely by the placer. -->
<!-- - **Net weights.** MLBuf-RePlAce keeps the slack-based net weights of TD-RePlAce.  -->

The MLBuf-RePlAce and Ad-Hoc results in Tables IV and V of the MLCAD25 paper were produced with the unit conversion issue and are superseded. Updated results will be reported in the revised arXiv paper. The MLCAD25 release remains available at commit `a10d262`.

We sincerely thank the researchers who identified and reported this issue for their careful investigation and constructive feedback.



## Code Structure
```bash
MLBuf/   
├── data/                     # Data
│   ├── buf_data.csv          # Buffer information
│   ├── feedback_estimator.json # Feature update coefficients
│   └── training_data         # Training dataset (ibex, jpeg)
├── mlbuf2/                   # MLBuf v2 
│   ├── schema.py             # Parse the training dataset
│   ├── sources.py            # Paths and training data sources
│   ├── build_dataset.py      # Build net records (buffered and unbuffered nets)
│   ├── calibrate_feedback.py # Refit the feature update of upper recursion levels (optional)
│   ├── levels.py             # Features and labels of each recursion level
│   ├── physics.py            # Buffer library and feature update
│   ├── dataset.py            # Load dataset
│   ├── model.py              # MLBuf v2 model
│   ├── losses.py             # Loss functions
│   ├── train.py              # Model training
│   └── infer.py              # Model inference
├── configs/                  # Training configuration
├── results/                  # Model results 
├── scripts/                  # Scripts for global placement and evaluation
│   ├── invs_scripts          # Commercial tool scripts for generating post-route results 
│   └── OR_scripts            # OpenROAD scripts for global placement and ORFS evaluation
├── train.py                  # Entry point for training the model
├── requirements.txt          # Dependencies for model training and inference
├── env_or.sh                 # Environments for OpenROAD and ORFS 
├── OR_branch_integration/    # MLBuf-RePlAce
│   ├── mlbuf_infer.py        # MLBuf v2 inference called by the placer
│   └── adhoc_baseline.py     # Ad-Hoc baseline for comparison
├── OR_inputs/                # Designs & platform
├── LICENSE                   # License
└── README.md                 # Project documentation
```

## Training MLBuf
To train MLBuf v2, run the following command (Python >= 3.9):
```bash
pip install -r requirements.txt
python train.py
```

Results are written to results/:
```bash
results/
├── model_dict/     # pre-trained model (.pt)
└── mlbuf_v2/       # trained model (model_best.pt), training log (log.csv)
```
You can directly use our pre-trained model:
```bash
results/model_dict/mlbuf-v2.pt
```
***Note: The pre-trained model is trained on ibex, jpeg, ariane133, ariane136, black_parrot and bsg_chip. This repository provides the training dataset of ibex and jpeg. The complete training dataset is available here.***

## Integrating MLBuf into OpenROAD for MLBuf-RePlAce
1. **Build OpenROAD** in the OR_branch_integration directory using the version [OpenROAD (mlbuf-v2 branch)](https://github.com/ytliu8464/OpenROAD_MLBuf/tree/mlbuf-v2) 
```bash
cd OR_branch_integration
git clone --recursive --branch mlbuf-v2 https://github.com/ytliu8464/OpenROAD_MLBuf.git OpenROAD
cd OpenROAD
mkdir build && cd build
cmake ..
make -j "$(nproc)"
```

***Note: To fully reproduce results, ensure the execution environment for OpenROAD and ORFS matches ours. The environment setup is provided in env_or.sh.***

2. Run **MLBuf-RePlAce**
```bash
cd scripts/OR_scripts/${DESIGN_NAME}
./run_mlbuf.sh
```
Set `MLBUF_PYTHON` to the Python interpreter that has the packages in requirements.txt (default: `python`).

Check that buffer area is reserved at every timing-driven iteration:
```bash
python ../check_reservation.py run_replace_mlbuf_mlbuf-v2.log
```

3. (Optional) Run baseline comparisons:

- **Ad-Hoc Baseline**
```bash
./run_baseline_adhoc.sh
```
- **RePlAce** (no timing-driven mode)
```bash
./run_baseline_no_timing.sh
```
- **TD-RePlAce** (default virtual buffering-based timing driven global placement in OpenROAD)
```bash
./run_baseline_rsz_virtual.sh
```
4. **Global placement results**

The global placement results (both .def and .odb) are generated in:
```bash
scripts/OR_scripts/${DESIGN_NAME}/mlbuf-v2/${DESIGN_NAME}_mlbuf.odb   # MLBuf-RePlAce
scripts/OR_scripts/${DESIGN_NAME}/${DESIGN_NAME}_rsz_adhoc.odb        # Ad-Hoc baseline
scripts/OR_scripts/${DESIGN_NAME}/${DESIGN_NAME}_no_timing.odb        # RePlAce
scripts/OR_scripts/${DESIGN_NAME}/${DESIGN_NAME}_rsz_virtual.odb      # TD-RePlAce
```
We also provide .odb global placement results generated by MLBuf-RePlAce, Ad-Hoc baseline, RePlAce and TD-RePlAce in the same directory.

## Evaluation

To generate post-route results, use the global placement results as the starting point for downstream flows in **OpenROAD-flow-scripts (ORFS)**.

**Steps:**

1. Set up **OpenROAD-flow-scripts (ORFS)** using the version [ORFS (mlbuf-orfs-v2 branch)](https://github.com/ytliu8464/OpenROAD-flow-scripts/tree/mlbuf-orfs-v2) 
```bash
git clone --branch mlbuf-orfs-v2 https://github.com/ytliu8464/OpenROAD-flow-scripts.git
```

2. Run the automation script to generate post-route results:
```bash
export FLOW_HOME=/path/to/OpenROAD-flow-scripts/flow
scripts/OR_scripts/run_orfs.sh ${DESIGN_NAME} ${METHOD}
```
- **DESIGN_NAME**: ibex, jpeg_encoder
- **METHOD**: mlbuf, adhoc, no_timing, rsz_virtual

3. After the run completes, the script prints WNS, TNS, power and routed wirelength. All metrics are available in:
```bash
scripts/OR_scripts/${DESIGN_NAME}/orfs_${METHOD}/logs/nangate45/*/run/*.log
```

***Note: To fully reproduce results, ensure the execution environment for OpenROAD and ORFS matches ours. The environment setup is provided in env_or.sh.***


## Reference
Please cite our work if you find our code useful.
```bash
A. B. Kahng, Y. Liu and Z. Wang, 
"Recursive Learning-Based Virtual Buffering for Analytical Global Placement", 
Proc. ACM/IEEE Intl. Symp. on Machine Learning for CAD (MLCAD), 2025. 
```
