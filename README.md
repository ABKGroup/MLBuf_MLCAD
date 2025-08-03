# MLBuf: Recursive Learning-Based Virtual Buffering for Analytical Global Placement

> An open-source learning-driven virtual 
buffering-aware analytical global placement framework, 
built on top of the OpenROAD infrastructure.


![MLBuf Model Structure](images/model_structure_details.png)

## Code Structure
```bash
MLBuf/   
├── data/                     # Data loading and preprocessing scripts
│   ├── buf_data.csv          # Buffer information
│   ├── data_loader.py        # Load dataset
│   └── training_data         # Training dataset
├── models/                   # Model architecture 
│   ├── model.py              # MLBuf model
│   ├── inference.py          # MLBuf model inference
│   ├── losses.py             # Loss functions
│   └── layers.py             # Custom layers or modules
├── utils/                    # Utility functions and helper scripts
│   ├── adhoc_baseline.py     # Ad-Hoc baseline for comparison
│   ├── util.py               # Utility functions such as feature update
│   └── plot_utils.py         # Visualize training curves
├── scripts/                  # Scripts for global placement and evaluation
│   ├── invs_scripts          # Commercial tool scripts for generating post-route results 
│   └── OR_scripts            # OpenROAD scripts for global placement
├── train.py                  # Entry point for training the model
├── OR_branch_integration/    # MLBuf-RePlAce
├── LICENSE                   # License
└── README.md                 # Project documentation