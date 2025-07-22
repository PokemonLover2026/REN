# Code

## Installation

Create a conda environment and install the dependencies except those for verification:
```bash
conda create --name lnc python=3.11
conda activate lnc
pip install -r requirements.txt
```

We use [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA.git) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git) for verification. To install both of them, run:
```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
(cd alpha-beta-CROWN/auto_LiRPA && pip install -e .)
(cd alpha-beta-CROWN/complete_verifier && pip install -r requirements.txt)
```

To set up the path:
```
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
```

## Verification
We provide `arguments.py` to set the configurations and `RVN_start.py` to start our tool REN. 

```bash
cd verification
export CONFIG_PATH=$(pwd)

```

