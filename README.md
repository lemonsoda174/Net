**Implementation of ST-related models**

Models implemented so far: HisToGene, ST-Net

**How to run**

Download data: Run "git clone https://github.com/almaan/her2st.git" in the "data" folder to clone the HER2ST dataset locally
To unzip files in ST-cnts, run "gunzip *.gz" with the current path in terminal as the ST-cnts folder

Train: Run ST_train.py. 
Test: Run ST_predict.py. Results printed are shown in metrics (printed in cmd line) and figures (imgs saved in the "figures" folder)
