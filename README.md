## 3rd year undergraduate lab focusing on MicroBooNE experiment
### Developed by O.G. Finnerud, A. Kedziora, & J. Waiton

This folder contains almost everything that is needed to run the lab, including:

#### /data/
Data no longer included here, available at the link below.
6 files included:
- DataSet_LSND.csv: Dataset for LSND used as comparison (42.9KB).
- DataSet_MiniBooNE.csv: Dataset for MiniBoone used as comparison (17.0KB).
- data_flattened.pkl: True data from MicroBooNE flattened (24.2MB). (A)
- MC_EXT_flattened.pkl: MC and EXT data from simulations & MicroBooNE (111.4MB). (A)
- bnb_run3_mc_larcv_slimmed.h5: Event display file (77.3MB).
- oscillated_data.pkl: Data file for usage in the closure test. (140MB)

#### NOTE! All files now accessible via this link for University of Manchester students.
https://livemanchesterac-my.sharepoint.com/:u:/g/personal/john_waiton_postgrad_manchester_ac_uk/ERr7l9mqqGZIveVcVG3UD2wBrCHCYTcJPT7jIpyBOM4zZg?e=Bh2uvB



#### MainAnalysis_template.ipynb
Template for student's usage throughout the lab.

#### MicroBooNE_Y3_Labbook.pdf
PDF including all information needed to complete this lab.


#### Neutrino_functions.py
Useful functions that are used throughout the lab.


#### event_display.ipynb & EventDisplay.py
Allows for visualisation of events.

#### keras_example.ipynb
An example of a simple NN in action for interest of the reader.

### INSTALLATION AND USAGE ON YEAR 3 LAB MACHINES

##### Ensure you have approximately 10GB of space on your OneDrive for this method

- Download this repository either via git or by pressing the green "Code" button and then "Download ZIP". 
- Extract the ZIP file into your local OneDrive folder.
- Download the bnb_run3_mc_larcv.h5 file from either USB or onedrive link (provided by TAs), and place within the /data/ folder
- Begin reading through the rest of the lab script, good luck!
