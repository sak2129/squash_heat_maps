# Squash_heat_maps
This project processes squash videos and creates heatmaps that players/coaches can use to improve their game. 

Colab description: https://colab.research.google.com/drive/1XQhroKiA16F3tn2QSVCW7ZbdzstIhkrc?usp=sharing

## Usage
### Step 1
Utilize exportframe.py to obtain court dimensions
./resources/video01.mp4

### Step 2
python3 squash_heat_map.py --video_path='./resources/video02.mp4' --coordinates=[[434,285],[830,285],[927,458],[340,457]] --verbose=TRUE

