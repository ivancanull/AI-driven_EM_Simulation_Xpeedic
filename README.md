# AI-driven_EM_Simulation

## Inference

1. Create a folder named `Data/` in the root directory.
2. Create a folder named `Models/` in the root directory.
3. Copy the trained model to `Models/` folder.
4. Create a folder named `Results/` in the root directory.
4. Modify the configuartion file in `Configs/Inference` folder. For example, `Configs/Inference/Final_Cases_7_1220_1.json`. Define the parameter ranges and steps in the json file. 
5. Run `python inference.py -s CONFIG_FILE` to inference the model.
