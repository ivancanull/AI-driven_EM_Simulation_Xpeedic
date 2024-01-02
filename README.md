# AI-driven_EM_Simulation

## Flow

1. Define the parameters variable with json file in `Configs/Parameters`, define the dataset generation configurations in `Configs/Generations`.
2. Set the working directory to `Codes/`.
3. Run `python dataset_generation.py` to generate initial dataset scripts. The samples are output to `Data/Samples`. 
4. Use TmlExpert/Metis to generate the corresponding dataset.
5. Run `python data_parser.py` to parse the dataset into the format `$DATASET_NAME_$PORT_concat.zip`.
6. Copy the dataset and `.pkl` files into the `Data/Dataset/$DATASET_NAME/` directory.
6. Run `python train.py` or `python tune.py` to train the model.
7. Run `python test.py` to test the model.

## Config Files 
1. `Configs/xxx.yml` defines the parameters for the model in training.
    See `Configs/Example.yml` for details.
2. `Configs/Generations/xxx.yml` defines the parameters for the dataset generation.
    See `Configs/Generations/Example.yml` for details.
3. `Configs/Parameters/xxx.json` defines the range and steps of the generated dataset, and also the variables.
    See `Configs/Parameters/Example.json` for details. It the following types of combination method:
    * `Dot`: It samples only one independent range of values.
    * `Pair`: It samples combinations of several different ranges, and the settings of each parameter are represented as list in the json file.
    * `BBDesign`: It uses the Box-Behnken design to sample the combinations of several different ranges.

## Dataset Generation
See `python dataset_generation.py --help`. Run

    python dataset_generation.py -s GENERATION_CONFIG_FILE
to generate the dataset. The data directory is kept in `Data/Samples/`.

## Data Parser
See `python data_parser.py --help`. Run

    python data_parser.py -d DATASET_NAME -r DATASET_PARENT_DIR -m MultiLayer -p PORT_NUM -n FREQ_NUM -b DATASET_BATCH_NUM

Modify the port need to be parsed in `generate_port_list_multilayer(port)` function in data_parser.py
## Tune
See `python tune.py --help`. Run

    python tune.py -s CONFIG_FILE

## Train
See `python train.py --help`. Run

    python train.py -s CONFIG_FILE

## Test
See `python test.py --help`. Run

    python test.py -s CONFIG_FILE