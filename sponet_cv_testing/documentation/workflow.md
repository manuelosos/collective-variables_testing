# Basic Usage

The testpipeline can be called from the command line by the main.py method.
The arguments for the runs can either be provided via the cli 
or they can be specified in an additional `CONFIG.json` file.
All the cli arguments are optional keyword parameters.
Every argument of the program can be specified in the config. 
If an argument is specified in the config and is parsed as a cli argument,
the cli argument will overwrite the config specification. 
This allows to configure some standard behaviour
while still being able to control exact behaviour for every individual run if the need arises.

The arguments `runfile_path`, `result_path` and `network_path` must be specified 
as a command line argument or in the config.
If not done so, the program will raise an exception.
Otherwise, all other parameters are completely optional.
For more specifics on the parameters, consider the documentation in `main.py`.  
The `CONFIG.json` has to be located in the same folder as the `main.py`. 
As the file ending suggests the file has to be a json file.

Here is one example how the config file can look like.
```json 
{
  "runfile_path": "runfiles/",
  "result_path": "results/",
  "network_path": "networks/",
  "num_threads": 20,
  "delete_runfile": false,
  "error_exit": true,
  "device": "X1"
}
```



Here is one example how a call to `main.py` could look like.
```
   python3 main.py --runfile_path runfiles/test1.json --result_path results/ --network_path networks/
```
The run specified in `test1.json` located in the `runfiles/` folder will be computed.  
If test1 requires an already created network this can be found in the folder `networks/`.  
The results of this test will be saved in the folder `results/`.   






