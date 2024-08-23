# Basic Usage

The testpipeline can be called from the command line by the `start_testpipeline.py` script.
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
For more specifics on the parameters, consider the documentation in `testpipeline.py`.  
The `CONFIG.json` has to be located in the same folder as the `start_testpipeline.py`. 
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


Here is one example how a call to `start_testpipeline.py` could look like.
```
   python3 start_testpipeline.py --runfile_path runfiles/test1.json --result_path results/ --network_path networks/
```
The run specified in `test1.json` located in the `runfiles/` folder will be computed.  
If test1 requires an already created network this must be located in `networks/`.  
The results of this test will be saved in the folder `results/`.


# Runfile Documentation
The testpipeline uses json files as input. 
Each file specifies the parameters of one testrun.
This document describes the structure of the files.


Parameters all have unique names which will be referred to as "keys".
Parameters are bundled in nested dictionaries. 
All keys are strings.
The datatype of the parameter is specified after the key.
Key and datatype are separated by a `:`.
Some keys are conditional.
Conditional keys must be specified when certain conditions are met.
These conditions will be noted in round brackets after the key specification.
If no condition is specified the value must always be present.
Optional values will be denoted with an "optional" in round brackets after the datatype.
If optional values are not specified in the runfile, default values are used.
These default values are also specified here.

For example:
`value: datatype (other_value=state)`, 
here value must be specified in the runfile if other_value is set to state.
Otherwise, it can be omitted.
If "other value" is also conditional, the corresponding condition will not be specified again in "value".

`value: int (optional default: 1)`
In this example value is optional. Meaning it can always be omitted from the runfile. 
If it is not specified, the default value 1 will be used for the run.

All parameters are independent of the individual machine that is used for computation.

## First layer
The first layer i.e. the first dictionary of the file consists of the following attributes:

- **run_id**: string
  - should be unique in the data-csv.
- **remark**: string (optional default value `""`) 
  - Place for comments and remarks on the run. Will be saved in misc data.
- **dynamic**: dict 
- **network**: dict
- **simulation**: dict

### Dynamic
The dynamic dict has the following attributes

- **model**: string 
  - Determines the dynamic model. Currently only "CNVM" is supported.
- **num_state**s: integer
  - Number of possible states that nodes can be in. 
  - Computations can be done with more states. 
  The results should be valid but the datamanagement and dashboard cannot handle it
- **rates**: dictionary 
  - currently specified for "CNVM" only
  - **r**: array with `shape=(num_states, num_states)`
    - Describes the rates of state transmission through neighbor influence
  - **r_tilde**: array with `shape=(num_states, num_states)`
    - Describes the rates of state transmission through "noise"


### Network
The network dict has the following attributes

- **generate_new**: boolean
  - States if a network should be generated new of if an already created network should be utilized
- **network_id**: string (generate_new=false)
  - specifies the id of the network that should be loaded. 
  - The id is just the name of the file where the network is saved.
- **model**: string 
  - States the model of the network.
    Currently only `albert-barabasi` and `holme-kim` are supported.
- **num_nodes**: integer (generate_new=true)
  - Number of nodes that the network should have
- **num_attachments**: integer (model="albert-barabasi", "holme-kim")
  - Parameter for models with preferential attachment. Determines to how many nodes a new node attaches.
- **triad_probability**: float (model="holme-kim")
  - Parameter for the holme-kim model. 
  Determines the probability that a newly attached node forms a triad with the neighbors of the already existing node. 


### Simulation
The simulation dict has the following attributes

- **sampling**: dictionary
  - **method**: string
    - Determines the method that is utilized for sampling.
    - Currently only "local_cluster" is supported
  - **lag_time**: float 
    - Specifies the duration of the simulation.
  - **short_integration_time**: float (optional)
    - If no short integration time is specified
    the short integration time will be chosen with respect to the maximum rates and the overall lag time.
    This is equivalent to setting the short integration time to a negative value.
  - **num_timesteps**: int (optional)
    - Number of with respect to time equidistant intermediate results.
      For every intermediate result the transition matrix and diffusion maps will be computed
      If not specified only the result at `t=lag_time` will be returned.
  - **num_anchor_points**: integer
    - Number of different anchorpoints that are sampled
  - **num_samples_per_anchor**: int
    - Number of Monte-Carlo samples per anchor
- **num_coordinates**: integer
  - specifies the number of coordinates of the collective variable that will be saved.
- **triangle speedup**: bool
  - If set to true the distance matrix will be computed lazily by exploiting the triangle inequality.


Example:
```json
{
    "run_id": "1",
    "dynamic": {
        "model": "CNVM",
        "num_states": 2,
        "rates": {
            "r": [[0, 0.98],
                  [1,0]],
            "r_tilde": [[0, 0.05],
                        [0.02, 0]]
        }
    },
    "network": {
        "generate_new": true,
        "network_id": "test_network_1",
        "model": "albert-barabasi",
        "num_nodes": 200,
        "num_attachments": 2
    },
    "simulation": {
        "sampling": {
            "method": "local_cluster",
            "lag_time": 4,
            "num_anchor_points": 50,
            "num_samples_per_anchor": 10,
            "num_timesteps": 5
        },
        "num_coordinates": 4
    }
}
```


# Performance


The computational code is all written in Python.
The code is sped up and parallelized using Numba.

## NUMA

The code runs well on single cpu machines with no shared memory.
The code will run on multi-cpu machines with shared memory, but it can be very slow
since the code is not optimized for shared memory access.
To avoid this pitfall one should restrict the number of threads to the number of cores on one cpu.
To our knowledge this should automatically restrict the threads to one CPU.

## Hyperthreading

If the code is run on a machine where Hyperthreading is enabled,
Numba will choose the number of threads equal to the number of "virtual cores".
Meaning there are more threads than there are physical cores.
This would lead to a slower computation.
Thus, the number of threads has to be set manually
with the `num_threads` option of `start_testpipeline.py` or in the config to the number of physical cores 
if the Hyperthreading cannot be disabled.

## Memory Usage

Currently, the memory usage is not optimized.
The majority of space is taken by the Monte-Carlo samples.
These stay in memory until all distance matrices are computed.
The size of the samples in the memory is directly dependent on 
The samples required for one distance matrix is saved in memory as an array of
`num_anchor_points * num_samples_per_anchor * num_nodes` float64.
With larger networks more anchor_point and more samples per anchor are required.
For simulations with many time-steps this can lead to 
very large memory usage in the magnitude of 100+ gb.


## Disk Space

Since the memory usage can inflate pretty quickly the disk usage also has to be of concern.
The samples also take up most of the space.
The samples are saved in uint8 or uint16 depending on the number of states.
With the option `delete_samples` set to true the samples will be deleted 
after the distance_matrices are computed.












