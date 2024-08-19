
## Runfile Documentation
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

### First layer
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
  - States the model of the network see network models for a list of the allowed networks
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
      If not specified only the result at t=lag_time will be returned.
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

##### Network models

Allowed network models:

* Albert-Barabasi
  * Key: albert-barabasi
  * In the literature mostly called "Barabási-Albert-Model"


##### Results

The results of each run will be saved in an individual folder. 
The folder will be named with the run_id and is located in the path specified by command line argument. 
The following quantities are saved.

- **network_anchor_points** 
  - Initial anchor points used to sample the dynamics.
  - Array of shape=(`num_anchor_points`, `num_nodes`)

- **network_dynamics_sample** 
  - Samples of the dynamics with initial values chosen as `network_anchor_points`
  `num_timesteps` determines the number of intermediate samples. Samples will be generated according to 
  `np.linspace(0, lag_time, num_timesteps)`.
  - Array of shape `(num_anchor_points, num_samples_per_anchor, num_timesteps, num_nodes)`