
### Runfile Documentation
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
Optional values will be denoted with an "optional" in round brackets.
If optional values are not specified in the runfile, default values which are described here will be used.

All parameters are independent of the individual machine that is used for computation.

For example:
`value: datatype (other_value=state)`, 
here value must be specified in the runfile if other_value is set to state.
Otherwise, it can be omitted.

If "other value" is also conditional, the corresponding condition will not be specified again in "value".

The first layer of the file consists of
* **run_id**: string 
  * should be unique in the data-csv
* **remark**: string (optional) 
  * Place for comments and remarks on the run. Will be saved in misc data.
* **dynamic**: dict 
* **network**: dict
* **simulation**: dict

The dynamic dict has the following attributes
* **model**: string 
  * Determines the dynamic model. Currently only "CNVM" is supported "CNTM" follows later.
* **num_state**s: integer
  * Number of possible states that nodes can be in. 
  * Computations can be done with more states. The datamanagement currently only supports num_states=2
* **rates**: dictionary 
  * currently specified for "CNVM" only
  * **r**: "array" of dimension: num_states x num_states
    * Describes the rates of state transmission through neighbors
  * **r_tilde**: "array" of dimension: num_states x num_states
    * Describes the rates of state transmission through "noise"

The network dict has the following attributes
* **generate_new**: boolean
  * States if a network should be generated new of if an already created network should be utilized
* **network_id**: string (generate_new=false)
  * specifies the id of the network that should be loaded. 
    If generate_new=true and archive=true, network_id can be set to archive the network under the given id.
    If in this case the passed network id is not unique, a warning will be logged and a new id will be assigned.
* **model**: string (generate_new=true)
  * States the model of the network see network models for a list of the allowed networks
* **num_nodes**: integer (generate_new=true)
  * Number of nodes that the network should have
* **num_attachments**: integer (model="albert-barabasi", "holme-kim")
  * Parameter for models with preferential attachment. Determines to how many nodes a new node attaches.
* **triad_probability**: float (model="holme-kim")
  * Parameter for the holme-kim model. 
  Determines the probability that a newly attached node forms a triad with the neighbors of the already existing node. 



The simulation dict has the following attributes
* **sampling**: dictionary
  * **method**: string
    * Determines the method that is utilized for sampling.
    * Currently only "local_cluster" is supported
  * **lag_time**: float 
    * Specifies the duration of the simulation.
  * **short_integration_time**: float (optional)
    * If no short integration time is specified
    the short integration time will be chosen with respect to the maximum rates and the overall lag time.
    This is equivalent to setting the short integration time to a negative value.
  * **num_timesteps**: int (optional)
    * Number of with respect to time equidistant intermediate results.
      For every intermediate result the transition matrix and diffusion maps will be computed
      If not specified only the result at t=lag_time will be returned.
  * **num_anchor_points**: integer
    * Number of different anchorpoints that are sampled
  * **num_samples_per_anchor**: int
    * Number of Monte-Carlo samples per anchor
* **num_coordinates**: integer
  * specifies the number of coordinates of the collective variable that will be saved.
* **triangle speedup**: bool
  * If set to true the distance matrix will be computed lazily by exploiting the triangle inequality.
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
        "archive": false,
        "model": "albert-barabasi",
        "num_nodes": 200,
        "num_attachments": 2
    },
    "simulation": {
        "sampling": {
            "method": "local_cluster",
            "lag_time": 4,
            "num_anchor_points": 50,
            "num_samples_per_anchor": 10
        },
        "num_coordinates": 1
    }
}
```

##### Network models

Allowed network models:
* Albert-Barabasi
  * Key: albert-barabasi
  * In the literature mostly called "Barabási-Albert-Model"


##### Results

The results of each run will be saved in an individual folder. The folder is located in the path specified in the 
command line argument. The folder is named after the run_id. 
The following quantities are saved.

* **network_anchor_points** 
  * Initial anchor points used to sample the dynamics.
  * Array of shape=(`num_anchor_points`, `num_nodes`)

* **network_dynamics_sample** 
  * Samples of the dynamics with initial values chosen as `network_anchor_points`
  `num_timesteps` determines the number of intermediate samples. Samples will be generated according to 
  `np.linspace(0, lag_time, num_timesteps)`.
  * Array of shape `(num_anchor_points, num_samples_per_anchor, num_timesteps, num_nodes)`