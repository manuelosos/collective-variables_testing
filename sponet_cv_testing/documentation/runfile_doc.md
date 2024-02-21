
### Runfile Documentation
The testpipeline uses json-files as input. 
Each file specifies the parameters of one testrun.
In this document, the specifications of these files is documented.
The json files have a Python dictionary like structure with nested dictionaries. 

All "keys" are strings. The corresponding value is specified after the key. 
If a value has to be present only when certain conditions are met, 
there will round brackets after the datatype in which the conditions are specified.

For example:
`value: datatype (other_value=state)`, 
here value must be specified in the runfile if other_value is set to state.
Otherwise, it can be omitted. 
If no condition is specified the value must always be present.
If "other value" is also conditional, the corresponding condition will not be specified again in "value".

The first layer of the file consists of
* **run_id**: string 
  * should be unique in the data-csv
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
* **archive**: boolean (generate_new=true)
  * Determines if the newly generated network should be saved in the central data structure.
    For details see "details on datamanagement" below.
* **model**: string (generate_new=true)
  * States the model of the network see network models for a list of the allowed networks
* **num_nodes**: integer (generate_new=true)
  * Number of nodes that the network should have
* **num_attachments**: integer (model="albert-barabasi)
  * Parameter for the albert-barabasi model. Determines to how many nodes a new node attaches.

The simulation dict has the following attributes
* **sampling**: dictionary
  * **method**: string
    * Determines the method that is utilized for sampling.
    * Currently only "local_cluster" is supported
  * **lag_time**: float
    * Specifies the timescale of the collective variable
  * **num_anchor_points**: integer
    * Number of different anchorpoints that are sampled
  * **num_samples_per_anchor**: int
    * Number of Monte-Carlo samples per anchor
* **num_coordinates**: integer
  * specifies the number of coordinates of the collective variable that will be saved.
  
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
  
### Details on Datamagement 
After computing a run the results are saved in a corresponding folder. 

#### Network Datamanagement
The network will be saved regardless of "archive". 
The network is always in the results directory with the rest of the intermediate results of the run.
This should not be a problem since a network with 1000 nodes just takes 100kb of space.
This assumes that the network is in the 
[graphml format](https://networkx.org/documentation/stable/reference/readwrite/graphml.html).
Archiving just means that the network is additionally saved in the central data structure.
Specific networks of special interest should be archived.
Otherwise, a new network should be generated every run.
