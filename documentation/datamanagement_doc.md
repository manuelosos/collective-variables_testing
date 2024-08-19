### Datamanagement Documentation

The results of the runs should be archived in one central datastructure.
The datamanagement.py script provides functions to archive this.
With "archive_run_result" the results of one run will be saved in the corresponding directory.
The run-parameters and some key result values will be saved in a separate csv table.
The table has following header:

dynamic_model, r_ab, r_ba, rt_ab, rt_ba, network_id, network_model, num_nodes, lag_time, num_anchor_points,
        num_samples_per_anchor, cv_dim num_coordinates, dimension_estimate, remarks

The remarks attribute contains additional information about the run which cannot be classifies otherwise. 
The additional information will be provided with codes. The following codes are used.
* `slb`: sigma lower bound: This will be set if the lower bound of the bandwidth sigma for the diffusion map is applied. 
In this case the dimension estimate may be to low. 
        
Specific parameters of Network will be saved in the additional information json file in the result folder.

