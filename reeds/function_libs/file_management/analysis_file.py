import copy

#top - layer dict
phase_dict = {
    "converged" : None,
    "steps": {}
}
# supsteps of phases:
step_dict = {
        "executed": False,
        "success" : None,
        "path": None,
        "next_dir": None,
        "last_jobID":None
}
## special step in optimization
optimization_iteration_addittion = {
    "optimization_paramter": None,
    "s-parameters": None,
    "energy-offsets": None,
    "roundTrips": None,
    "avg_roundTripTime": None,
    "samplingDistribution": None,
    "optSamplingDist_Dev": None,
}

def get_step_dict(step_name):
    result_dict = copy.copy(step_dict)
    result_dict['name'] = step_name
    return result_dict

def get_optimization_dict(step_name):
    result_dict =get_step_dict(step_name)
    result_dict = {**result_dict, **optimization_iteration_addittion}
    return result_dict

pipeline_phases = ["parameter_exploration", "parameter_optimization", "production"]
pipeline_exploration_steps = ["opt_states", "find_lower_bound", "eoff_estimation"]
pipeline_optimization_steps = ["s_optimization", "eoff_rebalancing"]

def get_standard_pipeline():
    standard_pipeline = {pipeline_phase: copy.copy(phase_dict) for pipeline_phase in pipeline_phases }

    exploration_steps = {ind : get_step_dict(step) for ind, step in enumerate(pipeline_exploration_steps)}
    standard_pipeline['parameter_exploration']["steps"] = exploration_steps

    optimization_steps = {ind : get_optimization_dict(step) for ind, step in enumerate(pipeline_optimization_steps)}
    standard_pipeline['parameter_optimization']["steps"] = optimization_steps
    standard_pipeline['parameter_optimization']["current_numberOfRoundtrips"] = None
    standard_pipeline['parameter_optimization']["current_avgRoundtripTime"] = None
    standard_pipeline['parameter_optimization']["current_samplingDistribution"] = None
    standard_pipeline['parameter_optimization']["current_devationOptimalSamplingDist"] = None
    #standard_pipeline['parameter_optimization']["convergedRule_numberOfRoundtrips"] = lambda nRT: nRT>5
    #standard_pipeline['parameter_optimization']["convergedRule_avgRoundtripTime"] = lambda RT_time: RT_time<500
    #standard_pipeline['parameter_optimization']["convergedRule_devationOptimalSamplingDist"] = lambda MAE_distSampling: MAE_distSampling<10
    #standard_pipeline['parameter_optimization']["convergedRule_stateSampling"] = lambda state_sampling_distribution: all([state_sampling>0.01 for state_sampling in state_sampling_distribution])


    standard_pipeline['parameter_optimization']["current_devationOptimalSamplingDist"] = None


    print(standard_pipeline)
    standard_pipeline["production"] = copy.copy(step_dict)
    return standard_pipeline