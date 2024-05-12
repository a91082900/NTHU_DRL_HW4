import numpy as np

def as_list(obs_dict, LENGTH0=1):
    res = []

    # target velocity field (in body frame)
    v_tgt = np.ndarray.flatten(obs_dict['v_tgt_field'])
    res += v_tgt.tolist()

    res.append(obs_dict['pelvis']['height'])
    res.append(obs_dict['pelvis']['pitch'])
    res.append(obs_dict['pelvis']['roll'])
    res.append(obs_dict['pelvis']['vel'][0]/LENGTH0)
    res.append(obs_dict['pelvis']['vel'][1]/LENGTH0)
    res.append(obs_dict['pelvis']['vel'][2]/LENGTH0)
    res.append(obs_dict['pelvis']['vel'][3])
    res.append(obs_dict['pelvis']['vel'][4])
    res.append(obs_dict['pelvis']['vel'][5])

    for leg in ['r_leg', 'l_leg']:
        res += obs_dict[leg]['ground_reaction_forces']
        res.append(obs_dict[leg]['joint']['hip_abd'])
        res.append(obs_dict[leg]['joint']['hip'])
        res.append(obs_dict[leg]['joint']['knee'])
        res.append(obs_dict[leg]['joint']['ankle'])
        res.append(obs_dict[leg]['d_joint']['hip_abd'])
        res.append(obs_dict[leg]['d_joint']['hip'])
        res.append(obs_dict[leg]['d_joint']['knee'])
        res.append(obs_dict[leg]['d_joint']['ankle'])
        for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
            res.append(obs_dict[leg][MUS]['f'])
            res.append(obs_dict[leg][MUS]['l'])
            res.append(obs_dict[leg][MUS]['v'])
    return res