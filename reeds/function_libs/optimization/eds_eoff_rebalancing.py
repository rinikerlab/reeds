import numpy as np
from typing import Dict
from scipy import constants as const


def rebalance_eoffs_directCounting(old_eoffs: np.array, sampling_stat: Dict[int, Dict[str, Dict[int, float]]],
                                   temperature: float = 298, pseudo_count: float = None, learningFactor: float = 1,
                                   sampling_type: str = "max_contributing_state",
                                   correct_for_s1_only: bool = True, verbose: bool = True) -> np.array:
    """
    This function uses a direct counting approach for (RE-)EDS in order to improve the sampling weights of the min-state sampling ratios.
    It compares the optimal maximal contributing sampling distribution for the system with the actual one from a simulation.
    According to this the energy offsets will be corrected/rebalanced in order to fit better to the desired otpimal sampling distribution.

    one sided function:
        $\DeltaE^R_i = - k_b T * \ln(\frac{p_{dominatingState}}{p_{optimalSamplingState}})$

    two sided function:
        \begin{equation}
                \Delta E^R_i =
                \begin{cases}
                    - k_b T * \ln(\frac{p^{metric}_{samp}}{p_{optSampling}}) & \text{if $p^{metric}_{samp}<p_{optSampling}$} \\
                    k_b T * \ln(\frac{p^{metric}_{samp}}{p_{optSampling}})-c & \text{otherwise}
                \end{cases}
        \end{equation}

    Requirements:
        for a RE-EDS run the efficiency of the approach is best, if round trips are present!
        This approach might require multiple iterations to result in good converged Eoffs.

    Parameters
    ----------
    old_eoffs : np.array
        give the old energy offsets as a 2D array
        (np.array([s-value1[Eoff1, Eoff2, ...], s-value2[Eoff1, Eoff2, ... ], ...])
    sampling_stat : Dict[int, float]
        sampling statistics from reeds-git function: reeds.function_libs.analysis.sampling.sampling_analysis
    temperature : float, optional
        Temperature defines the possible intensity of the correction. (default: 298K)
    pseudo_count : float, optional
        pseudo count increases the robustness of the approach. It sets a minimum cap for the minimum Sampling ratios and therefore avoids division by 0. (default: None)
        If None, the pseudo count is capped to a difference by a factor of 10 (max abs. correction: 5.7 kJ) from the optimal sampling.
    learningFactor : float, optional
        can be used to tune the how much the eoffs should be corrected in this run. actually equal to temperature, but might ease the use :) (default: 1 - turned off)
    correct_for_s1_only : bool, optional
        Should the energy offsets be corrected by only looking of the sampling behaviour of the first replica? If False, all replica will get individual eoffs. (default: True)
    verbose : bool, optional
        more information than before ;)  (default: True)

    Returns
    -------
    np.array
        returns the new rebalanced energy offsets as a 2D array.
        np.array([s-value1[eoff1, eoff2, ...], s-value2[eoff1, eoff2, ...], ...])
    """

    # Sampling filter, which sampling dists, are taken into account.
    samplingDists = []
    for key in sorted(sampling_stat):
        samplingDists.append(list(sampling_stat[key][sampling_type].values()))
        if (correct_for_s1_only):
            break
    samplingDists = np.array(samplingDists)

    if (verbose):
        print("Sampling:")
        print(samplingDists)
        print()

    # calculate Energy offset corrections:
    dEoff_corrected_matrix = calculate_Eoff_Correction(samplingDists=samplingDists,
                                                       pseudo_count=pseudo_count, learningFactor=learningFactor,
                                                       temperature=temperature,
                                                       verbose=verbose)

    # correct eoffs
    if (verbose):
        print("Eoff correction:")
        print(dEoff_corrected_matrix)
        print()
    new_eoffs = old_eoffs + dEoff_corrected_matrix

    if (verbose):
        print("old Eoffs:")
        print(old_eoffs)
        print()
        print("corrected Eoffs:")
        print(new_eoffs)
        print()

    return new_eoffs


def calculate_Eoff_Correction(samplingDists: np.array,
                              temperature: float = 298, learningFactor: float = 1,  pseudo_count: float = None,
                              _shift_eoff_zero: bool = True, _nstates:int =None,
                              verbose: bool = False) -> np.array:
    """
    This function is calculating the correction factor for the energy offset rebalancing Approach.:

    one sided function:
        this approach uses a boltzman factor to correct the energy offsets such that the state sampling balance is closing on the optimalSamplingState.
        This approach can push undersampled states.

        $\DeltaE^R_i = - k_b T * \ln(\frac{p_{dominatingState}}{p_{optimalSamplingState}})$

    two sided function:
        this approach uses a boltzman factor and an inverse boltzmann factor to correct the energy offsets such that the state sampling balance is closing on the optimalSamplingState.
        This appraoch can push less sampled states and punish oversampled states.

        \begin{equation}
                \Delta E^R_i =
                \begin{cases}
                    - k_b T * \ln(\frac{p^{metric}_{samp}}{p_{optSampling}}) & \text{if $p^{metric}_{samp}<p_{optSampling}$} \\
                    k_b T * \ln(\frac{p^{metric}_{samp}}{p_{optSampling}})-c & \text{otherwise}
                \end{cases}
        \end{equation}


    Parameters
    ----------
    sampling_stat : Dict[int, float]
        sampling statistics from reeds-git function: reeds.function_libs.analysis.sampling.sampling_analysis
    pseudo_count : float, optional
        pseudo count increases the robustness of the approach. It sets a minimum cap for the minimum Sampling ratios and therefore avoids division by 0. (default: None)
        If None, the pseudo count is capped to a difference by a factor of 10 (max abs. correction: 5.7 kJ) from the optimal sampling.
    temperature : float, optional
        Temperature defines the possible intensity of the correction. (default: 298K)
    learningFactor : float, optional
        can be used to tune the how much the eoffs should be corrected in this run. actually equal to temperature, but might ease the use :) (default: 1 - turned off)
    _shift_eoff_zero : bool, optional
        shift the eoffs such that eoffset of state 1 is 0. This makes it easier to compare the eoffs with the final ddG.However purely cosmetic. (default: True)
    _nstates : int, optional
        this option is for developing purposes, you shall only play with with it, if you intend to visualize the correction as function of the sampling ps. (default: None)
    verbose : bool, optional
        more information than before ;)  (default: True)

    Returns
    -------
    np.array
        corrections that need to be applied to the Eoffs (np.array(np.array()))

    """
    if (verbose): print("Calculating Eoff correction: ")

    # Get Parameters / Setup:
    beta = 1 / (temperature * (const.R / 1000))  # divide by 1000 => kJ
    if(_nstates is None):
        optimal_sampling = 1 / len(samplingDists[0])
    else:
        optimal_sampling = 1 / _nstates

    if (pseudo_count is None):  # if no pseudo count was defined, chose factor to maximally resemble max factor of 10
        pseudo_count = optimal_sampling / 10

    ## Get correction functional
    if (verbose): print("\tApproach: One-Sided")
    max_pseudocount = 1  # optimal_sampling*10 #Don't use max_pseudocount for single sided
    correctionFunction = lambda x: -(1 / beta) * np.log((x+pseudo_count) / (optimal_sampling+pseudo_count))

    if (verbose): print("\t\tOptimal Sampling fraction: ", optimal_sampling)

    correctionFunctionVector = lambda x: list(map(correctionFunction, x))

    # DO                                    
    # Apply boltzman correction Factor
    if (verbose):
        maxMinValues = np.array([pseudo_count, optimal_sampling, max_pseudocount])
        maxMinCorrection = np.round(correctionFunctionVector(maxMinValues), 2)

        print("\tCorrection intensity:")
        print("\t\tPseudo Count: Min=", pseudo_count, 'Max=', max_pseudocount)
        print("\t\tTemperature:", temperature)
        print("\t\tCorrection: MinMax=", maxMinCorrection[0], "Min=", maxMinCorrection[1], "MaxMax=",
              maxMinCorrection[2])
        print("\t\tLearning Factor: ", learningFactor)
        print()

    ##correction is calculated here:
    dEoff_corrected_matrix = np.array(list(map(correctionFunctionVector, samplingDists)))
    print("Sampling Distributions")
    print(samplingDists)
    print()

    print("raw correction")
    print(dEoff_corrected_matrix)
    print()

    # some eoff shifting (such that eoff 0 is)
    if (_shift_eoff_zero):
        for i in range(len(dEoff_corrected_matrix)):
            dEoff_corrected_matrix[i] -= dEoff_corrected_matrix[:, 0][i]

    # apply learning factor
    dEoff_corrected_matrix *= learningFactor
    if (verbose):

        print("Correction after learning Factor:")
        print(dEoff_corrected_matrix)
        print()

    return dEoff_corrected_matrix
