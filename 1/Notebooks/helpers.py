def get_CI(data, CI=95.):

    """
    Retrieve confidence interval.

    Parameters
    ----------
    data : np.ndarray or list
        the data
    CI : float
        The confidence interval. Must be 0<CI<100

    Returns
    -------
    float, float
        Tuple of floats with lower and upper confidence interval bounds
    """

    import numpy as np
    
    lower = np.percentile(data, (100.-CI)/2.)
    upper = np.percentile(data, 100. - (100.-CI)/2.)

    return lower, upper

def fit_model(model, n_evaluations, test_size, dataset, x_name, y_name, properties):
    """
    Fit a model and return report statistics.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        scikit-learn model that will be trained and tested
    n_evaulations : int
        number of evluations carried out for this model.
        Must be > 1
    test_size : float
        Size of the test set given as a fraction of the total
        dataset. Must be > 0
    dataset : pd.DataFrame
        Dataframe contain all dependant and independant varables
    x_name : list
        Name of independant variables (features)
    y_name : str
        Name of dependant variable
    properties : list[str]
        List of properties than should be extraced from the
        model instance after the fit

    Returns
    -------
    dict
        Dictionary with model fitting statistics
    model
        Best model
    """

    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import numpy as np

    r2_train  = np.zeros(n_evaluations, dtype=float)
    r2_test   = np.zeros(n_evaluations, dtype=float)
    mue_train = np.zeros(n_evaluations, dtype=float)
    mue_test  = np.zeros(n_evaluations, dtype=float)
    results   = {p:list() for p in properties}
    best_R2   = -999999999.
    best_model = None
    for i in range(n_evaluations):
        train, test  = train_test_split(dataset, test_size=test_size)
        reg_model    = model.fit(train[x_name], train.pIC50)
        y_pred       = reg_model.predict(test[x_name])
        y_train      = reg_model.predict(train[x_name])
        r2_train[i]  = metrics.r2_score(train.pIC50, y_train)
        r2_test[i]   = metrics.r2_score(test.pIC50, y_pred)
        mue_train[i] = metrics.mean_absolute_error(train[y_name], y_train)
        mue_test[i]  = metrics.mean_absolute_error(test[y_name], y_pred)
        if r2_test[i] > best_R2:
            best_model = reg_model
            best_R2    = r2_test[i]
            

        for p in properties:
            results[p].append(getattr(model, p))

    results["R2_train"]      = r2_train
    results["R2_test"]       = r2_test
    results["MUE_train"]     = mue_train
    results["MUE_test"]      = mue_test

    return results, reg_model


def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    
    from rdkit.Chem import Descriptors

    key_list   = []
    value_list = []
    counts = 0
    for nm, fn in Descriptors._descList:
        # some descriptors seem to occure more than once.
        # Make sure we skip those.
        if nm in key_list:
            continue
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        key_list.append(nm)
        value_list.append(val)
    return key_list, value_list


def calculate_all_features(smiles):
    """
    Generate features for a molecule.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    pandas.Series
        Features for the given molecule
    """
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd
    
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate relevant chemical properties
    key_list, value_list = getMolDescriptors(molecule)

    return pd.Series(value_list, index=key_list)


def calculate_ro5_properties(smiles):
    """
    Test if input molecule (SMILES) fulfills Lipinski's rule of five.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    pandas.Series
        Molecular weight, number of hydrogen bond acceptors/donor and logP value
        and Lipinski's rule of five compliance for input molecule.
    """
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import pandas as pd

    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate Ro5-relevant chemical properties
    molecular_weight = Descriptors.ExactMolWt(molecule)
    n_hba = Descriptors.NumHAcceptors(molecule)
    n_hbd = Descriptors.NumHDonors(molecule)
    logp = Descriptors.MolLogP(molecule)
    nrb  = Descriptors.NumRotatableBonds(molecule)
    # Check if Ro5 conditions fulfilled
    conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5, nrb < 10]
    ro5_fulfilled = sum(conditions) >= 3
    # Return True if no more than one out of four conditions is violated
    return pd.Series(
        [molecular_weight, n_hba, n_hbd, logp, nrb, ro5_fulfilled],
        index=["molecular_weight", "n_hba", "n_hbd", "logp", "N_rot_bonds", "ro5_fulfilled"],
    )


def _scale_by_thresholds(stats, thresholds, scaled_threshold):
    """
    Scale values for different properties that have each an individually defined threshold.

    Parameters
    ----------
    stats : pd.DataFrame
        Dataframe with "mean" and "std" (columns) for each physicochemical property (rows).
    thresholds : dict of str: int
        Thresholds defined for each property.
    scaled_threshold : int or float
        Scaled thresholds across all properties.

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled means and standard deviations for each physiochemical property.
    """
    
    # Raise error if scaling keys and data_stats indicies are not matching
    for property_name in stats.index:
        if property_name not in thresholds.keys():
            raise KeyError(f"Add property '{property_name}' to scaling variable.")
    # Scale property data
    stats_scaled = stats.apply(lambda x: x / thresholds[x.name] * scaled_threshold, axis=1)
    return stats_scaled


def _define_radial_axes_angles(n_axes):
    """Define angles (radians) for radial (x-)axes depending on the number of axes."""
    
    import math
    x_angles = [i / float(n_axes) * 2 * math.pi for i in range(n_axes)]
    x_angles += x_angles[:1]
    return x_angles


def plot_radar(
    y,
    thresholds,
    scaled_threshold,
    properties_labels,
    y_max=None,
    output_path=None,
    title=None,
):
    """
    Plot a radar chart based on the mean and standard deviation of a data set's properties.

    Parameters
    ----------
    y : pd.DataFrame
        Dataframe with "mean" and "std" (columns) for each physicochemical property (rows).
    thresholds : dict of str: int
        Thresholds defined for each property.
    scaled_threshold : int or float
        Scaled thresholds across all properties.
    properties_labels : list of str
        List of property names to be used as labels in the plot.
    y_max : None or int or float
        Set maximum y value. If None, let matplotlib decide.
    output_path : None or pathlib.Path
        If not None, save plot to file.
    title: None or str,
        If None, set no title.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    import math

    # Define radial x-axes angles -- uses our helper function!
    x = _define_radial_axes_angles(len(y))
    # Scale y-axis values with respect to a defined threshold -- uses our helper function!
    y = _scale_by_thresholds(y, thresholds, scaled_threshold)
    # Since our chart will be circular we append the first value of each property to the end
    y = pd.concat([y, y.head(1)])

    # Set figure and subplot axis
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    # Plot data
    ax.fill(x, [scaled_threshold] * len(x), "cornflowerblue", alpha=0.2)
    ax.plot(x, y["mean"], "b", lw=3, ls="-")
    ax.plot(x, y["mean"] + y["std"], "orange", lw=2, ls="--")
    ax.plot(x, y["mean"] - y["std"], "orange", lw=2, ls="-.")

    # From here on, we only do plot cosmetics
    # Set 0° to 12 o'clock
    ax.set_theta_offset(math.pi / 2)
    # Set clockwise rotation
    ax.set_theta_direction(-1)

    # Set y-labels next to 180° radius axis
    ax.set_rlabel_position(180)
    # Set number of radial axes' ticks and remove labels
    plt.xticks(x, [])
    # Get maximal y-ticks value
    if not y_max:
        y_max = int(ax.get_yticks()[-1])
    # Set axes limits
    plt.ylim(0, y_max)
    # Set number and labels of y axis ticks
    plt.yticks(
        range(1, y_max),
        ["5" if i == scaled_threshold else "" for i in range(1, y_max)],
        fontsize=16,
    )

    # Draw ytick labels to make sure they fit properly
    # Note that we use [:1] to exclude the last element which equals the first element (not needed here)
    for i, (angle, label) in enumerate(zip(x[:-1], properties_labels)):
        if angle == 0:
            ha = "center"
        elif 0 < angle < math.pi:
            ha = "left"
        elif angle == math.pi:
            ha = "center"
        else:
            ha = "right"
        ax.text(
            x=angle,
            y=y_max + 1,
            s=label,
            size=14,
            horizontalalignment=ha,
            verticalalignment="center",
        )

    # Add legend relative to top-left plot
    #labels = ("mean", "mean + std", "mean - std", "rule of five area")
    labels = ("rule of five area", "mean", "mean + std", "mean - std")
    ax.legend(labels, loc=(1.1, 0.7), labelspacing=0.3, fontsize=16)
    
    ax.set_title(title, size=18, pad=50)

    # Save plot - use bbox_inches to include text boxes
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)

    plt.show()
    

def calculate_mean_std(dataframe):
    """
    Calculate the mean and standard deviation of a dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Properties (columns) for a set of items (rows).

    Returns
    -------
    pd.DataFrame
        Mean and standard deviation (columns) for different properties (rows).
    """
    # Generate descriptive statistics for property columns
    stats = dataframe.describe()
    # Transpose DataFrame (statistical measures = columns)
    stats = stats.T
    # Select mean and standard deviation
    stats = stats[["mean", "std"]]
    return stats
