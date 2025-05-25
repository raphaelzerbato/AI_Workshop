from instrument_creation.instrument_creation_functions import extract_instruments, get_non_collinear_instruments

def init_instrument_creation(df_one_hot, exogenous_var, twodegree_polynomial_instruments=False):
    """
    Initialize instrument creation functions and configurations.
    This function is called at the start of the instrument creation pipeline
    and calls all the functions from instrument_creation.instrument_creation_functions
    """
    # Initialize any necessary configurations or parameters here
    # Extract instruments
    Z = extract_instruments(df_one_hot, exogenous_var,
                            marketvar = 'marketid',
                            twodegree_polynomial_instruments = twodegree_polynomial_instruments)

    # Reduce instruments to collinearity-proof instruments
    Z, instrvars = get_non_collinear_instruments(Z, df_one_hot[exogenous_var])

    return Z, instrvars