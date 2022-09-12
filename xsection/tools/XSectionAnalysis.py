import numpy as np

def TrapezoidQuadrature(x : np.array, y : np.array) -> float:
    return np.sum(np.convolve(y, np.ones(2) / 2, mode='valid') * np.convolve(x, np.array([1, -1]), mode='valid'))

def TrapezoidQuadratePropUncertainty(x : np.array, y_stderr : np.array) -> float:
    return np.sqrt(np.sum((np.sqrt(np.convolve(y_stderr ** 2, np.ones(2) / 2, mode='valid')) * np.convolve(x, np.array([1, -1]), mode='valid') / 2) ** 2))
    
# Returns sigma_tot in mb
def SigmaTot(b : np.array, S_AB : np.array, S_AB_stderr : np.array) -> tuple:
    sigma_tot = 10 * 2 * np.pi * 2 * TrapezoidQuadrature(b, b * (1 - S_AB))
    uncertainty = 10 * TrapezoidQuadratePropUncertainty(b, b * S_AB_stderr)
    
    return (sigma_tot, uncertainty)
    
# Returns sigma_el in mb
def SigmaEl(b : np.array, S_AB : np.array, S_AB_stderr : np.array) -> tuple:
    sigma_el = 10 * 2 * np.pi * TrapezoidQuadrature(b, b * (1 - S_AB) ** 2)
    uncertainty = 10 * TrapezoidQuadratePropUncertainty(b, b * np.sqrt(2) * (1 - S_AB) * S_AB_stderr)
    
    return (sigma_el, uncertainty)
    
# Returns sigma_el in mb
def SigmaRxn(b : np.array, S_AB : np.array, S_AB_stderr : np.array) -> tuple:
    sigma_rxn = 10 * 2 * np.pi * TrapezoidQuadrature(b, b * (1 - S_AB ** 2))
    uncertainty = 10 * TrapezoidQuadratePropUncertainty(b, b *  np.sqrt(2) * S_AB * S_AB_stderr)
    
    return (sigma_rxn, uncertainty)