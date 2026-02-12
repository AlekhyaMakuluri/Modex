import os
import sys
import math
import time
import json
import logging
import numpy as np
import scipy as sp


import sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from concrete.ml.sklearn import (
        LinearRegression as CML_LinearRegression,
        Ridge as CML_Ridge,
        RandomForestRegressor as CML_RandomForestRegressor,
        LogisticRegression as CML_LogisticRegression,
    )
    _HAS_CONCRETE_ML = True
except Exception:
   
    CML_LinearRegression = None
    CML_Ridge = None
    CML_RandomForestRegressor = None
    CML_LogisticRegression = None
    _HAS_CONCRETE_ML = False

try:
    import tenseal as ts
    _HAS_TENSEAL = True
except Exception:
    ts = None
    _HAS_TENSEAL = False


try:
    from Pyfhel import Pyfhel, PyCtxt, PyPtxt
    _HAS_PYFHEL = True
except Exception:
    Pyfhel = None
    PyCtxt = None
    PyPtxt = None
    _HAS_PYFHEL = False

def init_tenseal_ckks(poly_mod_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], scale=2**40):
    
    if not _HAS_TENSEAL:
        raise RuntimeError("TenSEAL not installed. pip install tenseal")
   
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_mod_degree=poly_mod_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    ctx.global_scale = scale
    
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx

def init_pyfhel_ckks(n=2**13, scale=2**30, sec=128):
    
    if not _HAS_PYFHEL:
        raise RuntimeError("Pyfhel not installed. pip install pyfhel")
    pyf = Pyfhel()             
    pyf.contextGen(scheme='CKKS', n=n, scale=scale, sec=sec)
    pyf.keyGen()
    pyf.relinKeyGen()
    pyf.rotateKeyGen()
    return pyf


def tenseal_encrypt_vector(ctx, vector):
    """Encrypt a numpy vector with TenSEAL CKKS context (returns CKKSVector)"""
    if not _HAS_TENSEAL:
        raise RuntimeError("TenSEAL not installed.")
    return ts.ckks_vector(ctx, vector)

def tenseal_decrypt_vector(ct_vector):
    """Decrypt a TenSEAL CKKSVector to numpy array"""
    if not _HAS_TENSEAL:
        raise RuntimeError("TenSEAL not installed.")
    return np.array(ct_vector.decrypt())

def pyfhel_encrypt_vector(pyf: "Pyfhel", vector):
    """Encrypt with Pyfhel CKKS (returns PyCtxt)"""
    if not _HAS_PYFHEL:
        raise RuntimeError("Pyfhel not installed.")
    ptxt = pyf.encodeFrac(vector.tolist())
    ctxt = pyf.encryptFrac(ptxt)
    return ctxt

def pyfhel_decrypt_vector(pyf: "Pyfhel", ctxt: "PyCtxt"):
    """Decrypt Pyfhel CKKS ciphertext to numpy array"""
    if not _HAS_PYFHEL:
        raise RuntimeError("Pyfhel not installed.")
    ptxt = pyf.decryptFrac(ctxt)
    return np.array(ptxt)


def print_he_backends():
    print("=== Homomorphic Encryption Backends Availability ===")
    print(f"Concrete-ML available: {_HAS_CONCRETE_ML}")
    print(f"TenSEAL (CKKS) available: {_HAS_TENSEAL}")
    print(f"Pyfhel (CKKS/BFV/BGV) available: {_HAS_PYFHEL}")
    print(f"PyTorch available: {_HAS_TORCH}")
    print("===================================================")


if __name__ == "__main__":
    print_he_backends()

   
    if _HAS_TENSEAL:
        ctx = init_tenseal_ckks()
        v = np.array([1.1, 2.2, 3.3])
        enc = tenseal_encrypt_vector(ctx, v)
        dec = tenseal_decrypt_vector(enc)
        print("TenSEAL roundtrip:", dec)

    
    if _HAS_PYFHEL:
        pyf = init_pyfhel_ckks()
        v = np.array([0.5, -1.25, 3.0])
        c = pyfhel_encrypt_vector(pyf, v)
        d = pyfhel_decrypt_vector(pyf, c)
        print("Pyfhel roundtrip:", d)

    
    if not _HAS_CONCRETE_ML:
        print("\nConcrete-ML not installed. Install: pip install concrete-ml")
    else:
        print("\nConcrete-ML is available. Use concrete.ml.sklearn.* models and .compile(...) before FHE execute.")
