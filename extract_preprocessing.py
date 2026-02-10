import joblib
import sys
import json
import numpy as np
from imblearn.base import BaseSampler

# 1. Define the LOFResampler class so pickle can find it
class LOFResampler(BaseSampler): 
    _sampling_type = "clean-sampling"
    _parameter_constraints = {}
    def __init__(self, n_neighbors=20, contamination=0.05):
        super().__init__()
    def _fit_resample(self, X, y): return X, y
setattr(sys.modules["__main__"], "LOFResampler", LOFResampler)

def summarize():
    try:
        pipeline = joblib.load("catboost_pipeline.joblib")
        params = {}
        
        # Original features required by the pipeline (from the first step or model)
        if hasattr(pipeline, "feature_names_in_"):
            params["feature_names_in"] = list(pipeline.feature_names_in_)
        
        for name, step in pipeline.steps:
            print(f"Processing step: {name} ({type(step).__name__})")
            
            if hasattr(step, "mean_") and hasattr(step, "scale_"):
                print(" - Found Scaler")
                params["scaler"] = {
                    "mean": step.mean_.tolist(),
                    "scale": step.scale_.tolist()
                }
            
            if hasattr(step, "get_support"):
                print(" - Found Selector")
                support = step.get_support()
                params["selector"] = {
                    "support": support.tolist(), # boolean mask
                    "indices": [int(i) for i in np.where(support)[0]]
                }
                
        with open("preprocessing.json", "w") as f:
            json.dump(params, f, indent=2)
            
        print("SUCCESS: Parameters saved to 'preprocessing.json'")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    summarize()
