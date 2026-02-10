import joblib
import sys
import os
from imblearn.base import BaseSampler

# 1. Define the LOFResampler class so pickle can find it
class LOFResampler(BaseSampler):
    _sampling_type = "clean-sampling"
    _parameter_constraints = {}
    def __init__(self, n_neighbors=20, contamination=0.05):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.sampling_strategy = "auto"
    def _fit_resample(self, X, y):
        return X, y

setattr(sys.modules["__main__"], "LOFResampler", LOFResampler)

def extract():
    print("Loading pipeline...")
    try:
        pipeline = joblib.load("catboost_pipeline.joblib")
        print("Pipeline loaded.")
        
        # Assume the classifier is the last step
        # Check if it's a pipeline
        if hasattr(pipeline, "steps"):
            final_step = pipeline.steps[-1][1]
            print(f"Found final step: {type(final_step)}")
            
            if hasattr(final_step, "save_model"):
                final_step.save_model("catboost_model.cbm")
                print("SUCCESS: Model saved to 'catboost_model.cbm'")
            else:
                print("ERROR: Final step does not have save_model method.")
        else:
            print("ERROR: Loaded object is not a pipeline (no .steps attribute).")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract()
