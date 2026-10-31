'''
The Challenge: PastHires Candidate Selection using Random Forest
================================================================
Goal: Predict whether a prospective job candidate will be HIRED (Yes/No)
      using a Random Forest Classifier ensemble.
'''

from pathlib import Path
import runpy

# Forward execution to O07_problem_to_solve.py
target_script = Path(__file__).parent / "O07_problem_to_solve.py"
runpy.run_path(str(target_script), run_name="__main__")
