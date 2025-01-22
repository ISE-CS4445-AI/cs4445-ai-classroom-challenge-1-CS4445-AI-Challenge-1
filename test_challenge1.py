# DO NOT CHANGE THIS FILE! If any code is changed, the instructor will be notified on Github classroom's assignment dashboard.

# Common imports
import os
import numpy as np

# sklearn imports
from sklearn.datasets import load_breast_cancer

# Importing student's solution
import challenge_1_export  # the python file created after nbconvert

def test_mcqfunction_1():
    # Test MCQ function 1's answer
    q1_ans = os.environ.get("C1_MCQF_1", "")
    assert q1_ans, "No C1_MCQF_1 found in environment!"
    
    assert challenge_1_export.answer_q1() == q1_ans, "Wrong answer for MCQ 1!"

def test_mcqfunction_2():
    # Test MCQ function 2's answer
    q2_ans = os.environ.get("C1_MCQF_2", "")
    assert q2_ans, "No C1_MCQF_2 found in environment!"

    assert challenge_1_export.answer_q2() == q2_ans, "Wrong answer for MCQ 2!"

def test_mcqfunction_3():
    # Test MCQ function 3's answer
    q3_ans = os.environ.get("C1_MCQF_3", "")
    assert q3_ans, "No C1_MCQF_3 found in environment!"
    
    assert challenge_1_export.answer_q3() == q3_ans, "Wrong answer for MCQ 3!"

def test_hidden_data_accuracy():
    # Train the model on partial dataset
    model = challenge_1_export.train_cancer_classifier(random_state=42)

    # Retrieve the org-level variable containing the unseen indexes
    unseen_str = os.environ.get("C1_TESTSET_INDICES", "")
    assert unseen_str, "No C1_TESTSET_INDICES found in environment!"

    row_list = list(map(int, unseen_str.split(",")))
    
    # loading the *full* dataset
    data = load_breast_cancer()
    X_full, y_full = data.data, data.target
    
    X_hidden = X_full[row_list]
    y_hidden = y_full[row_list]
    
    # Evaluate the student's model and get accuracy score
    preds_hidden, hidden_acc = challenge_1_export.predict_cancer(model, X_hidden, y_hidden)
    print(f"Hidden test accuracy: {hidden_acc:.3f}")
    
    assert hidden_acc >= 0.90, f"Hidden test accuracy < 0.90! Got {hidden_acc:.3f}"