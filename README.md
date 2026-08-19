# AI-ML

## Basic AI/ML Pipeline Practices

Use this checklist to build reliable, reproducible AI/ML workflows:

1. **Define the problem and metric first**
   - Write a clear objective.
   - Pick one primary success metric (for example: F1, AUC, RMSE).

2. **Version data and code**
   - Keep raw data immutable.
   - Track dataset versions and model training code changes.

3. **Split data correctly**
   - Create train/validation/test splits early.
   - Prevent leakage between splits.

4. **Build repeatable preprocessing**
   - Keep preprocessing in a reusable pipeline.
   - Apply the same transformations in training and inference.

5. **Establish a baseline model**
   - Start simple before using complex models.
   - Compare every new model against the baseline.

6. **Track experiments**
   - Record hyperparameters, metrics, and artifacts.
   - Keep enough metadata to reproduce results.

7. **Validate before deployment**
   - Test model quality, latency, and failure behavior.
   - Add basic checks for drift and data schema changes.

8. **Monitor in production**
   - Track prediction quality and service health.
   - Set alerts and rollback criteria.

9. **Automate with CI/CD for ML**
   - Run tests for data processing and model code.
   - Automate training/evaluation pipelines where practical.

10. **Prioritize security and governance**
    - Remove secrets from code and configs.
    - Document model assumptions, risks, and limitations.