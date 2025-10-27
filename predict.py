import pickle

# 1. Load the pipeline
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# 2. Prepare the input record
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# 3. Make the prediction (probability)
probability = pipeline.predict_proba([record])[0, 1]  # index 1 = probability of conversion

print(f"Probability that the lead will convert: {probability:.3f}")
