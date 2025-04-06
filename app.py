import streamlit as st
import pandas as pd

def main():
    # 1. Header and description
    st.title("Masters Tournament Prediction Analysis")
    st.header("Predicting 2025 Masters Outcomes Using Strokes Gained Data")
    st.write("""
    This app leverages strokes gained (SG) data and historical Masters tournament outcomes to build a linear regression model. 
    The model is backtested on previous Masters tournaments and then applied to predict the 2025 tournament. 
    A Monte Carlo simulation is used to estimate each player's win and top finish probabilities.
    The predictions are compared against bookies' odds to identify undervalued players.
    Additionally, the feature importance table shows which SG metrics have the greatest influence on the predictions.
    """)

    # 2. Load data from CSV files
    prediction = pd.read_csv("prediction.csv")
    coef_sorted = pd.read_csv("coef_sorted.csv")
    
    # Dynamic table: filter predictions based on dynamic Odds_rank threshold
    threshold = st.slider("Odds Rank Threshold", min_value=1, max_value=100, value=30, step=1)
    filtered_prediction = prediction[(prediction['priced'] == 'under') & (prediction['Odds_rank'] < threshold)]
    
    st.subheader(f"Filtered Predictions: Underpriced Players with Odds Rank < {threshold}")
    st.dataframe(filtered_prediction)
    
    # 3. Feature importance table and explanation
    st.subheader("Feature Importance")
    st.write("""
    The feature importance table below shows the impact of each strokes gained (SG) metric on the predicted Masters average score. 
    Higher absolute values indicate that a metric has a greater influence on the model's prediction.
    """)
    st.dataframe(coef_sorted)
    
    # 4. Display the full prediction data
    st.subheader("Full Prediction Data")
    st.dataframe(prediction)

if __name__ == "__main__":
    main()
