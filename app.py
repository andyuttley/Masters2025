import streamlit as st
import pandas as pd
import numpy as np

def main():
    # 1. Header and description
    st.title("Masters Tournament Prediction Analysis")
    st.header("Predicting 2025 Masters Outcomes Using Strokes Gained Data")
    st.write("""
    This app leverages strokes gained (SG) data and historical Masters tournament outcomes to build a linear regression model.
    The model is backtested on previous Masters tournaments and then applied to predict the 2025 tournament.
    A Monte Carlo simulation is used to estimate each player's win and top finish probabilities.
    The predictions are compared against bookies' odds to identify undervalued players with positive expected value.
    The feature importance table shows which SG metrics have the greatest influence on the model's predictions.
    """)

    # 2. Load data from CSV files
    prediction = pd.read_csv("prediction.csv")
    coef_sorted = pd.read_csv("coef_sorted.csv")

    # Dynamic table: filter predictions based on dynamic Odds_rank threshold (default now 100)
    threshold = st.slider("Odds Rank Threshold", min_value=1, max_value=100, value=100, step=1)
    filtered_prediction = prediction[(prediction['priced'] == 'under') & (prediction['Odds_rank'] < threshold)]
    
    st.subheader(f"Filtered Predictions: Underpriced Players with Odds Rank < {threshold}")
    st.dataframe(filtered_prediction)

    # 3. Feature importance table and explanation
    st.subheader("Feature Importance")
    st.write("""
    The feature importance table below shows the impact of each strokes gained (SG) metric on the predicted Masters average score.
    Higher absolute values indicate that a metric has a greater influence on the model's prediction.
    This insight helps to understand which aspects of a player's game are most critical for success at the Masters.
    It also provides context on the relative importance of each performance metric.
    These insights can guide strategy adjustments both on the course and in betting.
    """)
    st.dataframe(coef_sorted)

    # 4. Display the full prediction data
    st.subheader("Full Prediction Data")
    st.dataframe(prediction)

    # 5. Betting Section
    st.subheader("Betting Section")
    st.write("""
    In this section, we calculate the optimal betting allocation for undervalued players based on their expected value.
    The model compares each player's simulated win probability with the bookmaker's odds to determine if they are underpriced.
    For underpriced players, the absolute difference between the simulated win percentage and the odds-implied win probability is used as a weight.
    These weights are normalized to determine the percentage of your total bet pot to allocate to each player.
    A minimum bet of £0.50 is enforced, so only players with an allocation above this threshold are recommended.
    """)
    
    # 6. Let the user input their total bet pot (e.g. £100)
    total_bet = st.number_input("Enter your total bet pot (£)", value=100.0, step=1.0)
    
    # 7. Calculate betting recommendations using the prediction DataFrame
    # Filter to undervalued players meeting the slider threshold
    bet_candidates = prediction[(prediction['priced'] == 'under') & (prediction['Odds_rank'] < threshold)].copy()
    
    # Ensure a 'suggested bet %' column exists; if not, calculate it using absolute predictionVodds_win as weight.
    if 'suggested bet %' not in bet_candidates.columns:
        bet_candidates['suggested bet %'] = bet_candidates['predictionVodds_win'].abs()
    
    # Normalize suggested bet % so that they sum to 100 among underpriced players.
    total_weight = bet_candidates['suggested bet %'].sum()
    bet_candidates['suggested bet %'] = bet_candidates['suggested bet %'] / total_weight * 100
    
    # Compute the bet amount for each candidate.
    bet_candidates['bet_amount'] = total_bet * (bet_candidates['suggested bet %'] / 100)
    
    # Drop players whose bet amount is below £0.50.
    bet_candidates = bet_candidates[bet_candidates['bet_amount'] >= 0.50]
    
    # If any players remain, re-normalize the suggested bet % so that they sum to 100.
    if not bet_candidates.empty:
        new_total = bet_candidates['suggested bet %'].sum()
        bet_candidates['normalized_suggested_bet %'] = bet_candidates['suggested bet %'] / new_total * 100
        bet_candidates['bet_amount'] = total_bet * (bet_candidates['normalized_suggested_bet %'] / 100)
        bet_candidates = bet_candidates.sort_values(by="normalized_suggested_bet %", ascending=False)
    
    st.subheader("Betting Recommendations")
    if bet_candidates.empty:
        st.write("No players meet the minimum bet criteria based on the current threshold and total bet pot.")
    else:
        st.dataframe(bet_candidates[['Player', 'normalized_suggested_bet %', 'bet_amount']])

if __name__ == "__main__":
    main()
