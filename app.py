import streamlit as st
import pandas as pd
import numpy as np

def main():
    # Display Augusta image at the top
    st.image("augusta_image.jpeg", use_container_width=True)
    
    # Header and description
    st.title("Masters Tournament Prediction Analysis")
    st.header("Predicting 2025 Masters Outcomes Using Strokes Gained Data")
    st.write("""
    This app leverages strokes gained (SG) data and historical Masters tournament outcomes to build a linear regression model.
    The model is backtested on previous Masters tournaments and then applied to predict the 2025 tournament.
    A Monte Carlo simulation is used to estimate each player's win and top finish probabilities.
    The predictions are compared against bookies' odds to identify undervalued players with positive expected value.
    The feature importance table shows which SG metrics have the greatest influence on the model's predictions.
    """)
    
    # Load data from CSV files
    prediction = pd.read_csv("prediction.csv")
    coef_sorted = pd.read_csv("coef_sorted.csv")
    
    # Determine the default slider value as the distinct count of players in prediction
    n_golfers = prediction['Player'].nunique()
    # Slider label changed here:
    threshold = st.slider("Choose number of golfers to include", min_value=1, max_value=n_golfers, value=n_golfers, step=1)
    
    # Dynamic table: Filter predictions for underpriced players meeting the threshold
    filtered_prediction = prediction[(prediction['priced'] == 'under') & (prediction['Odds_rank'] < threshold)]
    
    # Reorder columns: first "Player", then "simulated_win_percentage" renamed to "Model Prediction", 
    # then "odds_Win Probability (%)" renamed to "Odds Prediction", followed by the rest.
    display_df = filtered_prediction.copy()
    display_df = display_df.rename(columns={
        "simulated_win_percentage": "Model Prediction",
        "odds_Win Probability (%)": "Odds Prediction"
    })
    fixed_cols = ["Player", "Model Prediction", "Odds Prediction"]
    other_cols = [col for col in display_df.columns if col not in fixed_cols]
    new_order = fixed_cols + other_cols
    display_df = display_df[new_order]
    
    st.subheader(f"Only include top {threshold} golfers")
    st.dataframe(display_df)
    
    # Betting Section
    st.subheader("Betting Section")
    st.write("""
    In this section, we calculate the optimal betting allocation for undervalued players based on their expected value (EV).
    We identify players who are underpriced by comparing the simulated win percentages with the bookmakers' implied probabilities.
    For these players, the absolute difference (predictionVodds_win) serves as a weight for allocating the bet.
    These weights are then normalized so that the suggested bet percentages sum to 100% of your total pot.
    A minimum bet of £0.50 is enforced, and the final recommended bets are adjusted accordingly.
    """)
    
    # Let the user input their total bet pot (e.g. £100)
    total_bet = st.number_input("Enter your total bet pot (£)", value=100.0, step=1.0)
    
    # Calculate betting recommendations using the prediction DataFrame
    bet_candidates = prediction[(prediction['priced'] == 'under') & (prediction['Odds_rank'] < threshold)].copy()
    
    # Create 'suggested bet %' based on the absolute value of predictionVodds_win
    if 'suggested bet %' not in bet_candidates.columns:
        bet_candidates['suggested bet %'] = bet_candidates['predictionVodds_win'].abs()
    
    total_weight = bet_candidates['suggested bet %'].sum()
    bet_candidates['suggested bet %'] = bet_candidates['suggested bet %'] / total_weight * 100
    
    # Compute the bet amount for each candidate
    bet_candidates['bet_amount'] = total_bet * (bet_candidates['suggested bet %'] / 100)
    
    # Filter out players with a bet amount below £0.50
    bet_candidates = bet_candidates[bet_candidates['bet_amount'] >= 0.50]
    
    # If players remain, re-normalize suggested bet % so that they sum to 100%
    if not bet_candidates.empty:
        new_total = bet_candidates['suggested bet %'].sum()
        bet_candidates['normalized_suggested_bet %'] = bet_candidates['suggested bet %'] / new_total * 100
        bet_candidates['bet_amount'] = total_bet * (bet_candidates['normalized_suggested_bet %'] / 100)
    
    # Format bet_amount as currency
    bet_candidates['bet_amount'] = bet_candidates['bet_amount'].apply(lambda x: f"£{x:.2f}")
    
    st.subheader("Betting Recommendations")
    if bet_candidates.empty:
        st.write("No players meet the minimum bet criteria based on the current threshold and total bet pot.")
    else:
        st.dataframe(bet_candidates[['Player', 'normalized_suggested_bet %', 'bet_amount']])
    
    # Display golfmoney image after the betting section
    st.image("golfmoney.jpg", use_container_width=True)
    
    # Feature Importance section (moved to bottom)
    st.subheader("Feature Importance")
    st.write("""
    The feature importance table below shows the impact of each strokes gained (SG) metric on the predicted Masters average score.
    Higher absolute values indicate a greater influence on the model's prediction.
    This information provides insight into which aspects of a player's game are most crucial for success.
    It also helps in understanding the model's decision-making process.
    These insights can guide strategy adjustments on the course and in betting decisions.
    """)
    st.dataframe(coef_sorted)
    
    # Full Prediction Data section (moved to bottom)
    st.subheader("Full Prediction Data")
    st.dataframe(prediction)

if __name__ == "__main__":
    main()
