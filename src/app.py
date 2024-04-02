import os
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr
from PIL import Image

# Read the CSV file
df = pd.read_csv('data/player_stats.csv')

# Function to filter DataFrame based on selected map
def filter_df(dataframe, selected_map):
    df_filtered = dataframe[['player', 'agent', 'map', 'kill', 'assist', 'death']]
    df_filtered = df_filtered.groupby(['player', 'agent', 'map'], as_index=False).sum()
    filtered_df = df_filtered[df_filtered['map'] == selected_map]
    return filtered_df

# Function to calculate KDA
def calculate_kda(row):
    return (row['kill'] + row['assist']) / row['death']

# Function to display agent images
def display_agent_image(agent):
    image_path = os.path.join('images', f'{agent}.png')
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=agent, width=150)

# Main function
def main():
    st.header('Agent Recommendation System')

    # Select map
    selected_map = st.selectbox(
        'Select map',
        df['map'].unique(),
        index=0
    )

    if selected_map:
        st.subheader(f'Enter your KDA for each agent on {selected_map}:')

        df_filtered = filter_df(df, selected_map)

        # Calculate KDA and add it to the DataFrame
        df_filtered['KDA'] = df_filtered.apply(calculate_kda, axis=1)

        valorant = pd.pivot_table(df_filtered, index='player', columns='agent', values='KDA', fill_value=0)

        kda = {}
        agents = valorant.columns.tolist()
        num_agents = len(agents)
        num_agents_per_row = 3
        num_rows = -(-num_agents // num_agents_per_row)  # Ceiling division to determine the number of rows
        
        for i in range(num_rows):
            col1, col2, col3 = st.columns(3)
            with col1:
                if i * num_agents_per_row < num_agents:
                    agent = agents[i * num_agents_per_row]
                    display_agent_image(agent)
                    kda[agent] = st.text_input(f'Enter your KDA for {agent}:', key=agent)
                    
            with col2:
                if i * num_agents_per_row + 1 < num_agents:
                    agent = agents[i * num_agents_per_row + 1]
                    display_agent_image(agent)
                    kda[agent] = st.text_input(f'Enter your KDA for {agent}:', key=agent)
                   
            with col3:
                if i * num_agents_per_row + 2 < num_agents:
                    agent = agents[i * num_agents_per_row + 2]
                    display_agent_image(agent)
                    kda[agent] = st.text_input(f'Enter your KDA for {agent}:', key=agent)
                    

        if st.button('Submit'):
            st.write('Calculating...')

            # Calculate similarity between user's KDA profile and other players' KDA profiles
            similarities = valorant.apply(lambda row: pearsonr(row, [float(kda.get(agent, 0)) for agent in valorant.columns])[0], axis=1)

            # Find the top 5 most similar players
            similar_players = similarities.nlargest(6)[1:]  # Exclude the user itself

            # Get the stats of those 5 similar players
            similar_players_stats = df_filtered[df_filtered['player'].isin(similar_players.index)]

            # Sort the agents according to their KDA among those similar players
            sorted_agents = similar_players_stats.groupby('agent')['KDA'].max().nlargest(5)

            # Display the recommended agents
            st.subheader("Top Recommended Agents:")
            for agent, kda_value in sorted_agents.items():
                display_agent_image(agent)
                st.write(f"{agent}")

if __name__ == '__main__':
    main()
