import json
import pandas as pd
import plotly.graph_objects as go
import re
from typing import List
import os
# solr library
import unidecode

# bing library for automation image
from bing_image_urls import bing_image_urls

# streamlit libraries
import streamlit as st 
from streamlit_searchbox import st_searchbox

# cosine similarity libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# langchain libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from streamlit_navigation_bar import st_navbar
st.set_page_config(page_title="Player Scouting Recommendation System", page_icon="⚽", layout="wide",initial_sidebar_state="collapsed")


# def main():
#     st.title("My Streamlit App")

#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", ["Generate Scouting Report", "Analysis"])

#     # Navigation logic
#     if page == "Generate Scouting Report":
#         generate_scouting_report()
#     elif page == "Analysis":
#         analysis()


def main():

    # Define pages and styles for the navigation bar
    pages = ["Player Scout", "Team Builder"]
    styles = {
        "nav": {
            "background-color": "rgb(0, 0, 0)",  # Black for a sleek look
            "color": "rgb(255, 255, 255)",  # White text for contrast
            "font-family": "'Roboto', sans-serif",  # Modern, clean font
            "font-size": "1.2em",  # Slightly larger font
        },
        "div": {
            "max-width": "20rem",  # Larger for more content
            "color": "rgb(255, 255, 255)",  # White text for visibility
        },
        "span": {
            "border-radius": "1rem",  # Even larger radius for a modern look
            "color": "rgb(255, 255, 255)",  # White for contrast
            "margin": "0 0.5rem",  # More margin for spacing
            "padding": "0.75rem 1rem",  # More padding for a larger click/touch area
        },
        "active": {
            "background-color": "rgba(255, 255, 255, 0)",  # White with transparency
        },
        "hover": {
            "background-color": "rgba(255, 255, 255, 0.75)",  # White with more transparency
        },
    }



    # Create the navigation bar
    page = st_navbar(pages, styles=styles)

    # Navigation logic
    if page == "Player Scout":
        generate_scouting_report()
    elif page == "Team Builder":
        analysis()
    with st.sidebar:
        st.write("Sidebar")

def generate_scouting_report():
    # used libraries 

    ############### Header #################
    # Set the page width to 'wide' to occupy the full width
    st.markdown("<h1 style='text-align: center;'>⚽🔍 Player Scouting Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>🌟 Comprehensive Player Search and Scouting Reports</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Powered Player Scouting 2024-2025: Scout, Analyze, and Compare Players with Google Gemini API - 👀 This is a Beta version</p>", unsafe_allow_html=True)
    # st.markdown("<p>Our comprehensive player search feature allows you to find players based on detailed criteria. View player statistics, compare similar players, and generate in-depth scouting reports to make informed decisions.</p>", unsafe_allow_html=True)


    ############### Simple Search Engine with Auto-Complete Query Suggestion ##############
    press = False
    choice = None


    if 'expanded' not in st.session_state:
        st.session_state.expanded = True

    if 'choice' not in st.session_state:
        st.session_state.choice = None


    df_player = pd.read_csv('data/football-player-stats-2023.csv')


    def remove_accents(text: str) -> str:
        return unidecode.unidecode(text)

    def search_csv(searchterm: str) -> List[str]:
        if searchterm:
            normalized_searchterm = remove_accents(searchterm.lower())
            df_player['NormalizedPlayer'] = df_player['Player'].apply(lambda x: remove_accents(x.lower()))
            filtered_df = df_player[df_player['NormalizedPlayer'].str.contains(normalized_searchterm, case=False, na=False)]
            suggestions = filtered_df['Player'].tolist()
            return suggestions
        else:
            return []

    selected_value = st_searchbox(
        search_csv,
        key="csv_searchbox",
        placeholder="🔍 Search a Football Player - CSV version"
    )

    st.session_state.choice = selected_value
    choice = st.session_state.choice

    ################### Organic result ###########################
    if choice:
        
        # Extract column names from the JSON result
        columns_to_process = list(df_player.columns)

        # Create a normalized copy of the player DataFrame
        df_player_norm = df_player.copy()

        # Define a custom mapping for the 'Pos' column
        custom_mapping = {
            'GK': 1,
            'DF,FW': 4,
            'MF,FW': 8,
            'DF': 2,
            'DF,MF': 3,
            'MF,DF': 5,
            'MF': 6,
            'FW,DF': 7,
            'FW,MF': 9,
            'FW': 10
        }

        # Apply the custom mapping to the 'Pos' column
        df_player_norm['Pos'] = df_player_norm['Pos'].map(custom_mapping)

        # Select a subset of features for analysis
        selected_features = ['Pos', 'Age', 'Int',
        'Clr', 'KP', 'PPA', 'CrsPA', 'PrgP', 'Playing Time MP',
        'Performance Gls', 'Performance Ast', 'Performance G+A',
        'Performance G-PK', 'Performance Fls', 'Performance Fld',
        'Performance Crs', 'Performance Recov', 'Expected xG', 'Expected npxG', 'Expected xAG',
        'Expected xA', 'Expected A-xAG', 'Expected G-xG', 'Expected np:G-xG',
        'Progression PrgC', 'Progression PrgP', 'Progression PrgR',
        'Tackles Tkl', 'Tackles TklW', 'Tackles Def 3rd', 'Tackles Mid 3rd',
        'Tackles Att 3rd', 'Challenges Att', 'Challenges Tkl%',
        'Challenges Lost', 'Blocks Blocks', 'Blocks Sh', 'Blocks Pass',
        'Standard Sh', 'Standard SoT', 'Standard SoT%', 'Standard Sh/90', 'Standard Dist', 'Standard FK',
        'Performance GA', 'Performance SoTA', 'Performance Saves',
        'Performance Save%', 'Performance CS', 'Performance CS%',
        'Penalty Kicks PKatt', 'Penalty Kicks Save%', 'SCA SCA',
        'GCA GCA', 
        'Aerial Duels Won', 'Aerial Duels Lost', 'Aerial Duels Won%',
        'Total Cmp', 'Total Att', 'Total Cmp', 'Total TotDist',
        'Total PrgDist', '1/3'
        ]



        ####################### Cosine Similarity #######################################

        # Normalization using Min-Max scaling
        scaler = MinMaxScaler()
        df_player_norm[selected_features] = scaler.fit_transform(df_player_norm[selected_features])

        # Calculate cosine similarity between players based on selected features
        similarity = cosine_similarity(df_player_norm[selected_features])

        # Find the Rk associated with the selected player's name
        index_player = df_player.loc[df_player['Player'] == choice, 'Rk'].values[0]

        # Calculate similarity scores and sort them in descending order
        similarity_score = list(enumerate(similarity[index_player]))
        similar_players = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Create a list to store data of similar players
        similar_players_data = []

        # Loop to extract information from similar players
        for player in similar_players[1:11]:  # Exclude the first player (self)
            index = player[0]
            player_records = df_player[df_player['Rk'] == index]
            if not player_records.empty:
                player_data = player_records.iloc[0]  # Get the first row (there should be only one)
                similar_players_data.append(player_data)

        # Create a DataFrame from the data of similar players
        similar_players_df = pd.DataFrame(similar_players_data)

    ########################## Analytics of the player chosen ##########################
        url_player = bing_image_urls(choice+ " "+df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0]+" 2023", limit=1, )[0]

        with st.expander("Features of The Player selected - The data considered for analysis pertains to the period of 2023 - 2024.", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(choice)
                st.image(url_player, width=356)

            with col2:
                st.caption("📄 Information of Player")
                col_1, col_2, col_3 = st.columns(3)

                with col_1:
                    st.metric("Nation", df_player.loc[df_player['Player'] == choice, 'Nation'].iloc[0], None)
                    st.metric("Position", df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0], None)

                with col_2:
                    st.metric("Born", df_player.loc[df_player['Player'] == choice, 'Born'].iloc[0], None)
                    st.metric("Match Played", df_player.loc[df_player['Player'] == choice, 'Playing Time MP'].iloc[0], None, help="In 2022/2023")

                with col_3:
                    st.metric("Age", df_player.loc[df_player['Player'] == choice, 'Age'].iloc[0], None)

                st.metric(f"🏆 League: {df_player.loc[df_player['Player'] == choice, 'Comp'].iloc[0]}", df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0], None)

            with col3:
                st.caption("⚽ Information target of Player")
                # GK
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "GK":
                        col_1, col_2 = st.columns(2)

                        with col_1:
                            st.metric("Saves", df_player.loc[df_player['Player'] == choice, 'Performance Saves'].iloc[0], None, help="Total number of saves made by the goalkeeper.")
                            st.metric("Clean Sheet", df_player.loc[df_player['Player'] == choice, 'Performance CS'].iloc[0], None, help="Total number of clean sheets (matches without conceding goals) by the goalkeeper.")

                        with col_2:
                            st.metric("Goals Against", df_player.loc[df_player['Player'] == choice, 'Performance GA'].iloc[0], None, help="Total number of goals conceded by the goalkeeper.")
                            st.metric("ShoTA", df_player.loc[df_player['Player'] == choice, 'Performance SoTA'].iloc[0], None, help="Total number of shots on target faced by the goalkeeper.")

                # DF
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,FW":
                    col_1, col_2, col_3 = st.columns(3)

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the defender.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the defender.")

                    with col_2:
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the defender.")
                        st.metric("Tackle", df_player.loc[df_player['Player'] == choice, 'Tackles TklW'].iloc[0], None, help="Total number of successful tackles made by the defender in 2022/2023.")

                    with col_3:
                        st.metric("Interception", df_player.loc[df_player['Player'] == choice, 'Int'].iloc[0], None, help="Total number of interceptions made by the defender.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes made by the defender.")

                # MF
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,FW":
                    col_1, col_2, col_3 = st.columns(3)

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                    with col_2:
                        st.metric("GCA", df_player.loc[df_player['Player'] == choice, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                        st.metric("Progressive PrgP", df_player.loc[df_player['Player'] == choice, 'Progression PrgP'].iloc[0], None, help="Total number of progressive passes by the player.")

                    with col_3:
                        st.metric("SCA", df_player.loc[df_player['Player'] == choice, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                # FW
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,DF":
                    col_1, col_2, col_3 = st.columns(3) 

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                    with col_2:
                        st.metric("SCA", df_player.loc[df_player['Player'] == choice, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                        st.metric("xG", df_player.loc[df_player['Player'] == choice, 'Expected xG'].iloc[0], None, help="Expected goals (xG) by the player.")
                        st.metric("xAG", df_player.loc[df_player['Player'] == choice, 'Expected xAG'].iloc[0], None, help="Expected assists (xAG) by the player.")

                    with col_3:
                        st.metric("GCA", df_player.loc[df_player['Player'] == choice, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                                
                        
        ################# Radar and Rank ######################### 
        col1, col2 = st.columns([1.2, 2])

        with col1:
            ###### Similar Players Component ###############
            st.subheader(f'Similar Players to {choice}')
            st.caption("The ranking presented here is generated using a model based on **Cosine Similarity**. It's important to recognize that, as a ranking, the results are inherently subjective.")
            selected_columns = ["Player", "Nation", "Squad", "Pos", "Age"]
            st.dataframe(similar_players_df[selected_columns], hide_index=True, use_container_width=True)

        with col2:
            ###### Radar Analytics #########################
            categories = ['Performance Gls', 'Performance Ast', 'KP', 'GCA GCA','Aerial Duels Won', 'Int', 'Tackles TklW', 'Performance Saves', 'Performance CS', 'Performance GA','Performance SoTA']
            selected_players = similar_players_df.head(10)

            fig = go.Figure()

            for index, player_row in selected_players.iterrows():
                player_name = player_row['Player']
                values = [player_row[col] for col in categories]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=player_name
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )
                ),
                showlegend=True,  
                legend=dict(
                    orientation="v", 
                    yanchor="top",  
                    y=1,  
                    xanchor="left",  
                    x=1.02,  
                ),
                width=750,  
                height=520  
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.write(" ")
    if choice:
        load_dotenv()

        # Configure the Google Generative AI client
        api_key = st.text_input("You need to enter the Gemini AI API Key:", placeholder="sk-...", type="password")
        genai.configure(api_key=api_key)

        # Load the CSV data
        df = pd.read_csv('data/football-player-stats-2023-COMPLETE.csv')

        # Streamlit App
        st.title("Football Player Scouting Report")

        # Player selection
        player_name = choice

        # Extract data for the selected player
        player_data = df[df['Player'] == player_name].iloc[0]

        # Calculate player's age
        birthday = datetime.strptime(str(player_data['Born']), '%Y')
        age = (datetime.now() - birthday).days // 365
        # Construct the scouting report prompt
        prompt = f"""
        I need you to create a scouting report on {player_name}. Can you provide me with a summary of their strengths and weaknesses?

        Here is the data I have on him:

        Player: {player_name}
        Position: {player_data['Pos']}
        Age: {age}
        Team: {player_data['Squad']}

        | Statistic                | Value               |
        |--------------------------|---------------------|
        | Playing Time MP          | {player_data['Playing Time MP']}     |
        | Playing Time Starts      | {player_data['Playing Time Starts']} |
        | Playing Time Min         | {player_data['Playing Time Min']}    |
        | Playing Time 90s         | {player_data['Playing Time 90s']}    |
        | Performance Gls          | {player_data['Performance Gls']}     |
        | Performance Ast          | {player_data['Performance Ast']}     |
        | Performance G+A          | {player_data['Performance G+A']}     |
        | Performance G-PK         | {player_data['Performance G-PK']}    |
        | Performance CrdY         | {player_data['Performance CrdY']}    |
        | Performance CrdR         | {player_data['Performance CrdR']}    |
        | Performance Fls          | {player_data['Performance Fls']}     |
        | Performance Fld          | {player_data['Performance Fld']}     |
        | Performance Off          | {player_data['Performance Off']}     |
        | Performance Crs          | {player_data['Performance Crs']}     |
        | Performance Int          | {player_data['Performance Int']}     |
        | Performance OG           | {player_data['Performance OG']}      |
        | Performance Recov        | {player_data['Performance Recov']}   |
        | Expected xG              | {player_data['Expected xG']}         |
        | Expected npxG            | {player_data['Expected npxG']}       |
        | Expected xAG             | {player_data['Expected xAG']}        |
        | Expected npxG+xAG        | {player_data['Expected npxG+xAG']}   |
        | Progression PrgC         | {player_data['Progression PrgC']}    |
        | Progression PrgP         | {player_data['Progression PrgP']}    |
        | Progression PrgR         | {player_data['Progression PrgR']}    |
        | Per 90 Minutes Gls       | {player_data['Per 90 Minutes Gls']}  |
        | Per 90 Minutes Ast       | {player_data['Per 90 Minutes Ast']}  |
        | Per 90 Minutes G+A       | {player_data['Per 90 Minutes G+A']}  |
        | Per 90 Minutes G-PK      | {player_data['Per 90 Minutes G-PK']} |
        | Per 90 Minutes xG        | {player_data['Per 90 Minutes xG']}   |
        | Per 90 Minutes xAG       | {player_data['Per 90 Minutes xAG']}  |
        | Per 90 Minutes npxG      | {player_data['Per 90 Minutes npxG']} |
        | Per 90 Minutes npxG+xAG  | {player_data['Per 90 Minutes npxG+xAG']} |

        Return the scouting report in the following markdown format:

        # Scouting Report for {player_name}

        ## Strengths
        < a list of 1 to 3 strengths >

        ## Weaknesses
        < a list of 1 to 3 weaknesses >

        ## Summary
        < a brief summary of the player's overall performance and if he would be beneficial to the team >
        """

    # Generate scouting report using Google Generative AI
    if choice:
        if st.button("Generate Scouting Report"):
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            # create_stats(player_name=player_name, df= df)
            # Extract the text content from the response
            report = response.text.strip()
            # Display the generated report in a formatted manner
            st.markdown(report)


def analysis():
    ############### Header #################
    # Set the page width to 'wide' to occupy the full width
    st.markdown("<h1 style='text-align: center;'>⚽🔍 Player Scouting Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>🏆 Find the Best Player for Your Team</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Powered Player Selection 2024-2025: Tailor Your Team's Lineup with Google Gemini API - 👀 This is a Beta version</p>", unsafe_allow_html=True)
    # st.markdown("<p>Our team optimization feature helps you find the best players for your team based on your preferred game style, player type, league, formation, and required skills. Get personalized recommendations to elevate your team's performance.</p>", unsafe_allow_html=True)


    ############### Simple Search Engine with Auto-Complete Query Suggestion ##############
    press = False
    choice = None

    # Initialises the streamlit session state useful for page reloading
    if 'expanded' not in st.session_state:
        st.session_state.expanded = True

    if 'choice' not in st.session_state:
        st.session_state.choice = None

    # Carica i dati dal file CSV
    df_player = pd.read_csv('data/football-player-stats-2023.csv')


    def remove_accents(text: str) -> str:
        return unidecode.unidecode(text)

    def search_csv(searchterm: str) -> List[str]:
        if searchterm:
            normalized_searchterm = remove_accents(searchterm.lower())
            df_player['NormalizedPlayer'] = df_player['Player'].apply(lambda x: remove_accents(x.lower()))
            filtered_df = df_player[df_player['NormalizedPlayer'].str.contains(normalized_searchterm, case=False, na=False)]
            suggestions = filtered_df['Player'].tolist()
            return suggestions
        else:
            return []

    selected_value = st_searchbox(
        search_csv,
        key="csv_searchbox",
        placeholder="🔍 Search a Football Player - CSV version"
    )
    st.write("Our team optimization feature helps you find the best players for your team based on your preferred game style, player type, league, formation, and required skills. Get personalized recommendations to elevate your team's performance.")
    st.session_state.choice = selected_value
    choice = st.session_state.choice

    ################### Organic result ###########################
    if choice:
        
        # Extract column names from the JSON result
        columns_to_process = list(df_player.columns)

        # Create a normalized copy of the player DataFrame
        df_player_norm = df_player.copy()

        # Define a custom mapping for the 'Pos' column
        custom_mapping = {
            'GK': 1,
            'DF,FW': 4,
            'MF,FW': 8,
            'DF': 2,
            'DF,MF': 3,
            'MF,DF': 5,
            'MF': 6,
            'FW,DF': 7,
            'FW,MF': 9,
            'FW': 10
        }

        # Apply the custom mapping to the 'Pos' column
        df_player_norm['Pos'] = df_player_norm['Pos'].map(custom_mapping)

        # Select a subset of features for analysis
        selected_features = ['Pos', 'Age', 'Int',
        'Clr', 'KP', 'PPA', 'CrsPA', 'PrgP', 'Playing Time MP',
        'Performance Gls', 'Performance Ast', 'Performance G+A',
        'Performance G-PK', 'Performance Fls', 'Performance Fld',
        'Performance Crs', 'Performance Recov', 'Expected xG', 'Expected npxG', 'Expected xAG',
        'Expected xA', 'Expected A-xAG', 'Expected G-xG', 'Expected np:G-xG',
        'Progression PrgC', 'Progression PrgP', 'Progression PrgR',
        'Tackles Tkl', 'Tackles TklW', 'Tackles Def 3rd', 'Tackles Mid 3rd',
        'Tackles Att 3rd', 'Challenges Att', 'Challenges Tkl%',
        'Challenges Lost', 'Blocks Blocks', 'Blocks Sh', 'Blocks Pass',
        'Standard Sh', 'Standard SoT', 'Standard SoT%', 'Standard Sh/90', 'Standard Dist', 'Standard FK',
        'Performance GA', 'Performance SoTA', 'Performance Saves',
        'Performance Save%', 'Performance CS', 'Performance CS%',
        'Penalty Kicks PKatt', 'Penalty Kicks Save%', 'SCA SCA',
        'GCA GCA', 
        'Aerial Duels Won', 'Aerial Duels Lost', 'Aerial Duels Won%',
        'Total Cmp', 'Total Att', 'Total Cmp', 'Total TotDist',
        'Total PrgDist', '1/3'
        ]



        ####################### Cosine Similarity #######################################

        # Normalization using Min-Max scaling
        scaler = MinMaxScaler()
        df_player_norm[selected_features] = scaler.fit_transform(df_player_norm[selected_features])

        # Calculate cosine similarity between players based on selected features
        similarity = cosine_similarity(df_player_norm[selected_features])

        # Find the Rk associated with the selected player's name
        index_player = df_player.loc[df_player['Player'] == choice, 'Rk'].values[0]

        # Calculate similarity scores and sort them in descending order
        similarity_score = list(enumerate(similarity[index_player]))
        similar_players = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Create a list to store data of similar players
        similar_players_data = []

        # Loop to extract information from similar players
        for player in similar_players[1:11]:  # Exclude the first player (self)
            index = player[0]
            player_records = df_player[df_player['Rk'] == index]
            if not player_records.empty:
                player_data = player_records.iloc[0]  # Get the first row (there should be only one)
                similar_players_data.append(player_data)

        # Create a DataFrame from the data of similar players
        similar_players_df = pd.DataFrame(similar_players_data)

    ########################## Analytics of the player chosen ##########################
        url_player = bing_image_urls(choice+ " "+df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0]+" 2023", limit=1, )[0]

        with st.expander("Features of The Player selected - The data considered for analysis pertains to the period of 2022 - 2023.", expanded=True):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(choice)
                st.image(url_player, width=356)

            with col2:
                st.caption("📄 Information of Player")
                col_1, col_2, col_3 = st.columns(3)

                with col_1:
                    st.metric("Nation", df_player.loc[df_player['Player'] == choice, 'Nation'].iloc[0], None)
                    st.metric("Position", df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0], None)

                with col_2:
                    st.metric("Born", df_player.loc[df_player['Player'] == choice, 'Born'].iloc[0], None)
                    st.metric("Match Played", df_player.loc[df_player['Player'] == choice, 'Playing Time MP'].iloc[0], None, help="In 2022/2023")

                with col_3:
                    st.metric("Age", df_player.loc[df_player['Player'] == choice, 'Age'].iloc[0], None)

                st.metric(f"🏆 League: {df_player.loc[df_player['Player'] == choice, 'Comp'].iloc[0]}", df_player.loc[df_player['Player'] == choice, 'Squad'].iloc[0], None)

            with col3:
                st.caption("⚽ Information target of Player")
                # GK
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "GK":
                        col_1, col_2 = st.columns(2)

                        with col_1:
                            st.metric("Saves", df_player.loc[df_player['Player'] == choice, 'Performance Saves'].iloc[0], None, help="Total number of saves made by the goalkeeper.")
                            st.metric("Clean Sheet", df_player.loc[df_player['Player'] == choice, 'Performance CS'].iloc[0], None, help="Total number of clean sheets (matches without conceding goals) by the goalkeeper.")

                        with col_2:
                            st.metric("Goals Against", df_player.loc[df_player['Player'] == choice, 'Performance GA'].iloc[0], None, help="Total number of goals conceded by the goalkeeper.")
                            st.metric("ShoTA", df_player.loc[df_player['Player'] == choice, 'Performance SoTA'].iloc[0], None, help="Total number of shots on target faced by the goalkeeper.")

                # DF
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "DF,FW":
                    col_1, col_2, col_3 = st.columns(3)

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the defender.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the defender.")

                    with col_2:
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the defender.")
                        st.metric("Tackle", df_player.loc[df_player['Player'] == choice, 'Tackles TklW'].iloc[0], None, help="Total number of successful tackles made by the defender in 2022/2023.")

                    with col_3:
                        st.metric("Interception", df_player.loc[df_player['Player'] == choice, 'Int'].iloc[0], None, help="Total number of interceptions made by the defender.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes made by the defender.")

                # MF
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "MF,FW":
                    col_1, col_2, col_3 = st.columns(3)

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                    with col_2:
                        st.metric("GCA", df_player.loc[df_player['Player'] == choice, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                        st.metric("Progressive PrgP", df_player.loc[df_player['Player'] == choice, 'Progression PrgP'].iloc[0], None, help="Total number of progressive passes by the player.")

                    with col_3:
                        st.metric("SCA", df_player.loc[df_player['Player'] == choice, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                # FW
                if df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['Player'] == choice, 'Pos'].iloc[0] == "FW,DF":
                    col_1, col_2, col_3 = st.columns(3) 

                    with col_1:
                        st.metric("Assist", df_player.loc[df_player['Player'] == choice, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                        st.metric("Goals", df_player.loc[df_player['Player'] == choice, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                        st.metric("Aerial Duel", df_player.loc[df_player['Player'] == choice, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                    with col_2:
                        st.metric("SCA", df_player.loc[df_player['Player'] == choice, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                        st.metric("xG", df_player.loc[df_player['Player'] == choice, 'Expected xG'].iloc[0], None, help="Expected goals (xG) by the player.")
                        st.metric("xAG", df_player.loc[df_player['Player'] == choice, 'Expected xAG'].iloc[0], None, help="Expected assists (xAG) by the player.")

                    with col_3:
                        st.metric("GCA", df_player.loc[df_player['Player'] == choice, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                        st.metric("Key Passage", df_player.loc[df_player['Player'] == choice, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                                
                        
        ################# Radar and Rank ######################### 
        col1, col2 = st.columns([1.2, 2])

        with col1:
            ###### Similar Players Component ###############
            st.subheader(f'Similar Players to {choice}')
            st.caption("This ranking list is determined through the application of a model based on **Cosine Similarity**. It should be noted that, being a ranking, the result obtained is inherently subjective.")
            selected_columns = ["Player", "Nation", "Squad", "Pos", "Age"]
            st.dataframe(similar_players_df[selected_columns], hide_index=True, use_container_width=True)

        with col2:
            ###### Radar Analytics #########################
            categories = ['Performance Gls', 'Performance Ast', 'KP', 'GCA GCA','Aerial Duels Won', 'Int', 'Tackles TklW', 'Performance Saves', 'Performance CS', 'Performance GA','Performance SoTA']
            selected_players = similar_players_df.head(10)

            fig = go.Figure()

            for index, player_row in selected_players.iterrows():
                player_name = player_row['Player']
                values = [player_row[col] for col in categories]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=player_name
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )
                ),
                showlegend=True,  
                legend=dict(
                    orientation="v", 
                    yanchor="top",  
                    y=1,  
                    xanchor="left",  
                    x=1.02,  
                ),
                width=750,  
                height=520  
            )

            st.plotly_chart(fig, use_container_width=True)
    
    ####################### Scouter AI Component ##################################

        dis = True
        st.header('⚽🕵️‍♂️ Scouter AI')
        choice = "Your chosen player"  # Assuming 'choice' is defined somewhere in your actual code
        message = f"Select the ideal characteristics for your team. Scouter AI will evaluate the most suitable player from the players most similar to **{choice}**"
        st.caption(message)

        api_key = st.text_input("You need to enter the Open AI API Key:", placeholder="sk-...", type="password")
        os.environ['GOOGLE_API_KEY'] = api_key

        if api_key:
            dis = False

        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            with st.form("my_form"):
                st.write("P R O M P T")
                # List of game styles and their descriptions
                game_styles = {
                    "Tiki-Taka": "This style of play, focuses on ball possession, control, and accurate passing.",
                    "Counter-Attack": "Teams adopting a counter-attacking style focus on solid defense and rapid advancement in attack when they regain possession of the ball.",
                    "High Press": "This style involves intense pressure on the opposing team from their half of the field. Teams practicing high pressing aim to quickly regain possession in the opponent's area, forcing mistakes under pressure.",
                    "Direct Play": "This style of play is more direct and relies on long and vertical passes, often targeting forwards or exploiting aerial play.",
                    "Pragmatic Possession": "Some teams aim to maintain ball possession as part of a defensive strategy, slowing down the game pace and limiting opponent opportunities.",
                    "Reactive": "In this style, a team adapts to the ongoing game situations, changing their tactics based on what is happening on the field. It can be used to exploit opponent weaknesses or respond to unexpected situations.",
                    "Physical and Defensive": "Some teams place greater emphasis on solid defense and physical play, aiming to frustrate opponents and limit their attacking opportunities.",
                    "Positional Play": "This style aims to dominate the midfield and create passing triangles to overcome opponent pressure. It is based on player positioning and the ability to maintain ball possession for strategic attacking.",
                    "Catenaccio": "This style, originating in Italy, focuses on defensive solidity and counterattacks. Catenaccio teams seek to minimize opponent scoring opportunities, often through zone defense and fast transition play.",
                    "Counter Attacking": "This style relies on solid defensive organization and quick transition to attack when the team regains possession of the ball. Forwards seek to exploit spaces left open by the opposing team during the defense-to-attack transition.",
                    "Long Ball": "This style involves frequent use of long and direct passes to bypass the opponent's defense. It relies on the physical strength of attackers and can be effective in aerial play situations."
                }

                # List of player experience levels
                player_experience = {
                    "Veteran": "A player with a long career and extensive experience in professional football. Often recognized for their wisdom and leadership on the field.",
                    "Experienced": "A player with experience, but not necessarily in the late stages of their career. They have solid skills and tactical knowledge acquired over time.",
                    "Young": "A player in the early or mid-career, often under 25 years old, with considerable development potential and a growing presence in professional football.",
                    "Promising": "A young talent with high potential but still needs to fully demonstrate their skills at the professional level."
                }

                # List of the leagues
                leagues = {
                    "Serie A": "Tactical and defensive football with an emphasis on defensive solidity and tactical play.",
                    "Ligue 1": "Open games with a high number of goals and a focus on discovering young talents.",
                    "Premier League": "Fast-paced, physical, and high-intensity play with a wide diversity of playing styles.",
                    "Bundesliga": "High-pressing approach and the development of young talents.",
                    "La Liga": "Possession of the ball and technical play with an emphasis on constructing actions."
                }

                # List of formations
                formations = ["4-3-1-2", "4-3-3", "3-5-2", "4-4-2", "3-4-3", "5-3-2", "4-2-3-1","4-3-2-1","3-4-1-2","3-4-2-1"]

                # List of player skills
                player_skills = [
                    "Key Passing", "Dribbling", "Speed", "Shooting", "Defending",
                    "Aerial Ability", "Tackling", "Vision", "Long Passing", "Agility", "Strength",
                    "Ball Control", "Positioning", "Finishing", "Crossing", "Marking",
                    "Work Rate", "Stamina", "Free Kicks", "Leadership","Penalty Saves","Reactiveness","Shot Stopping",
                    "Off the Ball Movement", "Teamwork", "Creativity", "Game Intelligence"
                ]

                ######### Inside FORM #####################
                st.subheader("Select a game style:")
                selected_game_style = st.selectbox("Choose a game style:", list(game_styles.keys()), disabled=dis)

                st.subheader("Select player type:")
                selected_player_experience = st.selectbox("Choose player type:", list(player_experience.keys()), disabled=dis)

                st.subheader("Select league:")
                selected_league = st.selectbox("Choose a league:", list(leagues.keys()), disabled=dis)

                st.subheader("Select formation:")
                selected_formation = st.selectbox("Choose a formation:", formations, disabled=dis)

                st.subheader("Select player skills:")
                selected_player_skills = st.multiselect("Choose player skills:", player_skills, disabled=dis)

                form = st.form_submit_button("➡️ Confirm features", disabled=dis)

        with col2:
            ######### Inside REPORT #####################
            st.info("The text is AI-generated by the Gemini-pro model. Please bear in mind that the accuracy and reliability of the content may differ. While the main aim is to offer general guidance in selecting a football player, it's advisable to cross-reference and validate any information from credible sources.", icon="ℹ️")

            if form:
                st.caption("Selected Options:")
                st.write(f"You have chosen a game style: {selected_game_style}. {game_styles[selected_game_style]} \
                This player must be {selected_player_experience} and have a good familiarity with the {selected_formation} and the skills of: {', '.join(selected_player_skills)}.")

                system_template = (
                    """You are a soccer scout and you must be good at finding the best talents in your team starting from the players rated by the similar player system."""
                )

                human_template = """
                    Generate a Football Talent Scout report based on the DATA PROVIDED (maximum 250 words) written in a formal tone FOLLOWING THE EXAMPLE.
                    It is essential to compare player attributes and select the most suitable candidate from the available options from among similar players, based on the TEAM REQUIREMENTS provided. It is important to note that the selection of players is not limited to the ranking of the players provided, as long as they meet the TEAM REQUIREMENTS.
                    THE PLAYER CHOSEN MUST NECESSARILY BE AMONG THE POSSIBLE PLAYERS CONSIDERED IN THE FOOTBALL SCOUT REPORT.
                    INDICATE the player chosen at the end of the REPORT.

                    DATA:
                    ------------------------------------
                    {content}
                    ------------------------------------ 

                    TEAM REQUIREMENTS:
                    Style of play: {style_t}
                    Player type required: {type_player}
                    Preferred league: {league}
                    Key ability: {ability}
                    Ideal formation: {formation}

                    EXAMPLE TO FOLLOW:
                    ### Report
                    After a detailed analysis of the data, we have identified candidates who best meet the requirements of your team. Below, we present three potential candidates:

                    ##### Three potential candidates:

                    **Player X**: Highlights strengths and addresses weaknesses based on data on the essential attributes for a player in his specific age group.
                    **Player Y**: Highlights strengths and addresses weaknesses based on data regarding the attributes a player must necessarily possess in his specific age group.
                    **Player Z**: Highlighting strengths and addressing weaknesses based on attribute data that a player must necessarily possess in his specific age group.
                    
                    [Provide the reasons for choosing the recommended player over the others].
                    
                    The recommended player: **Name of player recommended**.
                    
                    follow these exact same format as said sepcially the recommended player must be in between **name of the palyer**
                    """

                st.caption("Text generated by Scouter AI:")
                with st.spinner("Generating text. Please wait..."):
                    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
                    chat_prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(system_template),
                        HumanMessagePromptTemplate.from_template(human_template)
                    ])
                    result = llm.invoke(
                        [
                            SystemMessage(content=system_template),
                            HumanMessage(content=human_template.format(
                                content=similar_players_df.to_string(),
                                style_t=game_styles[selected_game_style],
                                type_player=player_experience[selected_player_experience],
                                league=leagues[selected_league],
                                ability=", ".join(selected_player_skills),
                                formation=selected_formation
                            ))
                        ]
                    )
                    print(result.content)
                    st.markdown(result.content)


                # Use a regular expression to find the name after "The recommended player: "
                pattern = r"The recommended player:\s*([^:]+)"

                # find the correspondence in the entire text
                matches = re.findall(pattern, result.content, re.IGNORECASE)
                if matches:
                    ultimo_nome = matches[0].rstrip('.')  # remove extra dot
                    if ultimo_nome.startswith('**') and ultimo_nome.endswith('**'):
                        ultimo_nome = ultimo_nome.strip('*')
                        # print(ultimo_nome)
        ####### Analytics of the recommended player ##############
        if form:  
            if matches:
                st.subheader("🌟 The features of the recommended player:")
                url_player = bing_image_urls(ultimo_nome+ " "+df_player.loc[df_player['Player'] == ultimo_nome, 'Squad'].iloc[0]+" 2023", limit=1, )[0]

                with st.expander("Selected Player", expanded=True):

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.subheader(ultimo_nome)
                            st.image(url_player, width=356)

                        with col2:
                            st.caption("📄 Information of Player")
                            col_1, col_2, col_3 = st.columns(3)

                            with col_1:
                                st.metric("Nation", df_player.loc[df_player['Player'] == ultimo_nome, 'Nation'].iloc[0], None)
                                st.metric("Position", df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0], None)

                            with col_2:
                                st.metric("Born", df_player.loc[df_player['Player'] == ultimo_nome, 'Born'].iloc[0], None)
                                st.metric("Match Played", df_player.loc[df_player['Player'] == ultimo_nome, 'Playing Time MP'].iloc[0], None, help="In 2022/2023")

                            with col_3:
                                st.metric("Age", df_player.loc[df_player['Player'] == ultimo_nome, 'Age'].iloc[0], None)

                            st.metric(f"🏆 League: {df_player.loc[df_player['Player'] == ultimo_nome, 'Comp'].iloc[0]}", df_player.loc[df_player['Player'] == ultimo_nome, 'Squad'].iloc[0], None)

                        with col3:
                            st.caption("⚽ Information target of Player")
                            # GK
                            if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "GK":
                                    col_1, col_2 = st.columns(2)

                                    with col_1:
                                        st.metric("Saves", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Saves'].iloc[0], None, help="Total number of saves made by the goalkeeper.")
                                        st.metric("Clean Sheet", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance CS'].iloc[0], None, help="Total number of clean sheets (matches without conceding goals) by the goalkeeper.")

                                    with col_2:
                                        st.metric("Goals Against", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance GA'].iloc[0], None, help="Total number of goals conceded by the goalkeeper.")
                                        st.metric("ShoTA", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance SoTA'].iloc[0], None, help="Total number of shots on target faced by the goalkeeper.")

                            # DF
                            if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "DF,FW":
                                col_1, col_2, col_3 = st.columns(3)

                                with col_1:
                                    st.metric("Assist", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the defender.")
                                    st.metric("Goals", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the defender.")

                                with col_2:
                                    st.metric("Aerial Duel", df_player.loc[df_player['Player'] == ultimo_nome, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the defender.")
                                    st.metric("Tackle", df_player.loc[df_player['Player'] == ultimo_nome, 'Tackles TklW'].iloc[0], None, help="Total number of successful tackles made by the defender in 2022/2023.")

                                with col_3:
                                    st.metric("Interception", df_player.loc[df_player['Player'] == ultimo_nome, 'Int'].iloc[0], None, help="Total number of interceptions made by the defender.")
                                    st.metric("Key Passage", df_player.loc[df_player['Player'] == ultimo_nome, 'KP'].iloc[0], None, help="Total number of key passes made by the defender.")

                            # MF
                            if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "MF,FW":
                                col_1, col_2, col_3 = st.columns(3)

                                with col_1:
                                    st.metric("Assist", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                                    st.metric("Goals", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                                    st.metric("Aerial Duel", df_player.loc[df_player['Player'] == ultimo_nome, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                                with col_2:
                                    st.metric("GCA", df_player.loc[df_player['Player'] == ultimo_nome, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                                    st.metric("Progressive PrgP", df_player.loc[df_player['Player'] == ultimo_nome, 'Progression PrgP'].iloc[0], None, help="Total number of progressive passes by the player.")

                                with col_3:
                                    st.metric("SCA", df_player.loc[df_player['Player'] == ultimo_nome, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                                    st.metric("Key Passage", df_player.loc[df_player['Player'] == ultimo_nome, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                            # FW
                            if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "FW,DF":
                                col_1, col_2, col_3 = st.columns(3) 

                                with col_1:
                                    st.metric("Assist", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                                    st.metric("Goals", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                                    st.metric("Aerial Duel", df_player.loc[df_player['Player'] == ultimo_nome, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                                with col_2:
                                    st.metric("SCA", df_player.loc[df_player['Player'] == ultimo_nome, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                                    st.metric("xG", df_player.loc[df_player['Player'] == ultimo_nome, 'Expected xG'].iloc[0], None, help="Expected goals (xG) by the player.")
                                    st.metric("xAG", df_player.loc[df_player['Player'] == ultimo_nome, 'Expected xAG'].iloc[0], None, help="Expected assists (xAG) by the player.")

                                with col_3:
                                    st.metric("GCA", df_player.loc[df_player['Player'] == ultimo_nome, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                                    st.metric("Key Passage", df_player.loc[df_player['Player'] == ultimo_nome, 'KP'].iloc[0], None, help="Total number of key passes by the player.")


        st.write(" ")
        st.subheader("About")
        st.caption(
            "📚 The Player Scouting Recommendation System was developed for fun and my love for football. "
            "This project showcases my skills and creativity in applying advanced information retrieval techniques. "
            "🤝 Please note that this system is a conceptual idea designed as a decision support tool and is not intended to replace any football scout or coach."
        )

if __name__ == "__main__":
    main()
