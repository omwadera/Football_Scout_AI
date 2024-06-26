# ⚽🔍 Player Scouting Recommendation System

## 🔍 Key Features

### 1. Scout Player
Create AI-generated scouting reports for the current season. This feature provides detailed insights into player performance, strengths, weaknesses, and overall impact on the team.

<img src="IMAGES/player scout gif.gif" alt="Scout Player Feature" width="1080">

### 2. Team Builder
Get recommendations on which players can replace the current player based on various features like play style, league, skills, age, and more. The system identifies players with similar attributes and suggests potential replacements.

<img src="IMAGES/team builder gif.gif" alt="Player Scout Recommendations Feature" width="1080">

---

## 📊 Data Source
The data used in this project comes from various reliable sources, ensuring accuracy and relevance. The key datasets include:
- `data/football-player-stats-2023.csv`
- `data/football-player-stats-2023-COMPLETE.csv`

These datasets provide comprehensive statistics on football players, including performance metrics, historical data, and other relevant attributes.
Other than that, data is webscraped from FBREF and Transfer Market

---

## 💻 Technologies Used
The Player Scouting Recommendation System leverages the following technologies:
- **Python 3.7+**
- **Streamlit**: For creating the web application interface.
- **Pandas**: For data manipulation and analysis.
- **Plotly**: For interactive visualizations.
- **Scikit-learn**: For machine learning algorithms, including similarity analysis.
- **Requests**: For handling HTTP requests.
- **BeautifulSoup**: For web scraping.
- **Unidecode**: For text normalization.
- **Bing Image Search Python Library**: For fetching player images.
- **Google Generative AI Library**: For generating detailed scouting reports.
- **dotenv**: For managing environment variables.

---

## 🛠️ Methodological Workflow
1. **Data Collection**: Gathering player statistics and relevant information from multiple sources.
2. **Data Preprocessing**: Cleaning and formatting the data for analysis.
3. **Feature Engineering**: Creating new features to enhance the analysis.
4. **Similarity Analysis**: Using cosine similarity to find and compare players.
5. **Report Generation**: Utilizing Google Generative AI to create detailed scouting reports.
6. **Web Application**: Developing an interactive interface with Streamlit to present the data and insights.

### Flowchart

<img src="IMAGES/data preprocessing.png" alt="Player Scout" width="500">
<img src="IMAGES/Project flowchart.jpg" alt="Player Scout"width="1500">

---

## 🔍 Key Features

### Player Scout
- **Player Search**:
  - Enhances user experience by providing real-time search suggestions.

- **Player Profile**:
  - <img src="IMAGES/player profile.png" alt="Player Profile" width="800">
  - Displays detailed player information including images, basic statistics, and metrics.
  - Provides a comprehensive view of a player's current form and historical performance.

- **Scouting Report**:
  - <img src="IMAGES/scouting report.png" alt="Player Scout" width="800">
  - Generates a detailed scouting report using Google Generative AI.
  - Offers deep insights into a player's abilities, strengths, and weaknesses.

### Team Builder
- **Similarity Analysis**:
  - <img src="IMAGES/similarity function.png" alt="Player Scout" width="800">
  - Uses cosine similarity to find and display similar players.
  - Helps scouts identify players with comparable skills and potential.

- **Player Recomendations report using AI**:
  - <img src="IMAGES/Screenshot 2024-06-28 001650.png" alt="Player Scout" width="800">
  - Helps Coach to find the perfect replacement for a player in the summer transfer window based on custom features.


### Team evaluator
- **Coming Soon**: 
  - Additional features for team building and analysis.
  - Will enable users to create and analyze custom teams based on specific criteria.

---

## 📄 How to get Gemini 1.5 Pro API key?
Detailed documentation and architectural diagrams can be found in the [docs](https://ai.google.dev/gemini-api/docs/api-key). 
Steps to get Gemini API key:
1. Go to Gemini
2. Open the Account Settings page
3. Open the API settings page
4. Click on Create API key
5. Choose the Scope
6. Name the API keys
7. Save API key and API Secret
8. Set the Permissions

---

## 🚧 Limits
- **Data Accuracy**: The system's recommendations are only as good as the data provided. Ensuring up-to-date and accurate data is crucial.
- **AI Limitations**: The generative AI model's reports are based on patterns in the data and may not always capture the nuances of player performance.
- **Feature Scope**: Some features, such as the Team Builder, are still under development and may have limited functionality initially.

---

## 📚 Acknowledgments
We would like to thank the following for their contributions and support:
- **Data Providers**: For supplying the comprehensive datasets used in this project.
- **Open Source Community**: For the libraries and tools that made this project possible.

---

### Prerequisites
- Python 3.7 or higher
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- Requests
- BeautifulSoup
- Unidecode
- Bing Image Search Python Library
- Google Generative AI Library
- dotenv

