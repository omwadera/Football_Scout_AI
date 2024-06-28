# ⚽🔍 Player Scouting Recommendation System

## [Visit the Website](http://your_website_url)

---

## Demo
Check out the video demonstration of the application:

[![Demo Video](path_to_thumbnail_image)](path_to_video)

Here are some screenshots of the application in action:

### Home Page
![Home Page](path_to_home_page_image)

### Player Search
![Player Search](path_to_player_search_image)

### Player Profile
![Player Profile](path_to_player_profile_image)

---

## 📊 Data Source
The data used in this project comes from various reliable sources, ensuring accuracy and relevance. The key datasets include:
- `data/football-player-stats-2023.csv`
- `data/football-player-stats-2023-COMPLETE.csv`

These datasets provide comprehensive statistics on football players, including performance metrics, historical data, and other relevant attributes.

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
*Include your flowchart here.*

![Flowchart](https://github.com/omwadera/Football_Scout_AI/blob/3f9f421c269aa69981f5951931157774bd9c6c3d/IMAGES/Customer%20Journey%20-%20Project%20workflow%20(1).jpg)

---

## 🔍 Key Features

### Player Scout
- **Player Search**: 
  - Allows users to search for players using an auto-complete search box.
  - Enhances user experience by providing real-time search suggestions.

- **Player Profile**: 
  - Displays detailed player information including images, basic statistics, and metrics.
  - Provides a comprehensive view of a player's current form and historical performance.

- **Similarity Analysis**: 
  - Uses cosine similarity to find and display similar players.
  - Helps scouts identify players with comparable skills and potential.

- **Scouting Report**: 
  - Generates a detailed scouting report using Google Generative AI.
  - Offers deep insights into a player's abilities, strengths, and weaknesses.

### Team Builder
- **Coming Soon**: 
  - Additional features for team building and analysis.
  - Will enable users to create and analyze custom teams based on specific criteria.

---

## 📄 Documentation and Complete Architecture
Detailed documentation and architectural diagrams can be found in the [docs](./docs) directory. This includes:
- System architecture overview.
- API documentation.
- Data flow diagrams.
- User guides and tutorials.

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
- **Team Members**: For their hard work and dedication in developing and refining the system.

---

## Setup Instructions

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

### Installation

#### Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
